import pandas as pd
from src.data.dataloader import DataLoader
from src.eval.helpers import get_results_for_evalset, evaluate_converse
from typing import TypeVar, Type

DataLoaderT = TypeVar("DataLoaderT", bound=DataLoader)


def df_metric_pct_change(df: pd.DataFrame, metric: str):
    """Calculate the percentage change of a metric"""
    df[f"{metric}_pct_change"] = 0.0
    for row in df[["model", "incontext_set", metric]].itertuples():
        baseline_metric = df[
            (df["model"] == row.model)
            & (df["incontext_set"] == row.incontext_set)
            & (df["num_examples"] == 0)
        ][metric].values[0]
        df.loc[row.Index, f"{metric}_pct_change"] = (
            (row[-1] - baseline_metric) / baseline_metric * 100
        )
    return df


def get_converse_results(
    dataset_name: str,
    dataset_prompt_loader: Type[DataLoaderT],
    subdirs: list[str],
    use_cached: bool = True,
):
    reference_data = dataset_prompt_loader().load_test_reference()

    converse_results = get_results_for_evalset(
        dataset_name,
        reference_data,
        evaluate_converse,
        subdirs,
        use_cached=use_cached,
    )

    # Flatten columns: history_length_X into {num_examples X, ...metrics}
    results = []
    for i, res in enumerate(converse_results):
        dataset_keys = [k for k in res.keys() if not k.startswith("history_length")]
        history_length_keys = [k for k in res.keys() if k.startswith("history_length")]
        for hist_key in history_length_keys:
            num_examples = int(hist_key.split("_")[-1])
            d = {k: res[k] for k in dataset_keys}
            d["num_examples"] = num_examples
            d |= res[hist_key]
            results.append(d)

    df = pd.DataFrame.from_records(results)
    # rt_converse_results_df = pd.DataFrame.from_records(rt_converse_results)

    # Calculate percentage change
    df["acc"] = df["matches"] / df["total"]
    df = df_metric_pct_change(df, "acc")

    print(len(df))
    return df
