import json
from pathlib import Path
from typing import Callable
from src.eval.eval_metric import evaluate, evaluate_converse


def get_results_from_folder(
    folder: Path,
    all_reference_data: list[str],
    eval_func: Callable,
    subdir: str = "iterative",
    use_cached: bool = True,
    verbose=False,
):
    """Get results from a folder containing prediction files.

    This function is a wrapper on top of evaluate / evaluate_converse
    """
    assert eval_func in (evaluate, evaluate_converse)

    pred_file = Path(folder) / subdir / "predictions.json"
    if not pred_file.is_file():
        if verbose:
            print(f"Warning: predictions.json not found in {str(folder)}")
        return {}

    # sort reference data according to eval_idxs
    eval_idxs_file = Path(folder) / subdir / "eval_idxs.json"
    if eval_idxs_file.is_file():
        ref_data = [all_reference_data[i] for i in json.load(open(eval_idxs_file, "r"))]
    else:
        ref_data = all_reference_data

    # Calculate metrics
    return eval_func(pred_file, ref_data, use_cached=use_cached)


def get_results_for_evalset(
    eval_set: str,
    all_ref_data: list[str],
    eval_func: Callable,
    subdirs: list[str] = ["iterative"],
    model_folder_names=["mistral-7b", "llama-7b", "gpt3.5", "gpt4"],
    experiment_path=Path("../experiments/"),
    use_cached: bool = True,
):
    """Ger results for a specific evaluation set

    This is a wrapper on top of get_results_from_folder
    """

    if eval_func not in (evaluate, evaluate_converse):
        raise ValueError("Unknown evaluation function")

    results = []
    for model_folder in model_folder_names:
        eval_folder = Path(experiment_path) / model_folder / f"eval_data_{eval_set}"
        if not eval_folder.is_dir():
            print(f"Model: {model_folder} is missing eval set {str(eval_folder)}")
            continue

        # extract results for each in context set
        for in_ctxt_folder in eval_folder.iterdir():
            res = []
            for f in in_ctxt_folder.iterdir():
                for subdir in subdirs:
                    try:
                        res.append(
                            get_results_from_folder(
                                f,
                                all_ref_data,
                                eval_func,
                                subdir,
                                use_cached=use_cached,
                            )
                        )
                    except Exception as e:
                        print(f"Error processing {str(f)}")
                        print(e)

            # res = joblib.Parallel(n_jobs=1)(
            #     joblib.delayed(get_results)(f, all_ref_data)
            #     for f in in_ctxt_folder.iterdir()
            # )
            # Remove empty results
            res = [r for r in res if r]
            results.extend(res)
    return results
