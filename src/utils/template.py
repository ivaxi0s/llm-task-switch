"""Generate the templates from the prompt and evaluation

NOTE: This is deprecated. Use PromptLoader instead (from data/dataloader.py)
"""


def template(eval_args, eval_sample, incontext_examples=[]):
    return f"{incontext_prompt_manager(eval_args, incontext_examples)}\n{eval_template(eval_args, eval_sample)}"


def incontext_prompt_manager(eval_args, examples):
    """This should be equivalent to the <ct> tag"""
    if len(examples) == 0:
        return ""
    if eval_args.incontext_data_name == "rt":
        out = "Please read the following pairs of movie reviews and sentiment:\n"
        for i, example in enumerate(examples):
            out += (
                # f"{str(i)}. "
                # + f"<Review>{example['Review']}</Review> <Sentiment>{example['Sentiment']}</Sentiment>\n"
                f"review: {example['Review']}\nsentiment: {example['Sentiment']}\n"
            )

    if eval_args.incontext_data_name == "gigaword":
        out = "Please read the following pairs of texts and summaries:\n"
        for i, example in enumerate(examples):
            out += (
                # str(i)
                # + f". <Text>{example['Text']}</Text> <Summary>{example['Summary']}</Summary>\n"
                f"article: {example['Text']}\nsummary: {example['Summary']}\n"
            )

    return out


def eval_template(eval_args, sample):
    """This is 'Eval Prompt Template' *without* the <ct> tag"""
    if eval_args.eval_data_name == "rt":
        # out = f"Please perform a Sentiment Classification task. Given the following movie review, assign a sentiment label from [`negative', `positive']. Please make sure to provide a reason for your sentiment label and then provide your final answer in the following format: <Sentiment> negative / positive </Sentiment>. It is very important that you provide the final sentiment within the tags <Sentiment> negative / positive </Sentiment> otherwise your answer will not be processed.\n <Review>{sample['Review']}</Review>"
        out = (
            "Please perform a Sentiment Classification task. "
            "Given the following movie review, assign a sentiment label from ['negative', 'positive']. "
            "Return only the sentiment label without any other text.\n"
            f"review: {sample['Review']}\n"
            "sentiment:"
        )

    if eval_args.eval_data_name == "gigaword":
        out = f"Please summarize the following text.\n{sample['Text']}"

    return out
