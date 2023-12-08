def template(eval_args, eval_sample, incontext_examples=[]):
    return f"{incontext_prompt_manager(eval_args, incontext_examples)}\n{eval_template(eval_args, eval_sample)}"


def incontext_prompt_manager(eval_args, examples):
    if len(examples) == 0:
        return ''
    if eval_args.incontext_data_name == "rt":
        out = "Please read the following pairs of movie reviews and sentiment:\n"
        for i,example in enumerate(examples):
            out+= str(i) + f". <Review>{example['Review']}</Review> <Sentiment>{example['Sentiment']}</Sentiment>\n"
    
    if eval_args.incontext_data_name == 'gigaword':
        out = "Please read the following pairs of texts and summaries:\n"
        for i,example in enumerate(examples):
            out+= str(i) + f". <Text>{example['Text']}</Text> <Summary>{example['Summary']}</Summary>\n"

    return out


def eval_template(eval_args, sample):
    if eval_args.eval_data_name == "rt":
        out = f"Please perform a Sentiment Classification task. Given the following movie review, assign a sentiment label from [`negative', `positive']. Please make sure to provide a reason for your sentiment label and then provide your final answer in the following format: <Sentiment> negative / positive </Sentiment>. It is very important that you provide the final sentiment within the tags <Sentiment> negative / positive </Sentiment> otherwise your answer will not be processed.\n <Review>{sample['Review']}</Review>"
    
    if eval_args.eval_data_name == "gigaword":
        out = f"Please summarize the following text.\n <Text>{sample['Text']}</Text>"

    return out
    