def incontext_prompt_manager(args, examples):
    if args.data_name == "rt":
        out = "Please read the following pairs of movie reviews and sentiment:\n"
        for i,example in enumerate(examples):
            out+= str(i) + f". [Review] {example['Review']} [Sentiment] {example['Sentiment']}\n"
    return out

def eval_template(args, sample):
    if args.data_name == "rt":
        out = f"Please perform a Sentiment Classification task. Given the following movie review, assign a sentiment label from [`negative', `positive']. Please make sure to provide a reason for your sentiment label and then provide your final answer in the following format: <Sentiment> negative / positive </Sentiment>. It is very important that you provide the final sentiment within the tags <Sentiment> negative / positive </Sentiment> otherwise your answer will not be processed.\n [Review] " + sample['Review']
    return out
    