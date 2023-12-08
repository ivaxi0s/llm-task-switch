import json
import re

def evaluate(pred_fpath, ref_data, ref_data_name):
    '''
        pred_fpath: contains model predictions as a list (of dict)
        ref_data: test data with reference labels
        ref_data_name: specifies the reference dataset name (influences evaluation metric)
    '''
    # load predictions from cache
    with open(pred_fpath, 'r') as f:
        pred_data = json.load(f)
    
    if ref_data_name == 'rt':
        return eval_rt(pred_data, ref_data)

def eval_rt(pred_data, ref_data):
    '''
        Accuracy
    '''
    matches = 0
    failed = 0
    for pred, ref in zip(pred_data, ref_data):
        pred = pred.lower()
        pred_sent = re.findall("<sentiment>(.*?)</sentiment>", pred, re.DOTALL)
        if len(pred_sent) != 1:
            failed +=1
            continue
        pred_sent = pred_sent[0]
        pred_sent = pred_sent.strip(' ')
        if pred_sent == ref['Sentiment']:
            matches +=1
    print(f'Generation format failed for {failed/len(pred_data)*100}% samples')
    return {'Accuracy':f'{matches/(len(pred_data)-failed)*100}%'}
