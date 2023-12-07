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
    for pred, ref in zip(pred_data, ref_data):
        if re.findall("<Sentiment>(.*?)</Sentiment>", pred, re.DOTALL) == ref['Sentiment']:
            matches +=1
    return {'Accuracy':f'{matches/len(pred_data)*100}%'}
