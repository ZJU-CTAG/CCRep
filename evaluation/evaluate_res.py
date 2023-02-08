
import os

from base_global import py_intepreter_path
from evaluation.pycocoevalcap.meteor.meteor import Meteor
from evaluation.pycocoevalcap.rouge.rouge import Rouge


def meteor_rouge(hyp, ref):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        res = {k: [v.strip().lower()] for k, v in enumerate(hypothesis)}
    with open(ref, 'r') as r:
        references = r.readlines()
        tgt = {k: [v.strip().lower()] for k, v in enumerate(references)}

    score_Meteor, scores_Meteor = Meteor().compute_score(tgt, res)
    print("Meteor: %s" % score_Meteor)

    score_Rouge, scores_Rouge = Rouge().compute_score(tgt, res)
    print("ROUGE: %s" % score_Rouge)

def eval_cmg_corec(pred_path, ref_path):
    B_Norm_script_path = f'./evaluation/B-Norm.py'
    results = os.popen(f'{py_intepreter_path} {B_Norm_script_path} {ref_path} < {pred_path}')
    print(results.read())
    print('*'*50)
    meteor_rouge(pred_path, ref_path)

def eval_cmg_fira(pred_path, ref_path):
    B_Norm_script_path = f'./evaluation/fira_metrics/Bleu-B-Norm.py'
    # B_Norm_penalty_script_path = f'./evaluation/fira_metrics/Bleu-Penalty.py'
    Meteor_script_path = f'./evaluation/fira_metrics/Meteor.py'
    Rouge_script_path = f'./evaluation/fira_metrics/Rouge.py'

    print('\nB-Norm:')
    results = os.popen(f'{py_intepreter_path} {B_Norm_script_path} {ref_path} < {pred_path}')
    print(results.read())
    print('\nMeteor:')
    results = os.popen(f'{py_intepreter_path} {Meteor_script_path} -r {ref_path} -g {pred_path}')
    print(results.read())
    print('\nRouge:')
    results = os.popen(f'{py_intepreter_path} {Rouge_script_path} -r {ref_path} -g {pred_path}')
    print(results.read())
    # print('\nBleu-Penalty:')
    # os.system(f'{py_intepreter_path} {B_Norm_penalty_script_path} {ref_path} < {pred_path}')

