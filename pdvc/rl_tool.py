import numpy as np
import sys
import os
from collections import OrderedDict
from os.path import dirname, abspath

PROJECT_ROOT_PATH='/apdcephfs_cq2/share_1367250/wybertwang/project/VideoDETR_jinrui'

pdvc_root_dir = dirname(dirname(abspath(__file__)))

for pdvc_dir in [PROJECT_ROOT_PATH, pdvc_root_dir]:
    sys.path.insert(0, pdvc_dir)
    sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
    sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))
    sys.path.append(os.path.join(pdvc_dir, "densevid_eval3/cider"))
    try:
        from pyciderevalcap.ciderD.ciderD import CiderD
    except:
        print('cider or coco-caption missing')

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu

def init_scorer(types=None, cached_tokens=None):
    if types == None:
        types  =['Meteor', 'CiderD']
    scorers = {}
    for type_ in types:
        if type_ in ['CiderD']:
            scorer = eval(type_)(df = cached_tokens)
        else:
            scorer = eval(type_)()
        scorers[type_] = scorer
    return scorers
    # global Cider_scorer
    # Cider_scorer = Cider()

def array_to_str_para(arr):
    para = []
    for i, sub_arr in enumerate(arr):
        s = array_to_str(sub_arr)
        if i < len(arr):
            s = s.rstrip('0')
        para.append(s)
    return ' '.join(para)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_caption_reward(scorers, greedy_res, gt_captions, gen_result, score_weights, is_para=False):
    greedy_res = greedy_res.detach().cpu().numpy()
    gen_result = gen_result.detach().cpu().numpy()
    batch_size = len(gen_result)

    res = OrderedDict()
    for i in range(batch_size):
        if is_para:
            res[i] = [array_to_str_para(gen_result[i])]
        else:
            res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        if is_para:
            res[batch_size + i] = [array_to_str_para(greedy_res[i])]
        else:
            res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(gt_captions)):
        gts[i] = [array_to_str(gt_captions[i][1:])]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}

    scores = {}
    for name, scorer in scorers.items():
        if name in ['CiderD']:
            _, scores_ = scorer.compute_score(gts, res_)
        else:
            _, scores_ = scorer.compute_score(gts, res__)
        scores[name] = np.array(scores_)

    scores = np.sum([score_weights[name] * scores[name] for name in scorers.keys()], 0)
    rewards = scores[:batch_size] - scores[batch_size:]

    return rewards, scores[:batch_size], scores[batch_size:]