from densevid_eval3.evaluate2018 import main as eval2018
from densevid_eval3.evaluate2021 import main as eval2021
from densevid_eval3.evaluate2018_cider import main as eval2018_cider

def eval_dvc(json_path, reference, no_lang_eval=False, topN=1000, version='2018', verbose=False):
    args = type('args', (object,), {})()
    args.submission = json_path
    args.max_proposals_per_video = topN
    args.tious = [0.3,0.5,0.7,0.9]
    args.verbose = verbose
    args.no_lang_eval = no_lang_eval
    args.references = reference
    if version=='2018':
        eval_func = eval2018
    elif version == '2021':
        eval_func = eval2021
    elif version == '2018_cider':
        eval_func = eval2018_cider
        args.verbose = True
        args.tious = [0.9]
    score = eval_func(args)
    return score

if __name__ == '__main__':
    p = '../save/pretrained_models/anet_c3d_pdvc/2021-08-21-23-40-05_debug_2021-08-21_20-46-20_epoch8_num4917_score0.json.top3.json'
    ref = ['../data/anet/captiondata/val_1.json', '../data/anet/captiondata/val_2.json']
    score = eval_dvc(p, ref, no_lang_eval=False, version='2018')
    print(score)