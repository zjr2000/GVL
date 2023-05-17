from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import torch
import numpy as np
import time
from os.path import dirname, abspath


pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))
# print(sys.path)

from eval_utils import evaluate
from pdvc.pdvc import build
from misc.utils import create_logger
from video_dataset import PropSeqDataset, collate_fn
from torch.utils.data import DataLoader
from os.path import basename
import pandas as pd
from misc.utils import set_seed

def create_fake_test_caption_file(metadata_csv_path):
    out = {}
    df = pd.read_csv(metadata_csv_path)
    for i, row in df.iterrows():
        out[basename(row['filename']).split('.')[0]] = {'duration': row['video-duration'], "timestamps": [[0, 0.5]], "sentences":["None"]}
    fake_test_json = '.fake_test_json.tmp'
    json.dump(out, open(fake_test_json, 'w'))
    return fake_test_json

def main(opt):
    folder_path = os.path.join(opt.eval_save_dir, opt.eval_folder)
    if opt.eval_mode == 'test':
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    logger = create_logger(folder_path, 'val.log')
    if opt.eval_model_path:
        model_path = opt.eval_model_path
        infos_path = os.path.join(folder_path, 'info.json')
    else:
        model_path = os.path.join(folder_path, 'model-best.pth')
        infos_path = os.path.join(folder_path, 'info.json')

    logger.info(vars(opt))

    with open(infos_path, 'rb') as f:
        logger.info('load info from {}'.format(infos_path))
        old_opt = json.load(f)['best']['opt']

    for k, v in old_opt.items():
        if k[:4] != 'eval':
            vars(opt).update({k: v})
    if opt.eval_gt_file_for_caption is not None:
        opt.gt_file_for_eval = opt.eval_gt_file_for_caption
    if True:
        # recover the lastest args
        if os.path.exists('.tmp/opts.json'):
            current_full_args = json.load(open('.tmp/opts.json'))
            for k,v in current_full_args.items():
                if k not in vars(opt):
                    vars(opt).update({k:v})
                    print('add missing args: {}={}'.format(k,v))
    opt.transformer_input_type = opt.eval_transformer_input_type
    opt.disable_tqdm = False
    opt.enable_init_query_embed = False
    opt.batch_size = opt.eval_batch_size

    if opt.eval_ec_alpha != -1:
        opt.ec_alpha = opt.eval_ec_alpha
    
    if opt.eval_disable_contrastive and opt.enable_contrastive:
        strict_load_pth = False
        opt.enable_contrastive = False
    elif opt.eval_not_strict_load:
        strict_load_pth = False
    else:
        strict_load_pth = True

    if not torch.cuda.is_available():
        opt.nthreads = 0
    # Create the Data Loader instance
    set_seed(opt.seed)

    if opt.eval_mode == 'test':
        if opt.test_video_meta_data_csv_path is not None:
            opt.eval_caption_file = create_fake_test_caption_file(opt.test_video_meta_data_csv_path)
            opt.visual_feature_folder = opt.test_video_feature_folder
    val_dataset = PropSeqDataset(opt.eval_caption_file,
                                 opt.visual_feature_folder,
                                 opt.dict_file, False, opt.eval_proposal_type,
                                 opt)
    loader = DataLoader(val_dataset, batch_size=opt.eval_batch_size,
                        shuffle=False, num_workers=opt.eval_nthreads, collate_fn=collate_fn)


    model, criterion, contrastive_criterion, postprocessors = build(opt)
    model.translator = val_dataset.translator



    while not os.path.exists(model_path):
        raise AssertionError('File {} does not exist'.format(model_path))

    logger.debug('Loading model from {}'.format(model_path))
    loaded_pth = torch.load(model_path, map_location=opt.eval_device)
    epoch = loaded_pth['epoch']

    # loaded_pth = transfer(model, loaded_pth, model_path+'.transfer.pth')
    model.load_state_dict(loaded_pth['model'], strict=strict_load_pth)
    model.eval()

    model.to(opt.eval_device)

    if opt.eval_mode == 'test':
        out_json_path = os.path.join(folder_path, 'dvc_results_test.json')
        evaluate(model, criterion, contrastive_criterion, postprocessors, loader, out_json_path,
                         logger, alpha=opt.ec_alpha, dvc_eval_version=opt.eval_tool_version, device=opt.eval_device, debug=opt.eval_debug, skip_lang_eval=True, verbose=opt.show_all_results)
    else:
        out_json_path = os.path.join(folder_path, '{}_epoch{}_num{}_alpha{}{}.json'.format(
            time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime()) + str(opt.id), epoch, len(loader.dataset), 
            opt.ec_alpha, "_debug" if opt.debug else ""))
        caption_scores, eval_loss = evaluate(model, criterion, contrastive_criterion, postprocessors, loader, out_json_path,
                         logger, alpha=opt.ec_alpha, dvc_eval_version=opt.eval_tool_version, device=opt.eval_device, debug=opt.eval_debug, skip_lang_eval=False, verbose=opt.show_all_results)
        avg_eval_score = {key: np.array(value).mean() for key, value in caption_scores.items() if key !='tiou'}
        avg_eval_score2 = {key: np.array(value).mean() * 4917 / len(loader.dataset) for key, value in caption_scores.items() if key != 'tiou'}

        logger.info(
            '\nValidation result based on all 4917 val videos:\n {}\n avg_score:\n{}'.format(
                                                                                       caption_scores.items(),
                                                                                       avg_eval_score))

        logger.info(
                '\nValidation result based on {} available val videos:\n avg_score:\n{}'.format(len(loader.dataset),
                                                                                           avg_eval_score2))

    logger.info('saving reults json to {}'.format(out_json_path))
    return out_json_path


def eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_save_dir', type=str, default='save')
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--eval_mode', type=str, default='eval', choices=['eval', 'test'])
    parser.add_argument('--test_video_feature_folder', type=str, nargs='+', default=None)
    parser.add_argument('--test_video_meta_data_csv_path', type=str, default=None)
    parser.add_argument('--eval_folder', type=str, required=True)
    parser.add_argument('--eval_model_path', type=str, default='')
    parser.add_argument('--eval_tool_version', type=str, default='2018', choices=['2018', '2021', '2018_cider'])
    parser.add_argument('--eval_caption_file', type=str, default='data/anet/captiondata/val_1.json')
    parser.add_argument('--eval_gt_file_for_caption', type=str, nargs='+', default=None)
    parser.add_argument('--eval_proposal_type', type=str, default='gt')
    parser.add_argument('--eval_transformer_input_type', type=str, default='queries', choices=['gt_proposals', 'queries'])
    parser.add_argument('--gpu_id', type=str, nargs='+', default=['0'])
    parser.add_argument('--eval_device', type=str, default='cuda')
    parser.add_argument('--eval_nthreads', type=int, default=4)
    parser.add_argument('--show_all_results', default=True)
    parser.add_argument('--eval_enable_matching_score', action='store_true', default=False)
    parser.add_argument('--eval_matching_score_weight', type=float, default=0.)
    parser.add_argument('--eval_ec_alpha', type=float, default=-1, help='-1 means using the ec_alpha from the pretrained model, while other values means using a new ec_alpha')
    
    parser.add_argument('--eval_calculate_query_counts', action='store_true', default=False)

    # For grounding
    parser.add_argument('--eval_enable_grounding', default=True)
    parser.add_argument('--eval_enable_maximum_matching_for_grounding', action='store_true', default=False)
    parser.add_argument('--eval_set_cost_class', type=float, default=0.)
    parser.add_argument('--eval_grounding_cost_alpha', type=float, default=0.25)
    parser.add_argument('--eval_grounding_cost_gamma', type=float, default=2)
    parser.add_argument('--eval_set_cost_cl', type=float, default=1.0)
    parser.add_argument('--eval_disable_captioning', action='store_true', default=False)
    parser.add_argument('--eval_disable_contrastive', action='store_true', default=False)
    parser.add_argument('--eval_gt_file_for_grounding', type=str, default='data/anet/captiondata/grounding/val1_for_grounding.json')
    parser.add_argument('--eval_for_multi_anno', action='store_true', default=False)
    parser.add_argument('--eval_enable_zeroshot_tal', action='store_true', default=False)
    parser.add_argument('--eval_prompt', type=str, default='a video of')
    parser.add_argument('--eval_use_amp', action='store_true', default=False)
    parser.add_argument('--eval_debug', action='store_true', default=False)

    parser.add_argument('--eval_num_queries', type=int, default=0)

    parser.add_argument('--eval_not_strict_load', action='store_true', default=False)


    opt = parser.parse_args()
    return opt
    
if __name__ == '__main__':
    opt = eval_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    if True:
        torch.backends.cudnn.enabled = False
    main(opt)
