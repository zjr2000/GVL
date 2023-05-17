import json
import pdb
import sys

import h5py
import numpy as np
from collections import OrderedDict

sys.path.append("external_tool/densevid_eval3/coco-caption3")
from pycocoevalcap.meteor.meteor import Meteor
Meteor_scorer = Meteor()

def remove_nonascii(text):
    PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "--", "...", ";", '\n', '\t', '\r']
    for p in PUNCTUATIONS:
        text = text.replace(p,' ')
    text = text.replace('  ',' ')
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

baseline_path = '/Users/wangteng/PycharmProjects/jirui_project/VideoDETR_jinrui/save/04_03PDVC_N30_v_2022-04-03-16-22-55_/prediction/num4917_epoch21.json_rerank_alpha1.0_temp2.0.json'
baseline2_path = '/Users/wangteng/PycharmProjects/dense-video-captioning-pytorch/data/generated_proposals/Luoweizhou_E2EMT_results/Copy of densecap_with_generated_segments_val_split.json'
Ours_full_path = '/Users/wangteng/PycharmProjects/jirui_project/VideoDETR_jinrui/save/03_25GVL_cost20_N30_newDict/prediction/num4917_epoch17.json_rerank_alpha1.0_temp2.0.json'
GT_path = 'data/anet/captiondata/val_1.json'

def iou(s1, e1, s2, e2):
    i = max(min(e2, e1) - max(s2, s1), 0)
    u = (e1 - s1) + (e2 - s2) - i
    return i / u

GT_data=json.load(open(GT_path))


def find_good_sample(Ours_full_path):

    # TECENT_BAF_path = '/data/huichengzheng/wangteng/dvc2_pytorch04/save/tecent/val_1000_proposal_result_without_joint_rank.json'
    # model_names = ['V0', 'tecent', 'V3']
    # model_paths=[baseline_path, TECTECENT_BAF_path, Ours_full_path]

    model_names = ['baseline', 'baseline2','ours']
    model_paths=[baseline_path, baseline2_path, Ours_full_path]

    refer_sets = [json.load(open(p)) for p in model_paths]

    print('GT_vid_num:',len(GT_data.keys()))
    print('refer_vid_num',[len(refer_set['results'])for refer_set in refer_sets])

    for vid, info in GT_data.items():

        # interest = 'v_xs5imfBbWmw v_CSDApI2nHPU v_8onOVVuN_Is v_A8xThM3onkc v_DvTZ5mmF8NM'.split()
        # if vid not in interest:
        #     continue

        all_ref_for_vid = {}
        for j,name in enumerate(model_names):
            all_ref_for_vid[name] = []
        gt_sents = info['sentences']

        FLAG=1
        for refer_data in refer_sets:
            if vid not in refer_data['results'].keys():
                FLAG=0
        if FLAG==0:
            continue

        for p_i, prop in enumerate(info['timestamps']):
            gt_s, gt_e = prop
            ref_prop_max_iou_list = []

            for i,refer_data in enumerate(refer_sets):


                ref_prop_max_iou = 0

                best_prop=None
                best_prop_sent = 'NONE'
                # best_prop_sent_score = 0
                # best_prop_id=5005

                for refer_prop in refer_data['results'][vid]:
                    s,e = refer_prop['timestamp']
                    iou_ = iou(gt_s, gt_e, s, e)
                    if  iou_ > ref_prop_max_iou:
                        #best_prop_id = refer_prop['num']
                        best_prop = [s,e]
                        best_prop_sent = refer_prop['sentence']
                        #best_prop_sent_score = refer_prop['sentence_confidence']
                        ref_prop_max_iou = iou_
                ref_prop_max_iou_list.append({'gt_prop':[gt_s, gt_e],
                                                  'ref_prop':best_prop,
                                                  'best_iou': ref_prop_max_iou,
                                                  'best_sent': best_prop_sent,
                                                  #'sent_score': best_prop_sent_score,
                                                   # 'prop_id': best_prop_id
                                             })

            for j,l in enumerate(ref_prop_max_iou_list):
                all_ref_for_vid[model_names[j]].append(l['best_sent'])
                # print(sent[p_i])
                # print(l)

        gts = OrderedDict()
        for i in range(len(gt_sents)):
            gts[i] = [remove_nonascii(gt_sents[i])]
        gts__ = {i: gts[i] for i in range(len(gts))}


        avg_score = []
        meteor_scores={}
        for name, sent in all_ref_for_vid.items():
            refs = OrderedDict()
            for i in range(len(sent)):
                refs[i] = [remove_nonascii(sent[i])]
            res__ = {i: refs[i] for i in range(len(refs))}
            # pdb.set_trace()
            _, meteor_score = Meteor_scorer.compute_score(gts__, res__)

            meteor_scores[name]=meteor_score

        # for name, sent in all_ref_for_vid.items():
        #     print(name)
        #     for j,s in enumerate(sent):
        #         print('{:.3} {}'.format(100*meteor_scores[name][j], s))
        #     print('\n')

        avg_score=[]
        for name in model_names:
            avg_score.append(np.array(meteor_scores[name]).mean())

        if avg_score[2] > avg_score[0] + 0.05 and avg_score[2] > 0.1:
        # if True:
        #     if (avg_score[2] > avg_score[1] + 0.03 and avg_score[1] > 0.05) :
            if (avg_score[2] > avg_score[1] + 0.05) :
            #if True:
                print('\n')
                print(vid)
                # d= LDA_data[vid].value
                # top5 = np.argsort(-d)[:5]
                # for topi in top5:
                #     print('LDA:Top1, score:{}, names:{}'.format(d[topi], LDA_names[topi]))

                print('GT')
                for p_time in info['timestamps']:
                    print(p_time)
                for sent in gt_sents:
                    print(sent.encode('utf-8'))

                for name, sent in all_ref_for_vid.items():
                    print(name)
                    for j, s in enumerate(sent):
                        print('(meteor={:.3}) {}'.format(100 * meteor_scores[name][j], s))
            pass

if __name__=='__main__':
    find_good_sample(Ours_full_path)