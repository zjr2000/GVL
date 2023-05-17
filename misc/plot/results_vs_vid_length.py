import json
import os
import pdb

p='data/anet/captiondata/val_1.json'
result_json = '/Users/wangteng/PycharmProjects/jirui_project/VideoDETR_jinrui/save/04_03PDVC_N30_v_2022-04-03-16-22-55_/prediction/num4917_epoch21.json_rerank_alpha1.0_temp2.0.json'
result_json2 = '/Users/wangteng/PycharmProjects/jirui_project/VideoDETR_jinrui/save/03_25GVL_cost20_N30_newDict/prediction/num4917_epoch17.json_rerank_alpha1.0_temp2.0.json'
out_folder = '/Users/wangteng/PycharmProjects/jirui_project/VideoDETR_jinrui/save/output/'
d=json.load(open(p))
d2=json.load(open(result_json))
duration = {}
output_by_duration = {}

clip_len = 10
max_clip_num = 30

for i in range(1, max_clip_num+1):
    duration[str(i*clip_len)] = []
    output_by_duration[str(i*clip_len)] = {'results': {},"version": "VERSION 1.0", "external_data": {"used:": True, "details": None}}

for k,v in d.items():
    for i in range(1, max_clip_num+1):
        if v['duration'] > (i-1) * clip_len and v['duration'] < (i) * clip_len:
            duration[str(i*clip_len)].append(k)
            # pdb.set_trace()
            output_by_duration[str(i*clip_len)]['results'][k] = d2['results'][k]

for k,v in output_by_duration.items():
    print(k, len(output_by_duration[k]['results']))
    out_path = os.path.join(out_folder + 'duration_{}.json'.format(k))
    json.dump(output_by_duration[k], open(out_path, 'w'))

