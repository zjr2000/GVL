import json
import os

tan2d_root = '/devdata1/VideoCaption/2D-TAN/data/TACoS'
cbp_root = '/devdata1/VideoCaption/CBP/datasets/tacos/data/save'

name_list = ['train.json', 'val.json', 'test.json']

# # CBP data
# for name in name_list:
#     our_data = {}
#     duration_info_path = os.path.join(tan2d_root, name)
#     anno_info_path = os.path.join(cbp_root, name)
#     with open(duration_info_path, 'r') as f:
#         duration_info = json.load(f)
#     with open(anno_info, 'r') as f:
#         anno_info = json.load(f)

#     for key, anno_item in anno_info.items():
#         item = {}
#         duration_key = key + '.avi'
#         print(key, duration_key)
#         duration = float(duration_info[duration_key]['num_frames']) / float(duration_info[duration_key]['fps'])
#         item['duration'] = duration
#         item['timestamps'] = anno_item['timestamps']
#         item['sentences'] = anno_item['sentences']
#         our_data[key] = item
#     with open(name, 'w') as f:
#         json.dump(our_data, f)

# 2d-TAN data
for name in name_list:
    anno_info_path = os.path.join(tan2d_root, name)
    with open(anno_info_path, 'r') as f:
       anno_info = json.load(f)
    our_data = {}
    for key, video_anno in anno_info.items():
        item = {}
        duration =  video_anno['num_frames'] / video_anno['fps']
        timestamps = video_anno['timestamps']
        timestamps = [[max(timestamp[0]/video_anno['fps'],0), min(timestamp[1]/video_anno['fps'],duration)] for timestamp in timestamps]
        sentences = video_anno['sentences']
        item['duration'] = duration
        item['timestamps'] = timestamps
        item['sentences'] = sentences
        our_data[key[:-4]] = item
    with open(os.path.join('./captiondata_tan', name), 'w') as f:
        json.dump(our_data, f)