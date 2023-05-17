import json
import os
import numpy as np

root = 'captiondata_tan' 
name_list = ['test', 'val']

for name in name_list:
    origin_path = os.path.join(root, name + '.json')
    with open(origin_path, 'r') as f:
        origin_data = json.load(f)
    new_data = {}
    new_data_for_grounding = {}
    new_data_for_para = {}
    for key, video_info in origin_data.items():
        duration = video_info['duration']
        timesteps = video_info['timestamps']
        sentences = video_info['sentences']
        timesteps_dict = {}
        for timestep, sentence in zip(timesteps, sentences):
            cur_key = tuple(timestep)
            if cur_key not in timesteps_dict:
                timesteps_dict[cur_key] =[]
            timesteps_dict[cur_key].append(sentence)
        max_anno_num = 0
        for _, sentences in timesteps_dict.items():
            max_anno_num = max(max_anno_num, len(sentences))
        for _, sentences in timesteps_dict.items():
            if len(sentences) >= max_anno_num:
                continue
            padding_num = max_anno_num - len(sentences)
            timesteps_dict[_].extend(np.random.choice(timesteps_dict[_], size=padding_num))
        for group_id in range(max_anno_num):
            group_key = "{0:03d}".format(group_id) + key
            group_timesteps = []
            group_sentences = []
            for unique_timestep in sorted(timesteps_dict.keys()):
                group_timesteps.append(list(unique_timestep))
                group_sentences.append(timesteps_dict[unique_timestep][group_id])
            group_item = {'duration':duration, 'timestamps':group_timesteps, 'sentences':group_sentences}
            group_grounding_item = {'duration':duration, 'timestamps':group_timesteps}
            group_para_item = ''
            for sent in group_sentences:
                group_para_item += (sent + '.')
            new_data[group_key] = group_item
            new_data_for_grounding[group_key] = group_grounding_item
            new_data_for_para[group_key] = group_para_item
    
    rebuild_save_path = os.path.join(root, 'rebuild_' + name + '.json')
    rebuild_grounding_save_path = os.path.join(root, 'grounding', 'rebuild_grounding_' + name + '.json')
    rebuild_para_save_path = os.path.join(root, 'para', 'rebuild_para_' + name + '.json')
    with open(rebuild_save_path, 'w') as f:
        json.dump(new_data, f)
    with open(rebuild_grounding_save_path, 'w') as f:
        json.dump(new_data_for_grounding, f)
    with open(rebuild_para_save_path, 'w') as f:
        json.dump(new_data_for_para, f)