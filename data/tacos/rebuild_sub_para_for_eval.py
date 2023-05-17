import json
import os
import numpy as np
import random

def get_ids_for_sub_para(total_event_num, random_para_event_num=False):
    if random_para_event_num:
        min_sub_para_num = total_event_num // max_para_event_num
        max_sub_para_num = total_event_num // min_para_event_num
        para_num = random.randint(min_sub_para_num, max_sub_para_num + 1)
        if para_num != 0:
            num_of_each_para = [total_event_num // para_num] * para_num
            if total_event_num % para_num != 0:
                num_of_each_para.append(total_event_num % para_num)
                para_num += 1
        else:
            num_of_each_para = [total_event_num]
            para_num += 1
    else:
        para_num = total_event_num // max_para_event_num
        remain = total_event_num - para_num * max_para_event_num
        num_of_each_para = [max_para_event_num] * para_num
        if remain > 0:
            num_of_each_para.append(remain)

    indices = list(range(total_event_num))
    random.shuffle(indices)
    indices_of_each_para = []
    start = 0
    for num in num_of_each_para:
        indices_of_each_para.append(sorted(indices[start:start + num]))
        start += num

    return indices_of_each_para 

root = 'captiondata_tan' 
name_list = ['test', 'val']
prefix = 'min2_max8_'
min_para_event_num = 2
max_para_event_num = 8
RANDOM = False

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

        total_event_num = len(timesteps_dict.keys())
        para_id = 0

        for group_id in range(max_anno_num):
            indices_of_each_para = get_ids_for_sub_para(total_event_num, random_para_event_num=RANDOM)
            origin_keys = sorted(timesteps_dict.keys())
            for cur_para_indices in indices_of_each_para:
                para_timesteps = []
                para_sentences = []
                para_key = "{0:03d}".format(para_id) + key
                cur_para_keys = [origin_keys[i] for i in range(total_event_num) if i in cur_para_indices]
                for unique_timestep in cur_para_keys:
                    para_timesteps.append(list(unique_timestep))
                    para_sentences.append(timesteps_dict[unique_timestep][group_id])
                para_item = {'duration':duration, 'timestamps':para_timesteps, 'sentences':para_sentences}
                para_grounding_item = {'duration':duration, 'timestamps':para_timesteps}
                para_para_item = ''
                for sent in para_sentences:
                    para_para_item += (sent + '.')
                new_data[para_key] = para_item
                new_data_for_grounding[para_key] = para_grounding_item
                new_data_for_para[para_key] = para_para_item
                para_id += 1
    
    rebuild_save_path = os.path.join(root, prefix + 'rebuild_' + name + '.json')
    rebuild_grounding_save_path = os.path.join(root, 'grounding', prefix + 'rebuild_grounding_' + name + '.json')
    rebuild_para_save_path = os.path.join(root, 'para', prefix + 'rebuild_para_' + name + '.json')
    with open(rebuild_save_path, 'w') as f:
        json.dump(new_data, f)
    with open(rebuild_grounding_save_path, 'w') as f:
        json.dump(new_data_for_grounding, f)
    with open(rebuild_para_save_path, 'w') as f:
        json.dump(new_data_for_para, f)

    