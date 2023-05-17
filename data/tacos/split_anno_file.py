import json
import os
import numpy as np
import random

'''
Split the origin anno file. Make sure every sample has 
less than K (sentence, time) pair
'''
K = 8
root = 'captiondata_tan'
save_dir = 'loss_ratio' 
name_list = ['train','test', 'val']

for name in name_list:
    origin_path = os.path.join(root, name + '.json')
    with open(origin_path, 'r') as f:
        origin_data = json.load(f)
    new_data = {}
    
    for key, video_info in origin_data.items():
        duration = video_info['duration']
        timestamps = video_info['timestamps']
        sentences = video_info['sentences']
        timesteps_dict = {}
        for timestep, sentence in zip(timestamps, sentences):
            cur_key = tuple(timestep)
            if cur_key not in timesteps_dict:
                timesteps_dict[cur_key] =[]
            timesteps_dict[cur_key].append(sentence)
        sentences = []
        timestamps = []
        for timestamp in sorted(timesteps_dict.keys()):
            for sentence in timesteps_dict[timestamp]:
                sentences.append(sentence)
                timestamps.append(list(timestamp))

        indicies = list(range(len(timestamps)))
        # Make sure on sample has more than one query
        if len(indicies) % K == 1:
            indicies.append(0) 
        random.shuffle(indicies)
        start = 0
        group_id = 0
        while start < len(indicies):
            group_key = "{0:03d}".format(group_id) + key
            group_indicies = sorted(indicies[start: start + K])
            group_timestamps = [timestamps[index] for index in group_indicies]
            group_sentences = [sentences[index] for index in group_indicies]
            new_data[group_key] = {
                'duration': duration,
                'timestamps': group_timestamps,
                'sentences': group_sentences
            }
            start += K
            group_id += 1
    split_save_path = os.path.join(save_dir, 'split_' + name + '.json')
    with open(split_save_path, 'w') as f:
        json.dump(new_data, f)
        