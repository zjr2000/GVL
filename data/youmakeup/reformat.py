import json
import time

def to_sec(time_str):
    t = time.strptime(time_str, '%H:%M:%S')
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec

origin_file = 'annotations/origin/captioning_test_origin.json'
target_file = 'annotations/caption/test.json'

with open(origin_file, 'r') as f:
    origin_data = json.load(f)

target_data = {}

for video_info in origin_data:
    video_id = 'v_' + video_info['video_id']
    title = video_info['video_title']
    duration = video_info['video_duration']
    timestamps = [[0, 0.5]]
    sentences = ['a a']
    areas = [['face']]
    # for _, step in video_info['step'].items():
    #     area = step['area']
    #     sentence = step['caption']
    #     start, end = step['startime'], step['endtime']
    #     timestamp = [to_sec(start), to_sec(end)]
    #     timestamps.append(timestamp)
    #     sentences.append(sentence)
    #     areas.append(area)
    new_item = {
        'duration':duration,
        'timestamps':timestamps,
        'sentences':sentences,
        'title': title,
        'areas':areas,
        "actions": ["makeup"], 
        "action_labels": [0]
    }
    target_data[video_id] = new_item

with open(target_file, 'w') as f:
    json.dump(target_data, f)