import json

origin_val_file = 'data/youmakeup/annotations/origin/grounding_test_origin.json'
grounding_val_file = 'data/youmakeup/annotations/grounding/test.json'

with open(origin_val_file, 'r') as f:
    data = json.load(f)

new_data = {}

for query in data:
    video_id = 'v_' + query['video_id']
    caption = query['caption']
    title = query['video_title']
    duration = query['video_duration']
    query_idx = query['query_idx']

    if video_id not in new_data:
        new_item = {
            'sentences': [],
            'timestamps': [],
            'duration':duration,
            'title': title,
            'query_indicies': []
        }
        new_data[video_id] = new_item

    new_data[video_id]['sentences'].append(caption)
    new_data[video_id]['query_indicies'].append(query_idx)
    new_data[video_id]['timestamps'].append([0, 0.5])

with open(grounding_val_file, 'w') as f:
    json.dump(new_data, f)