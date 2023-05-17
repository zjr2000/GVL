import json
import os

feature_dir = '/root/autodl-tmp/VideoDETR/data/anet/features/tsp'
anet1_3_path = '/root/autodl-tmp/VideoDETR/data/anet/anet1.3/activity_net.v1-3.min.json'
anet_cap_train_path = '/root/autodl-tmp/VideoDETR/data/anet/captiondata/train_modified.json'
save_dir = '/root/autodl-tmp/VideoDETR/data/anet/anet1.3'

anet1_3_train = {}
anet1_3_val = {}

with open(anet1_3_path, 'r') as f:
    anet1_3_all = json.load(f)
    anet1_3_all = anet1_3_all['database']

with open(anet_cap_train_path, 'r') as f:
    anet_cap_train = json.load(f)

for key, item in anet1_3_all.items():
    key = 'v_' + key
    feature_path = os.path.join(feature_dir, key+'.npy')
    if not os.path.exists(feature_path):
        print('key {} not exists'.format(key))
        break

    timestamps = []
    action_labels = []
    sentences = []
    duration = item['duration']
    is_train = item['subset'] == 'training'
    subset = item['subset']
    for anno in item['annotations']:
        timestamp = anno['segment']
        if timestamp[0] > timestamp[1]:
            print('Invalid item in {}'.format(key))
            continue
        sentences.append('')
        timestamps.append(timestamp)
        action_labels.append(anno['label'])
    if len(timestamps) == 0:
        continue

    new_item = {
        'duration': duration,
        'timestamps': timestamps,
        'sentences': sentences,
        'action_labels':action_labels
    }

    if not is_train and key in anet_cap_train.keys():
        # make sure val sample not in anet cap train set
        print('{} exists in anet cap train set, should remove'.format(key))
        break
    if is_train:
        anet1_3_train[key] = new_item
    elif subset == 'validation':
        anet1_3_val[key] = new_item

print('Train number: {}'.format(len(anet1_3_train)))
print('Val number: {}'.format(len(anet1_3_val)))

with open(os.path.join(save_dir, 'train.json'), 'w') as f:
    json.dump(anet1_3_train, f)
with open(os.path.join(save_dir, 'val.json'), 'w') as f:
    json.dump(anet1_3_val, f)

