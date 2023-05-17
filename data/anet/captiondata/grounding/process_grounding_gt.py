import json

p1 = 'data/anet/captiondata/val_1.json'
p2 = 'data/anet/captiondata/val_2.json'
# out_p = 'data/anet/captiondata/grounding/val_for_grounding.json'
out_p = 'data/anet/captiondata/grounding/val1_for_grounding.json'

val1 = json.load(open(p1))
val2 = json.load(open(p2))
val2 = {}

vid_keys = set(val1.keys()) | set(val2.keys())

out = {}
for vid in vid_keys:
    out[vid[2:]] = {}
    timetstamps = []
    if vid in val1:
        timetstamps.extend(val1[vid]['timestamps'])
        duration = val1[vid]['duration']
    if vid in val2:
        timetstamps.extend(val2[vid]['timestamps'])
        duration = val2[vid]['duration']
    out[vid[2:]]['timestamps'] = timetstamps
    out[vid[2:]]['duration'] = duration
json.dump(out, open(out_p, 'w'))
