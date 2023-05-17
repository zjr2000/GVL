import json
import sys

# p1 = sys.argv[1]
# p2 = sys.argv[2]
# out_p = sys.argv[3]

p1='save/0218_bs1_cl02_t05_clip_v_2022-02-18-13-16-00_/2022-02-21-03-29-31_0218_bs1_cl02_t05_clip_v_2022-02-18-13-16-00__epoch15_num4917_alpha0.3_cl0.0.json.grounding.json.new.json'
p2='save/0218_bs1_cl02_t05_clip_v_2022-02-18-13-16-00_/2022-02-21-14-25-46_0218_bs1_cl02_t05_clip_v_2022-02-18-13-16-00__epoch15_num4885_alpha0.3_cl0.0.json.grounding.json.new.json'

out_p='save/0218_bs1_cl02_t05_clip_v_2022-02-18-13-16-00_/merge.json'

gt1 = json.load(open('data/anet/captiondata/val_1.json'))

def merge(p1, p2):
    d1 = json.load(open(p1))['results']
    d2 = json.load(open(p2))['results']
    key1 = [k[:11] for k in d1.keys()]
    key2 = [k[:11] for k in d2.keys()]
    all_keys = set(key1) | set(key2)

    for vid in all_keys:
        if 'v_'+vid in gt1:
            pid = len(gt1['v_'+vid]['timestamps'])
        else:
            pid = 0
        if vid in key2:
            vid_p_count = key2.count(vid)
            for j in range(vid_p_count):
                old_vid = vid + '-' + str(j)
                new_vid = vid + '-' + str(j+pid)
                if old_vid in d2.keys():
                    d1[new_vid] = d2[old_vid]
    json.dump({'results':d1}, open(out_p, 'w'))

merge(p1, p2)