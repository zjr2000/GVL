import json
import sys
import numpy as np
dvc_json = sys.argv[1]
out_json = dvc_json + '.tal_proc.json'
enable_bg_class = False

tal_class = open('data/anet/anet1.3/action_name.txt').read().split('\n')
assert len(tal_class) == 200

out = {
   "version": "VERSION 1.3",
   "results": {},
   "external_data": {
    "used": True, # Boolean flag. True indicates used of external data.
    "details": "First fully-connected layer from VGG-16 pre-trained on ILSVRC-2012 training set"
    }
}

d = json.load(open(dvc_json))['results']
thres = 0.2
alpha = 1.0
enable_duplicate_proposal = False

for k,v in d.items():
    vid = k[2:]
    out['results'][vid] = []
    for p in v:
        segment = p['timestamp']
        # cl_scores = p['tal_cl_scores'] if 'tal_cl_scores' in p else p['cl_scores']
        cl_scores = p['aux_tal_cl_scores'] if 'aux_tal_cl_scores' in p else p['tal_cl_scores']
        assert len(cl_scores) == len(tal_class) or len(cl_scores) == len(tal_class) + 1
        prop_score = p['proposal_score']
        if not enable_duplicate_proposal:
            scores = [prop_score + alpha * cl_score for i, cl_score in enumerate(cl_scores)]
            if not enable_bg_class:
                scores = scores[:len(tal_class)]
            max_id = np.argmax(scores)
            if max_id >= len(tal_class):
                continue
            new_p = {
                "label": tal_class[max_id],
                "score": scores[max_id],
                "prop_score": prop_score,
                'cl_score': cl_scores[max_id],
                "segment": segment
            }
            out['results'][vid].append(new_p)
        else:
            for i, cl_score in enumerate(cl_scores):
                score = prop_score + alpha * (1) * cl_score
                if score >= thres:
                    new_p = { "label": tal_class[i],
                              "score": cl_score,
                              "segment": segment }
                    out['results'][vid].append(new_p)
json.dump(out, open(out_json, 'w'))
