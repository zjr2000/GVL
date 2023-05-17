import json

with open('word2id.json', 'r') as f: 
    data = json.load(f)

new_data = {'ix_to_word':{}, 'word_to_ix': {}}
for w, idx in data.items():
    new_data['ix_to_word'][idx] = w
    new_data['word_to_ix'][w] = idx

with open('vocabulary_tacos.json', 'w') as f:
    json.dump(new_data, f)