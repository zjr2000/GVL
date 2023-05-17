import enum
import json
file_list = [
    '/devdata1/VideoCaption/anet_challenge_2022/data/youmakeup/annotations/origin/train_origin.json',
    '/devdata1/VideoCaption/anet_challenge_2022/data/youmakeup/annotations/origin/val_origin.json',
    '/devdata1/VideoCaption/anet_challenge_2022/data/youmakeup/annotations/origin/grounding_test_origin.json'
]

def split_sentence(sentence):
    tokens = [',', ':', '!', '_', ';', '.', '?', '"', '\\n', '\\', '.']
    for token in tokens:
        sentence = sentence.replace(token, ' ')
    sentence_split = sentence.replace('.', ' . ').replace(',', ' , ').lower().split()
    return sentence_split

vocab_set = set()
sentences = []

for file in file_list:
    with open(file, 'r') as f:
        data = json.load(f)
    for item in data:
        if 'caption' in item:
            sentences.append(item['caption'])
        else:
            for _, step in item['step'].items():
                sentences.append(step['caption'])
    
for sent in sentences:
    words = split_sentence(sent)
    for word in words:
        vocab_set.add(word)

vocab_set.add('UNK')
vocab_set.add('<bos>')
vocab_set.add('<eos>')
vocabs = list(vocab_set)
print(len(vocabs))
data = {'ix_to_word':{}, 'word_to_ix':{}}

for idx, word in enumerate(vocabs, start=1):
    data['ix_to_word'][idx] = word
    data['word_to_ix'][word] = idx


with open('vocabulary_youmakeup.json', 'w') as f:
    json.dump(data, f)
