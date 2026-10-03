from gensim.models import Word2Vec
import json 
import time

t = time.time()
print(f"Starting Word2Vec training at {time.strftime("%H:%M:%S", time.localtime())}")

model = Word2Vec(
    corpus_file='../data/tokenized_instructions_train.txt',
    vector_size=300,
    hs=1,
    negative=0,
    window=10,
    cbow_mean=1,
    epochs=10,
    min_count=10,
    workers=20,
    )

print(f"Word2Vec training time: {time.time() - t} seconds.")

# Export model
model.save('../data/text/word2vec.model')

# Export json with vocab and indices
with open('../data/text/word2vec_vocab_idxs.json', 'w') as f:
    json.dump(model.wv.key_to_index, f)

# Export txt file of just vocab
corpus_vocab = list(model.wv.key_to_index.keys())
with open('../data/text/word2vec_vocab.txt', 'w') as f:
    for word in corpus_vocab:
        f.write(f"{word}\n")
