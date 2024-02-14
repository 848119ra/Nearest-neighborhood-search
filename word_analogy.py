import random
import numpy as np

vocabulary_file = 'D:/university/Tampere/courses/3rd/Intro to Pattern Recognition & Machine Learning/exercise3/word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

# Main loop for analogy
while True:
    input_term = input("\nEnter three words (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        # input = "king queen price"
        input_list = input_term.split(" ")
        # input_list = ["king", "queen", "prince"]


        x_1 = input_list[0]
        vector_1 = W[vocab[x_1]]
        x_2 = input_list[1]
        vector_2 = W[vocab[x_2]]
        subtract_vector = np.subtract(vector_2, vector_1)

        x_3 = input_list[2]
        vector_3 = W[vocab[x_3]]
        vector_4 = vector_3 + subtract_vector

        distances = (vector_4 - W) ** 2
        distances = np.sum(distances, axis=1)

        # Sort by distance
        sorted_indices = np.argsort(distances)[:2]

        a = [ivocab[i] for i in sorted_indices]

        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for x in a:
            print("%35s\t\t%f\n" % (x, distances[vocab[x]]))