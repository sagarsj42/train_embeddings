import os
os.chdir('/scratch/sagarsj42')

import json
import re
import pickle
import time

import nltk
import enchant
import numpy as np
import scipy.sparse as sp

start = time.time()
reviews = list()

for i in range(7):
    filename = 'review_words-' + str(i+1) + '.pkl'
    print('Opening', filename, end='  ')
    with open(filename, 'rb') as f:
        reviews_set = pickle.load(f)
        print('Contains', len(reviews_set), 'entries')
        reviews.extend(reviews_set)

end = time.time()
print('Loading data files', 'Time taken:', end - start)

start = time.time()
word_count = dict()

for review in reviews:
    for sentence in review:
        for word in sentence:
            if not word in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
                
vocab_size = len(word_count)
word_list = list(word_count.keys())
end = time.time()
print('Registering vocab', 'Time taken:', end - start)

start = time.time()
co_occ = np.zeros((vocab_size, vocab_size))

for i, review in enumerate(reviews):
    if i % 10000 == 0:
        print('Review no.:', i)
    for sentence in review:
        sent_len = len(sentence)
        for wi in range(sent_len):
            curr_word = sentence[wi]
            curr_ind = word_list.index(curr_word)

            left_ind = max(0, wi-3)
            right_ind = min(sent_len, wi+4)

            for cont_word_ind in range(left_ind, right_ind):
                cont_word = sentence[cont_word_ind]
                cont_ind = word_list.index(cont_word)
                co_occ[curr_ind, cont_ind] += 1

end = time.time()
print('Creating co-occurence matrix', 'Time taken:', end - start)

start = time.time()
sparsity = (1 - np.count_nonzero(co_occ) / co_occ.size) * 100
print('Sparsity:', sparsity)

co_occ_csr = sp.csr_matrix(co_occ)
end = time.time()

print('Sparsifying', 'Time taken:', end - start)

start = time.time()
u, s, vt = sp.linalg.svds(co_occ_csr, k=750)
end = time.time()

print('SVD', 'Time taken:', end - start)
print(u.shape, s.shape, vt.shape)

sp.save_npz('co_occ.npz', co_occ_csr)
np.save('u', u)
np.save('s', s)
np.save('vt', vt)

print('Sparsified co-occurence matrix & SVD matrices saved')

