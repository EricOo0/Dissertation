
import torchtext.vocab as Vocab

import gensim

W2V_BIN_FILE='./GoogleNews-vectors-negative300.bin'
W2V_TXT_FILE='./word2vec.txt'
model = gensim.models.KeyedVectors.load_word2vec_format(W2V_BIN_FILE,binary=True)
 
model.save_word2vec_format(W2V_TXT_FILE)