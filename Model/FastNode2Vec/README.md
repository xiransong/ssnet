requires gensim=3.8

need to modify gensim's source code:

models/word2vec.py: line 1704
- wv.vectors[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), wv.vector_size)
+ wv.vectors[i] = self.seeded_vector(str(wv.index2word[i]) + str(self.seed), wv.vector_size)

models/word2vec.py: line 1722
- newvectors[i - len(wv.vectors)] = self.seeded_vector(wv.index2word[i] + str(self.seed), wv.vector_size)
+ newvectors[i - len(wv.vectors)] = self.seeded_vector(str(wv.index2word[i]) + str(self.seed), wv.vector_size)
