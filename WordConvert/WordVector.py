from gensim.models import Word2Vec
from matplotlib import pyplot

# define training data
sentences = [['Converting', 'words', 'to', 'vector'],
			['this', 'is', 'the', 'second', 'sentence'],
			['and', 'the', 'final', 'sentence']]

# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.save('model2.bin')
# load model
new_model = Word2Vec.load('model2.bin')
print(new_model)
