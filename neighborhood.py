import gensim, queue, sys
from nltk.cluster.util import cosine_distance
from gensim.models import Word2Vec

class Neighborhood:
	def __init__(self, word, model):
		"""
		@param word: (type=str) 
		@param model: (type=Word2Vec Model) <(neighboring_word, edge_weight)>
		@field word: (type=str)
		@field similarity_neighbors: (type=list<tuple<str, float>>) neighbors to word based on 
		@field proximity_neighbors: (type=list<tuple<str, float>>) neighbors to word based on cosine_distance 
		"""
		self.word = word
		self.similarity_neighbors = model.wv.most_similar(positive=[word])
		"""q = queue.PriorityQueue(10)
		word_to_vec = {word:model.wv[word] for word in model.wv.vocab}
		word_vector = word_to_vec[word]
		for key in word_to_vec:
			dist_word_pair = (cosine_distance(word_to_vec[key], word_vector), key)
			q.put(dist_word_pair)
		self.proximity_neighbors = []
		while not q.empty():
			pair_to_add = q.get()
			self.proximity_neighbors.append((pair_to_add[1], pair_to_add[0]))""" #(not working am stupid)


#TO EXECUTE ON COMMAND LINE
#python neighborhood.py <word> <model_file>
word = sys.argv[1]
model = Word2Vec.load(sys.argv[2])
n = Neighborhood(word, model)
print(n.word)
print("Similarity Neighbors")
for neighbor in n.similarity_neighbors:
	print(neighbor)