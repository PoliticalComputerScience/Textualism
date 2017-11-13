from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance

class Clustering:

	def __init__(self, model):
		self.model = model
		self.word_to_vec = {word:model.wv[word] for word in model.wv.vocab}
		self.vectors = [model.wv[word] for word in model.wv.vocab]
		self.clusterer = KMeansClusterer(num_means=5, distance=cosine_distance)
		self.clusterer.cluster_vectorspace(self.vectors)
		self.central_words = []
		for centroid in self.clusterer._means:
			closest = None
			for word in self.word_to_vec:
				vector = self.word_to_vec[word]
				if not closest or (cosine_distance(vector, centroid) < cosine_distance(closest[1], centroid)):
					closest = (word, self.word_to_vec[word])
			self.central_words.append(closest)