from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance

class Clustering:

	def __init__(self, model):
		"""
		@param model: (type=Word2Vec model)
		"""
		self.model = model #store the Word2Vec model object in case of future use
		self.word_to_vec = {word:model.wv[word] for word in model.wv.vocab} #mapping from word strings to vectors
		self.vectors = [model.wv[word] for word in model.wv.vocab] 
		clusterer = KMeansClusterer(num_means=5, distance=cosine_distance) #the object that will cluster our vectors, num_means will eventually be parameterized
		clusterer.cluster_vectorspace(self.vectors)
		self.central_words = []

		#find closest words to centroids
		for centroid in clusterer._means:
			closest = None
			for word in self.word_to_vec:
				vector = self.word_to_vec[word]
				if not closest or (cosine_distance(vector, centroid) < cosine_distance(closest[1], centroid)):
					closest = (word, self.word_to_vec[word])
			self.central_words.append(closest)
		self.centroids = clusterer._means