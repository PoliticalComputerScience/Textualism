import gensim, queue, sys
import numpy as np
from nltk.cluster.util import cosine_distance
from gensim.models import Word2Vec

class Neighborhood:
	def __init__(self, word, model, neighbors=10):
		"""
		@param word: (type=str) 
		@param model: (type=Word2Vec Model) <(neighboring_word, edge_weight)>
		@field word: (type=str)
		@field similarity_neighbors: (type=list<tuple<str, float>>) neighbors to word based on 
		@field proximity_neighbors: (type=list<tuple<str, float>>) neighbors to word based on cosine_distance 
		"""
		self.word = word
		self.similarity_neighbors = model.wv.most_similar(positive=[word], topn=neighbors)
		"""q = queue.PriorityQueue(10)
		word_to_vec = {word:model.wv[word] for word in model.wv.vocab}
		word_vector = word_to_vec[word]
		for key in word_to_vec:
			dist_word_pair = (cosine_distance(word_to_vec[key], word_vector), key)
			q.put(dist_word_pair)
		self.proximity_neighbors = []
		print("done")
		while not q.empty():
			pair_to_add = q.get()
			self.proximity_neighbors.append((pair_to_add[1], pair_to_add[0])) #(not working am stupid)
        """

def get_neighboring_words(word, model, n=10, verbose=False):
    n = Neighborhood(word, model, neighbors=n)
    if verbose:
        print(n.word)
        print("Similarity Neighbors")
    words = []
    for neighbor in n.similarity_neighbors:
        if verbose:
            print(neighbor)
        words.append(neighbor[0])
    return words, n.similarity_neighbors

def get_svd_from_words(model, words, verbose=False):
    vecs = [model.wv[word] for word in words]
    mat = np.stack(vecs, axis=0)
    if verbose:
        print('shape of word embeddngs matrix:', mat.shape) #shape is (neighbors,32), neighbors defaults to 10
    U, s, V = np.linalg.svd(mat)
    if verbose:
        print('singular values:', s)   # Take a quick look at svd_test.py (run it) if you want to convince yourself of how svd works for m by n matrices
               # We basically want the first three row vectors in V; these are the eigenvectors that explain most of the variation in the rows (i.e. word embeddings) of the original matrix.
    return U, s, V

def get_coords_from_svd_projection(V, model, words, verbose=False):
    V_cut = V[:3,:]
    if verbose:
        print('matrix after removing less import eigenvectors:')
        print(V_cut)
        print('squared row magnitudes:')
        for i in range(3):     # this yields 1.0 every time, i.e. to compute projection coordinates we can ignore the a.a on the denominator (V_cut has unitary rows)
            print(V_cut[i,:].dot(V_cut[i,:]))


    #for each word, associate it with its projection coordinates in the V_cut basis.
    # This associates each words with a ordered triple of points, which allows us to graph in 3d,
    # or 2d (you can just use the first two coordinates if you want). One idea for the future would
    #be to label the axes of the graph with the word that the corresponding basis vector of our graph is closest to.
    return V_cut, {w: V_cut.dot(model.wv[w]) for w in words}

"""
Main utility method for any users of this file. Bundles up three sub-processes to allow you 
to get a three dimensional space on which to plot the similar words to the given one based on some model
The current method computes the svd of the similar words, along with the original word, but throws out words of length
2 or smaller by default.
"""
def get_points_from_word_and_model(word, model_path, verbose=False, bigger_than=2):
    model = Word2Vec.load(model_path)
    
    #gets the n "most similar" words to the initial word in this model. 
    #Also returns a dict containing those similarity scores, which is not used in this method
    words,_ = get_neighboring_words(word,model, n=10, verbose=verbose)

    # an intermediate processing step here that removes small words and adds in the main word we're considering?
    cond = (lambda w: len(w) > bigger_than) if bigger_than > 0 else (lambda w: True)
    words = [word] + [w for w in words if cond(w)]

    # computes the svd of the matrix containing the embeddings of the 10 most similar words as rows. 
    # U and s are not used, but are left in for readability. s contains the singular values, set verbose=True to print them.
    U, s, V = get_svd_from_words(model, words, verbose=verbose)
    
    basis, coords = get_coords_from_svd_projection(V, model, words, verbose=verbose)

    return basis, coords, model

#TO EXECUTE ON COMMAND LINE
#python neighborhood.py <word> <model_file>
if __name__ == '__main__':
    DEBUG = False
    word = sys.argv[1]
    model_path = sys.argv[2]
    basis, coords, _ = get_points_from_word_and_model(word, model_path, verbose=DEBUG)
    if DEBUG:
        print('basis vectors', basis)
    for elem in coords:
        print(elem, ':', coords[elem])

## Ideas ##    ## Basically just some useful attributes of our model to check out 
    #print(model.__dict__) #syn1neg might be useful: output matrix of all word embeddings?
    #print(model.accuracy) #also might be useful; see https://rare-technologies.com/word2vec-tutorial/ (also in useful_links.txt)
    #print(model.wv[words[0]].shape)  # shape is (32,)