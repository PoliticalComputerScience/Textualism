import gensim
from gensim.models import Word2Vec as Word2Vec
import sys

def read(file_name):
	"""
	Reads in a text file as a string, returning the stringified version
	@param file_name: (type=str) file to be read
	@return: (type=str)
	"""
	with open(file_name, "r") as f:
		data = f.read().replace("\n", "").replace(",", "")
	return data

def parse(text):
	"""
	Parses a stringified file into sentences, a list of lists of words
	@param text: (type=str) text to parse
	@return: (type=list<list<str>>) parsed text 
	"""
	sentences = text.split(".")
	sentences = [sentence.split(" ") for sentence in sentences]
	return sentences

def init_model(sentences, min_count=2, window=8, dim=20):
	"""
	Initializes a Word2Vec model
	@param sentences: (type=list<list<str>>) a list of lists of words
	@param min_count: (type=int) minimum frequency for inclusion of a word 
	@param window: (type=int) window to relate words
	@param dim: (type=int) dimension of word vectors
	@return: (type=Word2Vec) model
	"""
	model = Word2Vec(sentences, min_count=min_count, window=window, size=dim)
	return model

data = read(sys.argv[1])
sentences = parse(data)
model = init_model(sentences)
model.save("test_model")



