import gensim
from gensim.models import Word2Vec as Word2Vec
import sys, re
import nltk
from nltk.tokenize import word_tokenize
SENTENCE_TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle') 
QUOTES = re.compile("\u201c|\u201d")


def read(file_names):
	"""
	Reads in a text file as a string, returning the stringified version
	@param file_names: (type=list<str>) files to be read
	@return: (type=str)
	"""
	data = ""
	for file_name in file_names:
		with open(file_name, "r") as f:
			data += re.sub(QUOTES, "\"", f.read())
			data += "\n"
	return data

def parse(text):
	"""
	Parses a stringified file into sentences, a list of lists of words
	@param text: (type=str) text to parse
	@return: (type=list<list<str>>) parsed text 
	"""
	sentences = SENTENCE_TOKENIZER.tokenize(text)
	sentences = [word_tokenize(sentence) for sentence in sentences]
	return sentences

def init_model(sentences, min_count=2, window=8, dim=40):
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

data = read(sys.argv[2:])
sentences = parse(data)
model = init_model(sentences)
model.save(sys.argv[1])



