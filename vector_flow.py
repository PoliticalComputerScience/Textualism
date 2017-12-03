import gensim
from gensim.models import Word2Vec as Word2Vec
import sys, re, os
import nltk
from nltk.tokenize import word_tokenize
from eval_embedding import eval_sim
SENTENCE_TOKENIZER = nltk.data.load('./nltk_data/tokenizers/punkt/english.pickle') 
QUOTES = re.compile("\u201c|\u201d")

def read_dir(path):
	"""
	@param path: (type = str) path to dir that containes files
	@return: (type=str)
	"""
	assert os.path.exists(path)
	file_names = os.listdir(path)
	return read(file_names, path)

def read(file_names, path):
	"""
	Reads in a text file as a string, returning the stringified version
	@param file_names: (type=list<str>) files to be read
	@param path: (type = str) path to dir that containes files
	@return: (type=str)
	"""
	data = ""
	for file_name in file_names:
		with open(path + "/" + file_name, "r") as f:
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

def init_model(sentences):
	"""
	Initializes a Word2Vec model
	@param sentences: (type=list<list<str>>) a list of lists of words
	@param min_count: (type=int) minimum frequency for inclusion of a word 
	@param window: (type=int) window to relate words
	@param dim: (type=int) dimension of word vectors
	@return: (type=Word2Vec) model
	"""
	best_spearmen = 0
	best_model = None
	best_params = (0,0,0)
	count = 0
	for dim in range(20, 61):
		for window in range(5, 11):
			for min_count in range(2,5):
				model = Word2Vec(sentences, min_count=min_count, window=window, size=dim)
				similarity_spearmen = eval_sim(model)
				if similarity_spearmen > best_spearmen:
					best_model = model
					best_params = (min_count, window, dim)
					best_spearmen = similarity_spearmen
				print(str(count) + ": " + str(similarity_spearmen))
				count += 1
	print("min_count: " + str(best_params[0]), 
		"window: " + str(best_params[1]), 
		"dim: " + str(best_params[2]))
	print(best_spearmen)
	return best_model

#TO EXECUTE ON COMMAND LINE
#python vector_flow.py <model_file_name> <path_to_dir>
data = read_dir(sys.argv[2])
sentences = parse(data)
model = init_model(sentences)
model.save(sys.argv[1])



