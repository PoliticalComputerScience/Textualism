from web.evaluate import evaluate_similarity
from gensim.models import Word2Vec
from word_embeddings_benchmarks_master.web.datasets.similarity import fetch_WS353
from cluster import Clustering

def eval_sim(model):
	d = {word:model.wv[word] for word in model.wv.vocab}
	data = fetch_WS353(which="similarity")
	return evaluate_similarity(d, data.X, data.y)

