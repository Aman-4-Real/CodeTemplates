'''
Author: Aman
Date: 2022-08-19 23:56:17
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-10-17 18:54:26
'''
'''
This script clean the data of input dialogs.
'''

import multiprocessing
import re
import pickle
import string
import time as t
from datetime import datetime
from itertools import product

import rich.progress
from rich.progress import track
from tqdm import tqdm, trange
# from nltk.corpus import stopwords


def filter_session(session):
	contexts = [pre_clean_seq(uttr) for uttr in session['context']]
	response = pre_clean_seq(session['response'])
	
	# Remove empty contexts and too short responses
	if len(contexts) == 0 or len(response) < 10:
		return None
	# Remove sessions that is too long
	if len(contexts) > 20 or len(response) > 100:
		return None

	whole_string = " ".join(contexts + [response])

	# Remove sessions that contain too many non-English
	non_english = 0
	for char in whole_string:
		if char not in string.printable:
			non_english += 1
	if non_english / len(whole_string) > 0.5:
		return None

	# Remove sessions that contain too many stopwords
	# stopwords_set = set(stopwords.words('english'))
	# stopwords_count = 0
	# for word in whole_string.split():
	# 	if word in stopwords_set:
	# 		stopwords_count += 1
	# if stopwords_count / len(whole_string) > 0.6:
	# 	return None

	# Remove sessions with safe responses
	safe_responses = [
		"i don't know", "i do not know",
		"i don't know what to say", "i do not know what to say",
		"i don't know what you mean", "i do not know what you mean",
		"i don't understand", "i do not understand",
		"i don't understand what you mean", "i do not understand what you mean",
		"i can't understand", "i can not understand",
		"i can't understand what you mean", "i can not understand what you mean",
		"i'm the same as you", "i am the same as you",
		"i'm the same with you", "i am the same with you"
	]
	for safe_response in safe_responses:
		if (safe_response in response.lower() or response.lower() in safe_response) and \
			min(len(response), len(safe_response)) / max(len(response), len(safe_response)) > 0.8:
			return None

	return {'context': contexts, 'response': response}



### The following is from https://github.com/elmines/preprocessing/blob/master/corpus.py ###

def pre_clean_seq(text):
	"""
	Cleans a single text sequence with a host of regular expressions
	
	:param str text: Text to be cleaned.
	
	:returns: The cleaned text
	:rtype: str
	"""
	
	text = re.sub("(``)|('')", ' " ', text) #These are already in the corpus and could screw up the tokenizer

	text = re.sub(r" 's", r"'s", text)
	text = re.sub(r" n't", r"n't", text)
	text = re.sub(r" 'll", r"'ll", text)
	text = re.sub(r" 're", r"'re", text)
	text = re.sub(r" 'd", r"'d", text)
		
	#Punctuation/Symbols
	text = re.sub("&quot;", ' " ', text)         
	text = re.sub("&amp;", ' & ', text)          
	text = re.sub("(<.*?>)|(&.*?;)", "", text)            #HTML tags and entities
	text = re.sub(r'[\?\.\!\-]+(?=[\?\.\!\-])', '', text) #Duplicate end punctuation
	text = re.sub(r"\. \. \.", "...", text)               #Compress ellipses to one token
	text = re.sub('\s+', ' ', text ).strip()             #Replace special whitespace characters with simple spaces

	#Punctuation
	text = re.sub(r" \.", r".", text)
	text = re.sub(r" \?", r"?", text)
	text = re.sub(r" \!", r"!", text)
	text = re.sub(r" ,", r",", text)
	text = re.sub(r" ;", r";", text)
	text = re.sub(r" :", r":", text)
	text = re.sub(r" *-+ *", r"-", text) #Strip hyphens/dashes of preceding and trailing whitespace

	return text

### The End ###


class MyMultiProcess(object):

	def __init__(self, my_func, num_workers=8):
		self.my_func = my_func
		self.pool = multiprocessing.Pool(num_workers)
		self.num_workers = num_workers

	def __call__(self, inputs):
		if not inputs:
			print("Error! No input files!")
			exit(1)

		size = len(inputs)//self.num_workers
		starts = [i*size for i in range(self.num_workers)]
		param = [(inputs, size, self.num_workers, ) + (start,) for start in starts]

		res = self.pool.map(self.my_func, param)

		return res


def MyFunction(params):
	data, size, num_workers, start = params
	print(multiprocessing.current_process().name, 'data cleaning...')

	removed = 0
	error = 0
	end = start + size
	if start == (num_workers - 1) * size:
		end = len(data)
	data_iterator = tqdm(data[start:end],
						desc="%s" % (multiprocessing.current_process().name),
						bar_format="{l_bar}{r_bar}")
	res = []
	for item in tqdm(data_iterator):
		try:
			item_cleaned = filter_session(item)
			if item_cleaned is None:
				removed += 1
			else:
				res.append(item_cleaned)
		except:
			error += 1
			continue

	print(multiprocessing.current_process().name, \
		'data cleaning done. Success: %d/%d, Removed: %d, Error: %d' % (len(res), end - start, removed, error))

	return res


def main():
	start_time = datetime.now()
	print("Start time:", start_time)

	workers_num = 8  # num of multiprocers

	print("Loading data...")
	data = pickle.load(open("all_dialogs_107905734.pkl", "rb"))
	# data = data[90000000:]
	print("Data loaded.")
	print("len(data):", len(data))
	print("Time elapsed:", datetime.now() - start_time)

	my_multi_process = MyMultiProcess(MyFunction, workers_num)
	res = my_multi_process(inputs=data)

	# aggregate and save
	print("Aggregating and saving...")
	res = [item for sublist in res for item in sublist]
	print("Total res:", len(res))
	pickle.dump(res, open(f"all_dialogs_cleaned_{len(res)}_4.pkl", "wb"))
	print("Aggregating and saving done.")

	end_time = datetime.now()
	print("End time:", end_time)
	print("Total time:", end_time - start_time)


if __name__ == '__main__' :
	main()
