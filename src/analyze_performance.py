# Script to explore model performance
# - How many models >= 50%?
# - How many models < 50%?
# - SRNN performance (regardless of experiment, corpus)
# - LSTM performance (regardless of experiment, corpus)
# - GRU performance (regardless of experiment, corpus)
# - high performance (regardless of experiment, network)
# - base performance (regardless of experiment, network)
# - low performance (regardless of experiment, network)

import os
import pandas as pd

def numbers_to_words(detail_file):
	# TODO: Comment, clean up
	# TODO: split into 2 functions -> numbers2words, file2df
	'''Cleans up the saved detailed RNN predictions by removing additional characters and
	translating character indices back to legible strings.'''
	table = str.maketrans(dict.fromkeys(' []')) # Remove superfluous characters
	lines = []
	with open(detail_file, 'r') as f:
		for line in f:
			line = line.translate(table).strip().split(',')
			lines.append(line)
	prev_line = []
	mod_lines = []
	# Merge lines split by \n in saving
	for i in range(2,len(lines), 2):
		line = lines[i]
		prev_line = lines[i-1]
		prev_line.extend(line)
		mod_lines.append(prev_line)
	num_to_char = str.maketrans('12345', '][{}$') # 34 and 12 could be the other bracket pair and vice versa, does not matter
	mod_lines = [[line[0].translate(num_to_char)+line[1].translate(num_to_char), float(line[2]), int(line[3])] for line in mod_lines]
	df = pd.DataFrame(mod_lines, columns=['words', 'predictions', 'golds'])
	
	return df

def measure_performance(results):
	true_pos = len(results.loc[results.golds == 1].loc[results.predictions >= 0.5])
	false_pos = len(results.loc[results.golds == 0].loc[results.predictions >= 0.5])
	true_neg = len(results.loc[results.golds == 0].loc[results.predictions < 0.5])
	false_neg = len(results.loc[results.golds == 1].loc[results.predictions < 0.5])
	if true_pos+false_pos == 0:
		precision = 0.
		recall = 0.
		f1_score = 0.
	else:
		precision = float(true_pos/(true_pos+false_pos))
		recall = float(true_pos/(true_pos+false_neg))
		if precision+recall == 0.:
			f1_score = 0.
		else:
			f1_score = 2.*((precision*recall)/(precision+recall))
	return precision, recall, f1_score
	
def prepend_header(detail_file):
	f = open(detail_file,'r')
	temp = f.read()
	f.close()

	f = open(detail_file, 'w')
	f.write('words,predictions,golds\n')

	f.write(temp)
	f.close()


# DETAILLED
#detail_file = os.path.join('..', 'exp_results', 'base', 'LRD', 'detail_{}_{}.csv'.format('LSTM', '8')) # 91% accuracy LRD
detail_file = os.path.join('..', 'exp_results', 'base', 'LRD', 'detail_{}_{}.csv'.format('SRNN', '16')) # 27% accuracy LRD
sparse_file = os.path.join('..', 'exp_results', 'base', 'LRD', '{}_{}.csv'.format('SRNN', '16')) # 27% accuracy LRD
sparse = pd.read_csv(sparse_file)
details = numbers_to_words(detail_file)
print(details.describe())
precision, recall, f1_score = measure_performance(details)
sparse['precision'], sparse['recall'], sparse['f1_score'] = precision, recall, f1_score
sparse.to_csv(sparse_file, index=False)
#details = pd.read_csv(detail_file)
#print(details.head())

# 1. Determine which networks qualify for closer scrutiny (best performers, worst performers)
# Iterate through all detail files
corpora = ['base', 'high', 'low']
experiments = ['LRD', 'ND']
networks = ['SRNN', 'GRU', 'LSTM']
for corpus in corpora:
	for experiment in experiments:
		for network in networks:
			for hidden_units in [2**i for i in range(1,10)]:
				detail_file = os.path.join('..', 'exp_results', corpus, experiment, 'detail_{}_{}.csv'.format(network, str(hidden_units)))
				sparse_file = os.path.join('..', 'exp_results', corpus, experiment, '{}_{}.csv'.format(network, str(hidden_units)))
				print(sparse_file)
				sparse = pd.read_csv(sparse_file)
				details = numbers_to_words(detail_file)
				print(details.describe())
				precision, recall, f1_score = measure_performance(details)
				sparse['precision'], sparse['recall'], sparse['f1_score'] = precision, recall, f1_score
				sparse.to_csv(sparse_file, index=False)

				#prepend_header(detail_file)
				

# SURFACE
# infile_results = os.path.join('..', 'results', 'results.csv')
# results = pd.read_csv(infile_results)

# # print(results.head())
# # print(results.describe())

# LRD = results.loc[results.experiment == 'LRD']
# #print(LRD.describe())
# print(LRD.nlargest(3, 'accuracy'))
# ND = results.loc[results.experiment == 'ND']
# print(ND.describe())

# SRNN = results.loc[results.network == 'SRNN']#.loc[results.experiment == 'LRD']
# print("SRNN")
# print(SRNN.describe())

# LSTM = results.loc[results.network == 'LSTM']#.loc[results.experiment == 'LRD']
# print("LSTM")
# print(LSTM.describe())

# GRU = results.loc[results.network == 'GRU']#.loc[results.experiment == 'LRD']
# print("GRU")
# print(GRU.describe())

# high = results.loc[results.corpus == 'high']
# print("high")
# print(high.describe())

# base = results.loc[results.corpus == 'base']
# print("base")
# print(base.describe())

# low = results.loc[results.corpus == 'low']
# print("low")
# print(low.describe())