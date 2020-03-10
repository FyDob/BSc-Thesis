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
import corpus_tools

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
	data_matrix = [[line[0].translate(num_to_char)+line[1].translate(num_to_char), float(line[2]), int(line[3])] for line in mod_lines]
	
	return data_matrix

def prepare_dataframe(detail_file):
	'''Processes the data_matrix from numbers_to_words into a pd.DataFrame for further analysis.
		args:
			detail_file: Path to a prediction-per-word file
		returns:
			df: pd.DataFrame with all columns relevant for analysis'''
	data_matrix = numbers_to_words(detail_file)
	df = pd.DataFrame(data_matrix, columns=['words', 'predictions', 'golds'])
	# Append columns determining the
	# - kind of error in the word if the word is wrong
	# - position of the corrupted character if it can be determined
	# - nesting depth at error position
	# - running bracket distance at error position
	df['max_valid_nesting_depth'] = df.words.map(lambda word: corpus_tools.maxValidNestingDepth(word))
	df['error'] = df.words.map(lambda word: corpus_tools.determineError(word))
	df['error_pos'] = df.words.loc[df.error != 'none'].map(lambda word: corpus_tools.findErrorPosition(word))
	df['error_depth'] = df.words.loc[df.error != 'none'].map(lambda word: corpus_tools.nestingDepthAtPosition(word, corpus_tools.findErrorPosition(word)))
	df['error_distance'] = df.words.loc[df.error != 'none'].map(lambda word: corpus_tools.bracketDistanceAtPosition(word, corpus_tools.findErrorPosition(word)))
	
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
	
def breakdownPredictions(df):
	# Determine all predictions in their own series to analyze.
	true_positives = df.loc[df.golds == 1].loc[df.predictions >= 0.5]
	false_positives = df.loc[df.golds == 0].loc[df.predictions >= 0.5]
	true_negatives = df.loc[df.golds == 0].loc[df.predictions <= 0.5]
	false_negatives = df.loc[df.golds == 1].loc[df.predictions <= 0.5]

	#print(false_positives.loc[false_positives.error == 'open'].describe())
	#print(false_positives.loc[false_positives.error == 'closed'].describe())
	print(false_positives.error.value_counts())

	# TP = true_positives.words.map(lambda word: corpus_tools.maxValidNestingDepth(word))
	# FP = false_positives.words.map(lambda word: corpus_tools.maxValidNestingDepth(word))
	# TN = true_negatives.words.map(lambda word: corpus_tools.maxValidNestingDepth(word))
	# FN = false_negatives.words.map(lambda word: corpus_tools.maxValidNestingDepth(word))
	valid_nesting_depth = pd.DataFrame()
	valid_nesting_depth['true_positives'], valid_nesting_depth['false_positives'], valid_nesting_depth['true_negatives'], valid_nesting_depth['false_negatives'] = true_positives.max_valid_nesting_depth.describe(), false_positives.max_valid_nesting_depth.describe(), true_negatives.max_valid_nesting_depth.describe(), false_negatives.max_valid_nesting_depth.describe()
	
	print(valid_nesting_depth.to_string())
	
def prepend_header(detail_file):
	f = open(detail_file,'r')
	temp = f.read()
	f.close()

	f = open(detail_file, 'w')
	f.write('words,predictions,golds\n')

	f.write(temp)
	f.close()

def print_to_console(corpus, experiment, network, hidden_units, performance):
	detail_file = os.path.join('..', 'exp_results', corpus, experiment, 'detail_{}_{}.csv'.format(network, hidden_units))
	print("===== {} {} {} {}-{} =====".format(corpus, experiment, performance, network, hidden_units).upper())
	details = prepare_dataframe(detail_file)
	print(details.head())
	breakdownPredictions(details)
	

# DETAILLED
# detail_file = os.path.join('..', 'exp_results', 'base', 'LRD', 'detail_{}_{}.csv'.format('LSTM', '8')) # 91% accuracy LRD
# #sparse_file = os.path.join('..', 'exp_results', 'base', 'LRD', '{}_{}.csv'.format('SRNN', '16')) # 27% accuracy LRD
# #sparse = pd.read_csv(sparse_file)
# details = prepare_dataframe(detail_file)
# print(details.head())
# breakdownPredictions(details)
#print(details.loc[details.error == 'closed'].describe())
#print(details.loc[details.error == 'open'].head())
#print(details.error.value_counts())

# good: all networks with experiment accuracy > 55%
# bad: all networks with experiment accuracy < 45%
# LRD
LRD_base_good = ['LSTM 8', 'GRU 2', 'GRU 128']
LRD_base_bad = ['GRU 32', 'LSTM 128', 'LSTM 16', 'SRNN 16']

LRD_low_good = ['GRU 2', 'SRNN 4', 'LSTM 16', 'GRU 64']
LRD_low_bad = ['LSTM 4']

LRD_high_good = ['GRU 512']
LRD_high_bad = ['LSTM 8', 'GRU 8', 'LSTM 4']

# ND
ND_base_good = ['LSTM 128', 'LSTM 32', 'GRU 512', 'SRNN 2', 'GRU 4']
ND_base_bad = ['SRNN 128', 'SRNN 256']

ND_low_good = ['LSTM 512', 'GRU 64', 'SRNN 32', 'LSTM 16', 'GRU 4', 'LSTM 8']
ND_low_bad = ['SRNN 4', 'LSTM 32']

ND_high_good = ['SRNN 16']
ND_high_bad = ['LSTM 64']

for entry in LRD_base_good:
	network, hidden_units = entry.split()
	print_to_console('base', 'LRD', network, hidden_units, 'good')
	
for entry in LRD_base_bad:
	network, hidden_units = entry.split()
	print_to_console('base', 'LRD', network, hidden_units, 'bad')

for entry in LRD_low_good:
	network, hidden_units = entry.split()
	print_to_console('low', 'LRD', network, hidden_units, 'good')
	
for entry in LRD_low_bad:
	network, hidden_units = entry.split()
	print_to_console('low', 'LRD', network, hidden_units, 'bad')

for entry in LRD_high_good:
	network, hidden_units = entry.split()
	print_to_console('high', 'LRD', network, hidden_units, 'good')
	
for entry in LRD_high_bad:
	network, hidden_units = entry.split()
	print_to_console('high', 'LRD', network, hidden_units, 'bad')
	
for entry in ND_base_good:
	network, hidden_units = entry.split()
	print_to_console('base', 'ND', network, hidden_units, 'good')
	
for entry in ND_base_bad:
	network, hidden_units = entry.split()
	print_to_console('base', 'ND', network, hidden_units, 'bad')

for entry in ND_low_good:
	network, hidden_units = entry.split()
	print_to_console('low', 'ND', network, hidden_units, 'good')
	
for entry in ND_low_bad:
	network, hidden_units = entry.split()
	print_to_console('low', 'ND', network, hidden_units, 'bad')

for entry in ND_high_good:
	network, hidden_units = entry.split()
	print_to_console('high', 'ND', network, hidden_units, 'good')
	
for entry in ND_high_bad:
	network, hidden_units = entry.split()
	print_to_console('high', 'ND', network, hidden_units, 'bad')

# 1. Determine which networks qualify for closer scrutiny (best performers, worst performers)
# Iterate through all detail files
# corpora = ['base', 'high', 'low']
# experiments = ['LRD', 'ND']
# networks = ['SRNN', 'GRU', 'LSTM']
# for corpus in corpora:
	# for experiment in experiments:
		# for network in networks:
			# for hidden_units in [2**i for i in range(1,10)]:
				# continue
				# Appending precision, recall, f1_score to sparse_file
				# detail_file = os.path.join('..', 'exp_results', corpus, experiment, 'detail_{}_{}.csv'.format(network, str(hidden_units)))
				# sparse_file = os.path.join('..', 'exp_results', corpus, experiment, '{}_{}.csv'.format(network, str(hidden_units)))
				# print(sparse_file)
				# sparse = pd.read_csv(sparse_file)
				# details = numbers_to_words(detail_file)
				# print(details.describe())
				# precision, recall, f1_score = measure_performance(details)
				# sparse['precision'], sparse['recall'], sparse['f1_score'] = precision, recall, f1_score
				# sparse.to_csv(sparse_file, index=False)

				#prepend_header(detail_file)
				
