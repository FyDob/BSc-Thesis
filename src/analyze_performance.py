# analyze_performance.py
# Functions to prepare network predictions for further analysis.
# Creates dataframes collecting the predictions of all outlier networks for error analysis.
# Only false positives were discussed in this thesis.
# To be used as 'python analyze_performance.py > some_outfile.csv'

import os
import pandas as pd
import corpus_tools
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def numbers_to_words(corpus, experiment, network, hidden_units):
	'''Cleans up the saved detailed RNN predictions by removing additional characters and
	translating character indices back to legible strings.
			args:
				corpus: string, name of the corpus
				experiment: string, name of the experiment
				network: string, name of the network architecture
				hidden_units: int, number of hidden units
			returns:
				data_matrix: list of clean data, ready to be turned into a pd.DataFrame'''
	detail_file = os.path.join('..', 'exp_results', corpus, experiment, 'detail_{}_{}.csv'.format(network, hidden_units))
	training_df = pd.read_csv('../training/{}.csv'.format(corpus),delimiter=',',encoding='latin-1')
	
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
	# Decode numbers to characters
	if experiment == 'LRD':
		num_to_char = {'1': ']', '2': '[', '3': '{', '4': '}', '5': '$'}
	else:
		num_to_char = {'1': ']', '2': '{', '3': '[', '4': '}', '5': '$'}
	# Prepare matrix to be turned into a df
	data_matrix = [[line[0].translate(line[0].maketrans(num_to_char))+line[1].translate(line[1].maketrans(num_to_char)), float(line[2]), int(line[3])] for line in mod_lines]
	
	return data_matrix

def prepare_dataframe(corpus, experiment, network, hidden_units, GLOBAL_WORDS):
	'''Processes the data_matrix from numbers_to_words into a pd.DataFrame for further analysis.
			args:
				detail_file: Path to a prediction-per-word file
			returns:
				df: pd.DataFrame with all columns relevant for analysis'''
	data_matrix = numbers_to_words(corpus, experiment, network, hidden_units)
	df = pd.DataFrame(data_matrix, columns=['words', 'predictions', 'golds'])
	df['words'] = GLOBAL_WORDS
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
	#print(df.head())
	return df

def measure_performance(results):
	'''Calculates precision, recall and F1 score of a dataframe containing predictions and gold labels.
			args:
				results: Dataframe containing predictions and gold labels
			returns:
				precision: TPs/(TPs+FPs)
				recall: TPs/(TPs+FNs)
				f1_score: 2*((precision*recall)/(precision+recall))'''
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
	
def extend_performance_measures():
	'''Iterates through all files containing experiment results (accuracy, loss) and their corresponding detail files, which contain every single word, prediction and gold label. From that, precision, recall and F1 score are calculated and added to the experiment results file.
			args:
				none
			returns:
				none'''
	corpora = ['base', 'high', 'low']
	experiments = ['LRD', 'ND']
	networks = ['SRNN', 'GRU', 'LSTM']
	for corpus in corpora:
		for experiment in experiments:
			for network in networks:
				for hidden_units in [2**i for i in range(1,10)]:
					# Appending precision, recall, f1_score to sparse_file
					detail_file = os.path.join('..', 'exp_results', corpus, experiment, 'detail_{}_{}.csv'.format(network, str(hidden_units)))
					sparse_file = os.path.join('..', 'exp_results', corpus, experiment, '{}_{}.csv'.format(network, str(hidden_units)))
					sparse = pd.read_csv(sparse_file)
					# Processing values in the details file to calculate additional performance measures
					details = numbers_to_words(detail_file)
					precision, recall, f1_score = measure_performance(details)
					sparse['precision'], sparse['recall'], sparse['f1_score'] = precision, recall, f1_score
					sparse.to_csv(sparse_file, index=False)
					# Saving the extended values
					prepend_header(detail_file)
	
def breakdownPredictions(df, corpus, experiment, network, hidden_units, performance):
	'''Transforms a full dataframe of word-prediction-gold_label data into 4 dataframes specific to a single model: true positives, false positives, true negatives and false negatives.
		args:
			df: Complete dataframe of word-prediction-gold_labels
			corpus, experiment, network, hidden_units, performance: Strings filtering the df for the specific model
		returns:
			true_positives, false_positives, true_negatives, false_negatives: Dataframes containing all TPs, FPs, TNs and FNs of a specific model'''
	generals = pd.DataFrame()
	generals['corpus'], generals['experiment'], generals['network'], generals['hidden_units'], generals['performance'] =  pd.Series(corpus), experiment, network, hidden_units, performance
	# Determine all predictions in their own series to analyze.
	true_positives = df.loc[df.golds == 1].loc[df.predictions >= 0.5]
	false_positives = df.loc[df.golds == 0].loc[df.predictions >= 0.5]
	true_negatives = df.loc[df.golds == 0].loc[df.predictions <= 0.5]
	false_negatives = df.loc[df.golds == 1].loc[df.predictions <= 0.5]
	
	true_positives = true_positives.join(generals)
	false_positives = false_positives.join(generals)
	true_negatives = true_negatives.join(generals)
	false_negatives = false_negatives.join(generals)
	
	for column in generals.columns:
		true_positives[column] = true_positives[column].fillna(generals[column][0])
		false_positives[column] = false_positives[column].fillna(generals[column][0])
		true_negatives[column] = true_negatives[column].fillna(generals[column][0])
		false_negatives[column] = false_negatives[column].fillna(generals[column][0])
	
	return true_positives, false_positives, true_negatives, false_negatives

def prepend_header(detail_file):
	'''Includes a descriptive header in a file containing individual words, predictions and their gold label.
		args:
			detail_file: Path to a file containing individual words, predictions and their gold label
		returns:
			none'''
	f = open(detail_file,'r')
	temp = f.read()
	f.close()

	f = open(detail_file, 'w')
	f.write('words,predictions,golds\n')

	f.write(temp)
	f.close()

def create_mega_df():
	'''Creates and saves four dataframes containing every single word, prediction and gold label for every outlier model (accuracy >55%/<45%), split into true positives, false positives, true negatives and false negatives.'''
	global_words_LRD = numbers_to_words('base', 'LRD', 'LSTM', '8')
	df = pd.DataFrame(global_words_LRD, columns=['words', 'predictions', 'golds'])
	GLOBAL_WORDS_LRD = df['words']
	global_words_ND = numbers_to_words('base', 'ND', 'SRNN', '2')
	df = pd.DataFrame(global_words_ND, columns=['words', 'predictions', 'golds'])
	GLOBAL_WORDS_ND = df['words']
	
	true_positives_frames = []
	false_positives_frames = []
	true_negatives_frames = []
	false_negatives_frames = []
	valid_nesting_depth_frames = []
	error_pos_frames = []
	error_depth_frames = []
	networks = ['base LRD LSTM 8 good', 'base LRD GRU 2 good', 'base LRD GRU 128 good',
		'base LRD GRU 32 bad', 'base LRD LSTM 128 bad', 'base LRD LSTM 16 bad', 'base LRD SRNN 16 bad',
		'low LRD GRU 2 good', 'low LRD SRNN 4 good', 'low LRD LSTM 16 good', 'low LRD GRU 64 good',
		'low LRD LSTM 4 bad',
		'high LRD GRU 512 good',
		'high LRD LSTM 8 bad', 'high LRD GRU 8 bad', 'high LRD LSTM 4 bad',
		'base ND LSTM 128 good', 'base ND LSTM 32 good', 'base ND GRU 512 good', 'base ND SRNN 2 good', 'base ND GRU 4 good',
		'base ND SRNN 128 bad', 'base ND SRNN 256 bad',
		'low ND LSTM 512 good', 'low ND GRU 64 good', 'low ND SRNN 32 good', 'low ND LSTM 16 good', 'low ND GRU 4 good', 'low ND LSTM 8 good',
		'low ND SRNN 4 bad', 'low ND LSTM 32 bad',
		'high ND SRNN 16 good',
		'high ND LSTM 64 bad']
		
	for entry in networks:
		corpus, experiment, network, hidden_units, performance = entry.split()
		# Set translation table for numbers_to_words
		if experiment == 'LRD':
			GLOBAL_WORDS = GLOBAL_WORDS_LRD
		else:
			GLOBAL_WORDS = GLOBAL_WORDS_ND
		details = prepare_dataframe(corpus, experiment, network, hidden_units, GLOBAL_WORDS)
		true_positives, false_positives, true_negatives, false_negatives = breakdownPredictions(details, corpus, experiment, network, hidden_units, performance)
		true_positives_frames.append(true_positives)
		false_positives_frames.append(false_positives)
		true_negatives_frames.append(true_negatives)
		false_negatives_frames.append(false_negatives)
	# Concatenate all results
	mega_true_positives = pd.concat(true_positives_frames)
	mega_false_positives = pd.concat(false_positives_frames)
	mega_true_negatives = pd.concat(true_negatives_frames)
	mega_false_negatives = pd.concat(false_negatives_frames)

	# Save results
	tp_out = os.path.join('..','results','mega_true_positives.csv')
	fp_out = os.path.join('..','results','mega_false_positives.csv')
	tn_out = os.path.join('..','results','mega_true_negatives.csv')
	fn_out = os.path.join('..','results','mega_false_negatives.csv')
	mega_true_positives.to_csv(tp_out)
	mega_false_positives.to_csv(fp_out)
	mega_true_negatives.to_csv(tn_out)
	mega_false_negatives.to_csv(fn_out)	

def count_error_types(fp, experiment, network, corpus, performance):
	'''Determines number of open/closed bracket error words in a dataframe.
			args:
				fp: dataframe containing either false positives or true negatives (since other categories do not have incorrect words)
				experiment: string, name of the experiment
				network: string, name of the network architecture
				hidden_units: int, number of hidden units
			returns:
				none'''
	open = fp.loc[fp.experiment == experiment].loc[fp.network == network].loc[fp.corpus == corpus].loc[fp.performance == performance].loc[fp.error == 'open'].count()[1]
	closed = fp.loc[fp.experiment == experiment].loc[fp.network == network].loc[fp.corpus == corpus].loc[fp.performance == performance].loc[fp.error == 'closed'].count()[1]
	total = open + closed
	if total:
		if closed:
			ratio = open/closed
		else:
			ratio = np.inf
		
		print("{},{},{},{},{},{},{},{}".format(network, experiment, corpus.capitalize(), performance, open, closed, total, ratio))

def print_bracket_ratio_table():
	'''Creates a table splitting false positives into open/closed error types for each outlier network, sorted by 'good' and 'bad' outliers.'''
	fp_in = os.path.join('..','results','mega_false_positives.csv')
	fp = pd.read_csv(fp_in)
	experiments = ['LRD', 'ND']
	networks = ['SRNN', 'LSTM', 'GRU']
	corpora = ['base', 'low', 'high']
	print("network,experiment,corpus,performance,open,closed,total,ratio")
	for experiment in experiments:
		for corpus in corpora:
			for network in networks:
				count_error_types(fp, experiment, network, corpus, 'good')
	for experiment in experiments:
		for corpus in corpora:
			for network in networks:
				count_error_types(fp, experiment, network, corpus, 'bad')	

create_mega_df()
print_bracket_ratio_table()