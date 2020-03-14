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
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

MAX_LEN = ((20-2)*2)+1+2 # Maximum length of words in all corpora and experiments -- LRD length formula

# def filter_length(df, max_len_train=MAX_LEN):
	# '''Filters words of length > max_len_train out of a dataframe.
		# args:
			# df: pd.DataFrame of words
			# max_len_train: Maximum length words in df should have
		# returns:
			# df: pd.DataFrame of words'''
	# mask = (df['word'].str.len() <= max_len_train)
	# df = df.loc[mask]
	
	# return df
	
# def get_tokenizer_config(training_df):
	# '''Preprocesses training data by splitting it into train/test and transforming the strings into sequences of numbers.
		# args:
			# df: pd.DataFrame of words
			# max_word_length: Maximum wanted length of words
		# returns:
			# tok.get_config: Configuration of tokenizer fitted on training corpus, to be used on experiment corpus; returned as dict'''
	# training_df = filter_length(training_df)
	# X = training_df.word
	# Y = training_df.value
	# # Encode target values
	# le = LabelEncoder()
	# Y = le.fit_transform(Y)
	# Y = Y.reshape(-1,1)

	# X_train_temp,X_test_temp,Y_train,Y_test = train_test_split(X,Y,test_size=0.2) # == TEST_SIZE in RNN_classifier.py

	# # Preprocess words into sequences of numbers corresponding to the characters
	# tok = Tokenizer(char_level=True)
	# tok.fit_on_texts(X_train_temp)
	
	# return tok.get_config() # dict of config

def numbers_to_words(corpus, experiment, network, hidden_units):
	# TODO: Comment, clean up
	# TODO: split into 2 functions -> numbers2words, file2df
	'''Cleans up the saved detailed RNN predictions by removing additional characters and
	translating character indices back to legible strings.'''
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
	
def breakdownPredictions(df, corpus, experiment, network, hidden_units, performance):
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
	f = open(detail_file,'r')
	temp = f.read()
	f.close()

	f = open(detail_file, 'w')
	f.write('words,predictions,golds\n')

	f.write(temp)
	f.close()

def print_to_console(corpus, experiment, network, hidden_units, performance, GLOBAL_WORDS):
	print(",{} {} {} {}-{},".format(corpus, experiment, performance, network, hidden_units).upper())
	details = prepare_dataframe(corpus, experiment, network, hidden_units, GLOBAL_WORDS)
	breakdownPredictions(details, corpus, experiment, network, hidden_units)

def create_mega_df():
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
		print("{} DONE!".format(entry))
	mega_true_positives = pd.concat(true_positives_frames)
	mega_false_positives = pd.concat(false_positives_frames)
	mega_true_negatives = pd.concat(true_negatives_frames)
	mega_false_negatives = pd.concat(false_negatives_frames)
	print(mega_false_positives.loc[mega_false_positives.performance == 'good'].describe())
	print(mega_false_positives.loc[mega_false_positives.performance == 'bad'].describe())
	tp_out = os.path.join('..','results','mega_true_positives.csv')
	fp_out = os.path.join('..','results','mega_false_positives.csv')
	tn_out = os.path.join('..','results','mega_true_negatives.csv')
	fn_out = os.path.join('..','results','mega_false_negatives.csv')
	mega_true_positives.to_csv(tp_out)
	mega_false_positives.to_csv(fp_out)
	mega_true_negatives.to_csv(tn_out)
	mega_false_negatives.to_csv(fn_out)	

#create_mega_df()

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
tp_in = os.path.join('..','results','mega_true_positives.csv')
fp_in = os.path.join('..','results','mega_false_positives.csv')
tn_in = os.path.join('..','results','mega_true_negatives.csv')
fn_in = os.path.join('..','results','mega_false_negatives.csv')
#tp = pd.read_csv(tp_in)
fp = pd.read_csv(fp_in)
#tn = pd.read_csv(tn_in)
#fn = pd.read_csv(fn_in)

def plot_results(df, experiment, column_tex, performance):
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	if column_tex == 'Max Valid Nesting Depth':
		max_ylim = 15.
	elif column_tex == 'Error Pos':
		max_ylim = 25.
	else:
		max_ylim = 1.
	column = column_tex.lower().replace(' ', '_')
	variable_to_title = {'Max Valid Nesting Depth' : 'Maximum Nesting Depth', 'Error Pos' : 'Error Position', 'Predictions' : 'Output Layer Activation', 'LRD':'Experiment 1', 'ND':'Experiment 2'}
	
	columns_analysis = [column]
	corpora = ['base', 'low', 'high']
	networks = ['SRNN', 'LSTM', 'GRU']
	networks_tex = [r'SRNN', r'LSTM', r'GRU']
	values = []
	for corpus in corpora:
		corp_values = []
		for network in networks:
			#network_values = []
			value = df.loc[df.experiment == experiment].loc[df.network == network].loc[df.corpus == corpus].loc[df.performance == performance].describe().loc['mean', column]
			value = np.nan_to_num(value)
			print(value, corpus, column, network)
			corp_values.append(value)
		values.append(corp_values)
	print(values)
	print(len(values))
	width = 0.25
	x = np.arange(len(networks_tex))

	fig, ax = plt.subplots()
	
	rects1 = ax.bar(x - width, values[0], width, label='base')
	rects2 = ax.bar(x, values[1], width, label='low')
	rects3 = ax.bar(x + width, values[2], width, label='high')
	
	rects = [rects1, rects2, rects3]
	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel(r'Mean {}'.format(variable_to_title[column_tex]))
	ax.set_title(r'{}: {} Performance of {} Outliers'.format(variable_to_title[experiment], variable_to_title[column_tex], performance.capitalize()))
	ax.set_xticks(x)
	ax.set_xticklabels(networks_tex)
	ax.legend(loc='upper center', ncol=3)


	def autolabel(rects):
		"""As long as bar height isn't 0, attach a text label above each bar in *rects*, displaying its height."""
		for rect in rects:
			height = rect.get_height()
			if height == 0.0:
				continue
			else:
				ax.annotate('{:.3f}'.format(height),
							xy=(rect.get_x() + rect.get_width() / 2, height),
							xytext=(0, 3),  # 3 points vertical offset
							textcoords="offset points",
							ha='center', va='bottom')

	for rect in rects:
		autolabel(rect)
		
	axes = plt.gca()
	axes.set_ylim([0., max_ylim])

	fig.tight_layout()
	
	plot_out = os.path.join('..', 'latex', 'fig', '{}_{}_{}.pdf'.format(experiment, column, performance))
	plt.savefig(plot_out, bbox_inches='tight')
	return fig
	# save as PDF

def count_error_types(fp, experiment, network, corpus, performance):
	open = fp.loc[fp.experiment == experiment].loc[fp.network == network].loc[fp.corpus == corpus].loc[fp.performance == performance].loc[fp.error == 'open'].count()[1]
	closed = fp.loc[fp.experiment == experiment].loc[fp.network == network].loc[fp.corpus == corpus].loc[fp.performance == performance].loc[fp.error == 'closed'].count()[1]
	total = open + closed
	if total:
		if closed:
			ratio = open/closed
		else:
			ratio = np.inf
		
		print("{},{},{},{},{},{},{},{}".format(network, experiment, corpus, performance, open, closed, ratio, total))

# PLOTTING
# pred_lrd = plot_results(fp, 'LRD', 'Predictions', 'bad')
# pred_lrd = plot_results(fp, 'LRD', 'Max Valid Nesting Depth', 'bad')
# pred_lrd = plot_results(fp, 'LRD', 'Error Pos', 'bad')
# pred_lrd = plot_results(fp, 'LRD', 'Predictions', 'good')
# pred_lrd = plot_results(fp, 'LRD', 'Max Valid Nesting Depth', 'good')
# pred_lrd = plot_results(fp, 'LRD', 'Error Pos', 'good')
# pred_lrd = plot_results(fp, 'ND', 'Predictions', 'bad')
# pred_lrd = plot_results(fp, 'ND', 'Max Valid Nesting Depth', 'bad')
# pred_lrd = plot_results(fp, 'ND', 'Error Pos', 'bad')
# pred_lrd = plot_results(fp, 'ND', 'Predictions', 'good')
# pred_lrd = plot_results(fp, 'ND', 'Max Valid Nesting Depth', 'good')
# pred_lrd = plot_results(fp, 'ND', 'Error Pos', 'good')

experiments = ['LRD', 'ND']
networks = ['SRNN', 'LSTM', 'GRU']
corpora = ['base', 'low', 'high']
print(",,,,open,closed,ratio,total")
for experiment in experiments:
	for corpus in corpora:
		for network in networks:
			count_error_types(fp, experiment, network, corpus, 'good')
for experiment in experiments:
	for corpus in corpora:
		for network in networks:
			count_error_types(fp, experiment, network, corpus, 'bad')

#lstm_lrd = plot_results(fp, 'LRD', 'LSTM', 'bad')
#plt.show()
#srnn_lrd = plot_results(fp, 'LRD', 'SRNN', 'bad')
#plt.show()

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
				
