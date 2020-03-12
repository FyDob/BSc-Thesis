# Training, testing and experimenting on RNN models for Fynn Dobler's Bachelor's Thesis.
# - Preprocessing data
# - Building models
# - Training and testing models with a train/val/test split
# - Saving models
# - Saving loss, accuracy on test split
# - Saving loss, accuracy on experiment data

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
tf.keras.backend.clear_session() # Reduces memory leak during training - seems to be issue with TF2.0
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense, Embedding, SimpleRNN, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

MODE = sys.argv[1]
NETWORK = sys.argv[2]
HIDDEN_UNITS = int(sys.argv[3])
CORPUS = sys.argv[4]
EXPERIMENT = sys.argv[5]
EPOCHS = 1#100
TEST_SIZE = 0.2
VOCAB_SIZE = 7 # 6 characters + 1 do combat off by 1 error in Embedding Layer
MAX_LEN = ((20-2)*2)+1+2 # Maximum length of words in all corpora and experiments -- LRD length formula
BATCH = 512
CHECKPOINT_DIR = os.path.join('..', 'saved_models', CORPUS, NETWORK, str(HIDDEN_UNITS))
CORPUS_DF = pd.read_csv('../training/{}.csv'.format(CORPUS),delimiter=',',encoding='latin-1')
EXP_DF = pd.read_csv('../experiment/{}.csv'.format(EXPERIMENT))

# ============ PROCESSING INPUT DATA ============

def filter_length(df, max_len_train):
	'''Filters words of length > max_len_train out of a dataframe.
		args:
			df: pd.DataFrame of words
			max_len_train: Maximum length words in df should have
		returns:
			df: pd.DataFrame of words'''
	mask = (df['word'].str.len() <= max_len_train)
	df = df.loc[mask]
	df.head()
	df.info()
	
	return df

def prepare_training_data(df, max_word_length=MAX_LEN):
	'''Preprocesses training data by splitting it into train/test and transforming the strings into sequences of numbers.
		args:
			df: pd.DataFrame of words
			max_word_length: Maximum wanted length of words
		returns:
			X_train: Preprocessed words for the train split
			X_test: Preprocessed words for the test split
			Y_train: Target values for the train split
			Y_test: Target values for the test split
			tok: Tokenizer fitted on training corpus, to be used on experiment corpus'''
	df = filter_length(df, max_word_length)
	X = df.word
	Y = df.value
	# Encode target values
	le = LabelEncoder()
	Y = le.fit_transform(Y)
	Y = Y.reshape(-1,1)

	X_train_temp,X_test_temp,Y_train,Y_test = train_test_split(X,Y,test_size=TEST_SIZE)

	# Preprocess words into sequences of numbers corresponding to the characters
	tok = Tokenizer(char_level=True)
	tok.fit_on_texts(X_train_temp)
	sequences = tok.texts_to_sequences(X_train_temp)
	X_train = sequence.pad_sequences(sequences,maxlen=MAX_LEN)
	test_sequences = tok.texts_to_sequences(X_test_temp)
	X_test = sequence.pad_sequences(test_sequences,maxlen=MAX_LEN)
	
	return X_train, X_test, Y_train, Y_test, tok
	
# def prepare_training_data_NOSPLIT(dataframe, max_word_length):
	'''Prepares training data without a train_test_split.'''
	# df = filter_length(dataframe, max_word_length)
	# X = df.word
	# Y = df.value
	# le = LabelEncoder()
	# Y = le.fit_transform(Y)
	# Y = Y.reshape(-1,1)

	# X_train = X
	# Y_train = Y

	# tok = Tokenizer(char_level=True)
	# tok.fit_on_texts(X_train)
	# sequences = tok.texts_to_sequences(X_train)
	# sequences_matrix = sequence.pad_sequences(sequences,maxlen=MAX_LEN)
	
	# return sequences_matrix, Y_train, tok

def prepare_experiment_data(dataframe, tok, max_word_length=MAX_LEN):
	'''Preprocesses experiment data by splitting it into train/test and transforming the strings into sequences of numbers.
		args:
			df: pd.DataFrame of words
			max_word_length: Maximum wanted length of words
			tok: Tokenizer as fitted on the training corpus	
		returns:
			X: Preprocessed words
			Y: Target values'''
	df = filter_length(dataframe, max_word_length)
	X_temp = df.word
	Y = df.value
	# Encode target values
	le = LabelEncoder()
	Y = le.fit_transform(Y)
	Y = Y.reshape(-1,1)
	
	# Preprocess words into sequences of numbers corresponding to the characters
	# The same tokenizer is used for training and experiment data (training tokenizer is called in this function)
	sequences = tok.texts_to_sequences(X_temp)
	X = sequence.pad_sequences(sequences,maxlen=MAX_LEN)
	
	return X, Y

# ============ MODELS ============
def build_model(layer_size=HIDDEN_UNITS, network=NETWORK, vocabulary=VOCAB_SIZE, max_len=MAX_LEN):
	'''Creates a LSTM, GRU or SRNN based model with a specified number of hidden units.
		args:
			layer_size: Number of hidden units in the LSTM, GRU or SRNN layer
			network: Name of network architecture
			vocabulary: Number of units in the Embedding layer
			max_len: Maximum length of a single input sequence
		returns:
			model: Trainable model'''
	model = Sequential()
	model.add(Embedding(vocabulary,vocabulary,input_length=max_len)) # Embeds the input sequence in the same dimensionality - simplifies the tf dataflow
	if network == 'LSTM':	
		model.add(LSTM(layer_size))
	elif network == 'GRU':
		model.add(GRU(layer_size))
	elif network == 'SRNN':
		model.add(SimpleRNN(layer_size))
	else:
		raise ValueError("{} is not a valid network name. Valid network names are 'SRNN', 'LSTM' and 'GRU'.".format(NETWORK))
		return 0
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
	model.summary()
	
	return model
	
def train_model(X, Y, checkpoint_dir=CHECKPOINT_DIR):
	'''Trains the model on provided training data and saves the model weights for every epoch.
	Training stops either after the global maximum number of EPOCHS or if the the loss on the validation set val_loss does not improve for 3 epochs in a row, in which case the previously best model is used for the rest of the session.
		args:
			X: Preprocessed words
			Y: Target values
			hidden_units: Number of hidden units in the LSTM, GRU or SRNN layer
			corpus: Name of the training corpus
		returns:
			model: Trained model
	'''
	# Saving checkpoints: Directory, filenames, callback
	# Name of the checkpoint files, saving the values of loss, accuracy, val_loss and val_accuracy
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}")
	#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch:02d}-{loss:.4f}-{accuracy:.4f}") # for len8 experiments
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
	
	# Create and train model
	model = build_model()
	model.fit(X,Y,batch_size=BATCH,epochs=EPOCHS,validation_split=0.15,
				callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, restore_best_weights=True),
							checkpoint_callback])

	return model

# ============ TRAINING/TESTING/EXPERIMENTS ============
def set_best_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
	'''tf.train.latest_checkpoint() loads the latest saved model weights by referring to the file 'checkpoint' in the respective checkpoint directory.
	However, the latest model is not always the best.
	To circumvent this issue, this function overwrites the file 'checkpoint' to refer
	to the best (= lowest val_loss) model weight configuration, so tf.train.latest_checkpoint() instead retrieves the best model.
		args:
			checkpoint_dir: Directory whose checkpoint file is rewritten
		returns:
			None'''
	files = os.listdir(checkpoint_dir)
	weight_files = files[1:] # Ignore checkpoint file
	filenames = [filename[:-6] for filename in weight_files if filename.endswith('.index')]
	
	# Determine index of model weights with minimal val_loss
	val_losses = []
	for filename in filenames:
		prefix, train_loss, train_acc, val_loss, val_acc = filename.split('-')
		val_losses.append(float(val_loss))
	idx = val_losses.index(min(val_losses))
	best_weights = filenames[idx]
	
	# Overwrite checkpoint
	outfile_name = os.path.join(checkpoint_dir, 'checkpoint')
	outfile = open(outfile_name, 'w')
	outfile.write('model_checkpoint_path: "{0}"\nall_model_checkpoint_paths: "{0}"'.format(best_weights))
	outfile.close()
	
	return None

def train_test():
	'''Trains and tests the model with a train_test_split. Saves the accuracy and loss on the test data in a file with name '{NETWORK}_{HIDDEN_UNITS}.csv'.
	Overwrites 'checkpoint' file TF2.0 uses to store the name of the latest saved weights file with the name of the lowest val_loss weights file.
		args:
			none
		returns:
			none'''
	#X_train, Y_train, tokenizer = prepare_training_data_NOSPLIT(CORPUS_DF, max_word_length) #len8 experiment
	# Preprocessing
	X_train, X_test, Y_train, Y_test, tokenizer = prepare_training_data(CORPUS_DF)

	# Training model on train_split
	model = train_model(X_train, Y_train)
	
	# Testing on test_split
	test_accr = model.evaluate(X_test, Y_test) # test_accr = [loss, accuracy] on test data
	
	# Creating results file for test_split results
	test_dir = os.path.join('..', 'test_results', CORPUS)
	try:
		os.makedirs(test_dir)
	except FileExistsError:
		print("Directory {} already exists, proceeding with saving test results.".format(test_dir))
	test_out_filename = '{}_{}.csv'.format(NETWORK, HIDDEN_UNITS)
	test_out_path = os.path.join(test_dir, test_out_filename)
	
	# Saving test_split results
	test_out = open(test_out_path, 'w')
	test_out.write('Loss,Accuracy\n{:0.3f},{:0.3f}'.format(test_accr[0],test_accr[1]))
	test_out.close()
	
	# Set checkpoint file to refer to the best weights rather than the latest ones
	set_best_checkpoint()
	
def run_experiment(max_word_length=MAX_LEN, checkpoint_dir=CHECKPOINT_DIR):
	'''Makes the model classify experiment data specified in EXP_DF. Saves loss and accuracy in a '{NETWORK}_{HIDDEN_UNITS}.csv' file.
	Saves predictions per word in a 'detail_{NETWORK}_{HIDDEN_UNITS}.csv file in the shape of [preprocessed word], [predicted_value], [correct_value] for
	further analysis.
		args:
			max_word_length: Maximum wanted length of words
			corpus: Name of the training corpus
			network: Name of network architecture
			hidden_units: Number of hidden units in the LSTM, GRU or SRNN layer
		returns:
			None'''	
	# Building model and assigning the weights of the model specified in its checkpoint file: model with the lowest val_loss
	model = build_model()
	model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
	
	# Creating tokenizer from training corpus to apply to experiment corpus
	discard_matrix, discard_matrix, discard_Y_train, discard_Y_test, tokenizer = prepare_training_data(CORPUS_DF, max_word_length)
	# Preprocessing experiment data
	exp_sequences_matrix, Y = prepare_experiment_data(EXP_DF, tokenizer)
	
	# Complete evaluation
	exp_accr = model.evaluate(exp_sequences_matrix,Y,batch_size=BATCH,verbose=0) # exp_accr = [loss, accuracy] on experiment data
	# Classifying experiment words on a word by word basis for detailed results
	exp = model.predict(exp_sequences_matrix)
	
	# Creating results file
	exp_dir = os.path.join('..', 'exp_results', CORPUS, EXPERIMENT)
	try:
		os.makedirs(exp_dir)
	except FileExistsError:
		print("Directory {} already exists, proceeding with experiment.".format(exp_dir))

	exp_out_filename = '{}_{}.csv'.format(NETWORK, HIDDEN_UNITS)
	exp_out_path = os.path.join(exp_dir, exp_out_filename)
	
	# Saving results
	exp_out = open(exp_out_path, 'w')
	exp_out.write('Loss,Accuracy\n{:0.3f},{:0.3f}'.format(exp_accr[0],exp_accr[1]))	
	exp_out.close()
	
	# Creating detailed results file
	detail_out_filename = 'detail_{}_{}.csv'.format(NETWORK, HIDDEN_UNITS)
	detail_out_path = os.path.join(exp_dir, detail_out_filename)
	
	# Saving detailed results
	detail_out = open(detail_out_path, 'w')
	for i in range(len(exp)):
		detail_out.write('{},{},{}\n'.format(exp_sequences_matrix[i], exp[i][0], Y[i][0]))
	detail_out.close()
		
	return None

if MODE == 'train':
	train_test()
elif MODE == 'exp':
	run_experiment()
else:
	raise ValueError("{} is not an appropriate mode. Use 'train' or 'test' instead.".format(MODE))