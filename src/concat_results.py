# concat_results.py
# Creates a .csv overview over all model performances.
# Collecting calculated performance measures for all RNN models
import os
import pandas as pd

CORPORA = ['high','base','low']
EXPERIMENTS = ['ND','LRD']
OUTPATH_CSV = os.path.join('..', 'results')
OUTPATH_LATEX = os.path.join('..', 'latex', 'tab')
PERFORMANCE_COLUMNS = ['network', 'accuracy', 'precision', 'recall', 'f1_score', 'val_acc']

def read_checkpoint(ckpt_file):
	'''Reads 'checkpoint' file for epoch, train_loss, train_acc, val_loss and val_acc of the lowest val_loss model.
		args:
			ckpt_file: Path to the 'checkpoint' file
		returns:
			epoch: Epoch of the lowest val_loss model
			train_loss: Loss on training data of the lowest val_loss model
			train_acc: Accuracy on training data of the lowest val_loss model
			val_loss: Loss on validation data of the lowest val_loss model
			val_acc: Accuracy on validation data of the lowest val_loss model'''
	in_file = open(ckpt_file, 'r')
	lines = in_file.readlines()
	# Relevant field is in line 0 of checkpoint, second word and in quotation marks
	# Accessing relevant field and stripping quotation marks
	ckpt_string = lines[0].split()[1][1:-1] # "ckpt_epoch-train_loss-train_acc-val_loss-val_acc" without quotation marks
	ckpt_values = ckpt_string.split('_')[1]
	epoch = int(ckpt_values.split('-')[0])
	train_loss, train_acc, val_loss, val_acc = [float(value) for value in ckpt_values.split('-')[1:]]
	
	return epoch, train_loss, train_acc, val_loss, val_acc
		
def create_results(corpora=CORPORA, experiments=EXPERIMENTS):
	'''Collect all results across all corpora and experiments in a list of pd.DataFrames to create a dataframe containing all results. Used for data exploration and to create tables for the thesis.
			args:
				corpora: list of corpora to include in the big dataframe
				experiments: list of experiments to include in the big dataframe
			returns:
				results: dataframe with all specified results'''
	single_frames = []
	for corpus in corpora:
		for experiment in experiments:
			directory = os.path.join('..', 'exp_results', corpus, experiment)
			dirs = os.listdir(directory)
			for filename in dirs: # Going through results for all hidden_units configurations
				if not filename.startswith('detail_'): # Disregarding the detailed results
					network, hidden_units = filename[:-4].split('_')
					model_directory = os.path.join('..', 'saved_models', corpus, network, hidden_units)
					models = os.listdir(model_directory)
					ckpt = os.path.join(model_directory, models[0])
					epoch, train_loss, train_acc, val_loss, val_acc = read_checkpoint(ckpt)
					
					df = pd.read_csv(os.path.join(directory, filename),delimiter=',')
					df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
					df = df.rename(str.lower, axis='columns')
					
					# Putting all measured values for this model together
					df['val_acc'], df['val_loss'], df['train_acc'], df['train_loss'], df['epoch'], df['network'], df['experiment'], df['corpus'], df['hidden_units']  = val_acc, val_loss, train_acc, train_loss, epoch, network, experiment, corpus, hidden_units
					col_order = ['network', 'hidden_units', 'corpus', 'experiment', 'accuracy', 'precision', 'recall', 'f1_score', 'epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss']
					df = df[col_order]
					single_frames.append(df)
	# Join all small dfs into the complete overview of all results
	results = pd.concat(single_frames, ignore_index=True)
	results = results.sort_values(by='accuracy', ascending=False)
	
	return results
	
def save2file(dataframe, experiment, corpus, mode, outpath):
	'''Saves a given dataframe as either a .csv or .tex table. Latex formatting was then manually modified.
			args:
				dataframe: dataframe containing measurements
				experiment: string, name of the current experiment
				corpus: string, name of the current corpus
				mode: string, csv/latex, determines which file to generate
				outpath: string, determines where the file is saved_models
			returns:
				none'''
	try:
		os.makedirs(outpath)
	except:
		print("Directory {} exists, proceeding.".format(outpath))
	if mode=='csv':
		dataframe.to_csv(os.path.join(outpath, 'results_{}_{}.csv'.format(experiment, corpus)), index=False)
	elif mode=='latex': # Latex output has been used as a basis for tables in the thesis, heavy modifications were made
		dataframe.to_latex(os.path.join(outpath, 'results_{}_{}.tex'.format(experiment, corpus)), columns=PERFORMANCE_COLUMNS, float_format='{:0.3f}'.format, index=False)

def create_all_tables(df, mode):
	'''Creates tables collecting all results given the supplied mode.
			args:
				df: dataframe containing measurements
				mode: string, LRD/ND/SRNN/LSTM/GRU, determines what to filter the dataframe for
			returns:
				none'''
	corpora = ['base', 'low', 'high']
	measures = []
	if mode in ('SRNN', 'LSTM', 'GRU'):
		data = df.loc[df.network == mode]
		for corpus in corpora:
			data_corpus = data.loc[data.corpus == corpus]
			save2file(data_corpus, mode, corpus, 'csv', OUTPATH_CSV)
			print(mode, corpus)
			mean = data_corpus.mean()
			std = data_corpus.std()
			total = pd.concat([mean, std], axis=1)
			print(total.T)
			measures.append(total.T)
		overview = pd.concat(measures)
		print(overview)
	elif mode in ('LRD', 'ND'):
		data = df.loc[df.experiment == mode]
		for corpus in corpora:
			data_corpus = data.loc[data.corpus == corpus]
			save2file(data_corpus, mode, corpus, 'csv', OUTPATH_CSV)
	
results = create_results()
results['hidden_units'] = results['hidden_units'].astype(int)
results = results.sort_values(by=['network', 'hidden_units'], ascending=True)

create_all_tables(results, 'LRD')
create_all_tables(results, 'ND')
create_all_tables(results, 'SRNN')
create_all_tables(results, 'LSTM')
create_all_tables(results, 'GRU')