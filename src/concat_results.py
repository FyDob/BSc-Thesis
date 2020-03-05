# Creates a .csv overview over all model performances.
# Collecting acquired loss, accuracy values for RNN models on experiment data.
# TODO: Include test_loss, test_val
import os
import pandas as pd

corpora = ['high','base','low']
experiments = ['ND','LRD']

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
		

# Collect all results across all corpora and experiments in a list of pd.DataFrames
single_frames = []
for corpus in corpora:
	for experiment in experiments:
		directory = os.path.join('..', 'exp_results', corpus, experiment)
		dirs = os.listdir(directory)
		for filename in dirs:
			if not filename.startswith('detail_'):
				network, hidden_units = filename[:-4].split('_')
				model_directory = os.path.join('..', 'saved_models', corpus, network, hidden_units)
				models = os.listdir(model_directory)
				file = os.path.join(model_directory, models[0])
				epoch, train_loss, train_acc, val_loss, val_acc = read_checkpoint(file)
				#print(epoch, train_loss, train_acc, val_loss, val_acc)
				df = pd.read_csv(os.path.join(directory, filename),delimiter=',')
				df['Val_Acc'], df['Val_Loss'], df['Train_Acc'], df['Train_Loss'], df['Epoch'], df['Network'], df['Experiment'], df['Corpus']  = val_acc, val_loss, train_acc, train_loss, epoch, network, experiment, corpus
				single_frames.append(df)
				
# Join all small dfs into the complete overview of all results
results = pd.concat(single_frames, ignore_index=True)
results = results.rename(str.lower, axis='columns')
results = results.sort_values(by='accuracy', ascending=False)
outpath = os.path.join('..', 'results')
try:
	os.makedirs(outpath)
except:
	print("Directory {} exists, proceeding.".format(outpath))
results.to_csv(os.path.join(outpath, 'results.csv'), index=False)