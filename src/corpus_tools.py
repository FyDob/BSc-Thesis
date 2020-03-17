# corpus_tools.py
# Works with a base file of correct Dyck words to create
# Generates classification datasets with a 50/50 split on correct/incorrect words each from a basic sampling of correct Dyck(2) words.
# Creates following datasets:
#	TRAINING
# 	- base
#	- high LRD
#	- low LRD
#	EXPERIMENTS
#	- LRD
#	- ND

import os
import sys
import random
import pickle
import numpy as np

INPUT_FILE = 'cumlen20_1180000.txt'
INPUT_PATH = os.path.join('..', 'corpus', INPUT_FILE)
SIZE = 1000000 # Training Corpus Size Target
LENGTH = int(INPUT_FILE[6:8]) # max length of word in set
POSITIVE_RATIO = 0.5
NEGATIVE_RATIO = 1-POSITIVE_RATIO
BD_CUTOFF = 19
MAX_BD_CUTOFF = 10
OUTPUT_TRAINING = os.path.join('..', 'training', 'base.csv')
OUTPUT_HIGH_LRD = os.path.join('..', 'training', 'high.csv')
OUTPUT_LOW_LRD = os.path.join('..', 'training', 'low.csv')
OUTPUT_LRD = os.path.join('..', 'experiment', 'LRD.csv')
OUTPUT_ND = os.path.join('..', 'experiment', 'ND.csv')

def samePair(char1, char2):
	'''Helper function to check whether the two characters are a valid bracket pair.
		args:
			char1: string, character in word
			char2: string, character in word
		returns:
			bool'''
	if char1 == '{' and char2 == '}':
		return True
	elif char1 == '[' and char2 == ']':
		return True
	else:
		return False

def maxNestingDepth(word):
	'''Calculates the maximum nesting depth within a D_2 word.
			args:
				word: string, Dyck word consisting of [, {, }, ] as brackets.
			returns:
				max_depth: int'''
	# Remove EOW symbol when working with processed corpus
	if word[-1] == '$':
		word = word[:-1]

	max_depth = 0
	depth = 0
	for character in word:
		if character == "[" or character == "{":
			depth += 1
		else: # Any other character must be a closing bracket and thus reduce depth
			depth -= 1
		if depth < 0:
			return -1 # Indicates a corrupted word with a superfluous closing bracket in analysis
			
		if depth > max_depth:
			max_depth = depth
			
	return max_depth
	
def maxValidNestingDepth(word):
	'''Calculates the maximum valid nesting depth within a word -- if the word was corrupted, negative nesting depth might occur.
	This function disregards that.
			args:
				word: string, Dyck word consisting of [, {, }, ] as brackets.
			returns:
				max_depth: int'''
	# Remove EOW symbol when working with processed corpus
	if word[-1] == '$':
		word = word[:-1]

	max_depth = 0
	depth = 0
	for character in word:
		if character == "[" or character == "{":
			depth += 1
		elif character == "]" or character == "}":
			depth -= 1
		else:
			continue
		if depth < 0:
			continue
			
		if depth > max_depth:
			max_depth = depth
			
	return max_depth

def nestingDepthAtPosition(word, position):
	'''Calculates the nesting depth of the character at word[position].
			args:
				word: string, Dyck word consisting of [, {, }, ] as brackets.
				position: int, index for word.
			returns:
				depth: int'''
	depth = 0
	for i in range(position):
		if word[i] == "[" or word[i] == "{":
			depth += 1
		else:
			depth -= 1
			
	return depth

def maxBracketDistance(word):
	'''returns maximum distance in characters between an opening and its corresponding closing bracket, i.e. maxBracketDistance('[{[]}]') is 4.
		args:
			word: string, Dyck word consisting of [, {, }, ] as brackets.
		returns:
			max(max_distance_square,max_distance_curly): int, maximum distance for either of the two bracket pair types = maximum distance in the word.'''
	# Remove EOW symbol when working with processed corpus.
	if word[-1] == '$':
		word = word[:-1]
	distance_square = 0
	max_distance_square = 0
	distance_curly = 0
	max_distance_curly = 0
	square_stack = []
	curly_stack = []
	
	# Counts 'seen' characters between two corresponding brackets
	for character in word:
		# Fill/empty the two stacks to keep count of unclosed brackets
		if character == "[":
			square_stack.append(character)
		elif character == "{":
			curly_stack.append(character)
		elif character == "]":
			square_stack.pop()
		else:
			curly_stack.pop()
			
		 # Check if stack is empty - means all opening brackets have been closed
		if not square_stack:
			# Save new max_distance_square and reset 'seen' characters counter
			if distance_square > max_distance_square:
				max_distance_square = distance_square - 1 # Adjust for first character being counted
			distance_square = 0
		else: # Increment 'seen' characters counter as long as there is still an unclosed bracket on the stack
			distance_square += 1
		
		# Same process as for square_stack
		if not curly_stack:
			if distance_curly > max_distance_curly:
				max_distance_curly = distance_curly - 1
			distance_curly = 0
		else:
			distance_curly += 1
			
	return max(max_distance_square, max_distance_curly)
	
def bracketDistanceAtPosition(word, position):
	'''returns distance from closing bracket at a given position to
	its corresponding opening bracket.
		args:
			word: string, Dyck word consisting of [, {, }, ] as brackets.
			position: position: int, index for word.
		returns:
			distance: int, number of characters between word[position] and the corresponding opening bracket.'''	
	distance = 0
	stack = []
	character = word[position]
	
	# Make sure a closing bracket is at the position. Otherwise, it's an opening bracket, for which the measure does not make sense
	if character == "]":
		opener = "["
	elif character == "}":
		opener = "{"
	else:
		return 0
	
	# Go backwards through the word, starting at position. Same logic applies as in maxBracketDistance(word)
	for i in range(position, -1, -1):
		if word[i] == opener:
			stack.pop()
		elif word[i] == character:
			stack.append(character)
			
		if not stack:
			return distance-1
		else:
			distance += 1

def determineError(word):
	'''Given a corrupted D_2 word, this function determines the kind of error - too many opening or too many closing brackets.
		args:
			word: corrupted Dyck-2 word
		returns:
			error: type of error'''
	open = ('[', '{')
	closed = (']', '}')
	o = 0
	c = 0
	for character in word:
		if character in open:
			o+=1
		elif character in closed:
			c+=1
		else:
			continue
	error = o-c
	if error < 0:
		return 'closed'
	elif error > 0:
		return 'open'
	else:
		return 'none'

def findErrorPosition(word):
	'''Given a corrupted D_2 word, this function determines the position of the corrupted bracket.
		args:
			word: corrupted Dyck-2 word
		returns:
			position: character position of corrupted bracket'''
	word = word.lstrip('0')
	# Stacks are filled with character positions. If a stack is not empty by the time the word is over,
	# the remaining position is the error position.
	# If there is an attempt to pop an empty stack before the word is over, the current position is the error position.
	square_stack = []
	curly_stack = []
	
	for i in range(len(word)):
		if word[i] == '[':
			square_stack.append(i)
		elif word[i] == '{':
			curly_stack.append(i)
		elif word[i] == ']':
			if square_stack:
				square_stack.pop()
			else:
				return i
		elif word[i] == '}':
			if curly_stack:
				curly_stack.pop()
			else:
				return i
		else:
			continue
	if square_stack:
		return square_stack[0]
	elif curly_stack:
		return curly_stack[0]
	else:
		raise ValueError("Encountered unknown corruption in word\n{}".format(word))

def measureLength(corpus):
	'''Calculates average length and variance thereof for all words in the corpus.
		args:
			corpus: list of strings, list of Dyck words.
		returns:
			avg: float, average length of words in corpus.
			var: float, variance of length of words in corpus.'''
	if len(corpus) == 0:
		return 0., 0.
	total = []
	for word in corpus:
		total.append(len(word))
	avg = sum(total)/float(len(total))
	var = sum((length - avg)**2 for length in total)/float(len(total))
	
	return avg, var
	
def measureMaxNestingDepth(corpus):
	'''Calculates average maximum nesting depth and variance thereof for all words in the corpus.
		args:
			corpus: list of strings, list of Dyck words.
		returns:
			avg: float, average maximum nesting depth of words in corpus.
			var: float, variance of maximum nesting depth of words in corpus.'''
	if len(corpus) == 0:
		return 0., 0.
	total = []
	for word in corpus:
		total.append(maxNestingDepth(word))
	avg = sum(total)/float(len(total))
	var = sum((depth - avg)**2 for depth in total)/float(len(total))
	
	return avg, var
	
def measureMaxBracketDistance(corpus):
	'''Calculates average maximum bracket distance and variance thereof for all words in the corpus.
		args:
			corpus: list of strings, list of Dyck words.
		returns:
			avg: float, average maximum bracket distance of words in corpus.
			var: float, variance of maximum bracket distance of words in corpus.'''
	if len(corpus) == 0:
		return 0., 0.
	total = []
	for word in corpus:
		total.append(maxBracketDistance(word))
	avg = sum(total)/float(len(total))
	var = sum((dist - avg)**2 for dist in total)/float(len(total))
	
	return avg, var
	
def evaluateCorpus(corpus):
	'''Calculates average and variance for three measures over all words in a corpus: word length, maximum nesting depth and maximum bracket distance.
		args:
			corpus: 
		returns:
			3 x (avg, var): 3 tuples of floats, average and variance for the respective measure.'''
	corpus = [entry[0][:-1] for entry in corpus if entry[1]] # Removing EOW
	
	return measureLength(corpus), measureMaxNestingDepth(corpus), measureMaxBracketDistance(corpus)

def printStats(size, avgLen, varLen, avgMaxND, varMaxND, avgMaxBD, varMaxBD):
	'''Prints a table with all calculated corpus stats to the console. Table is for copy-pasting into .tex file.
		args:
			size: int, number of words in the corpus
			avgLen: float, average length of word in the corpus
			varLen: float, variation of length of words in the corpus
			avgMaxND: float, average maximum nesting depth of word in the corpus
			varMaxND: float, variation of maximum nesting depth of words in the corpus
			avgMaxBD: float, average maximum bracket depth of word in the corpus
			varMaxBD: float, variation of maximum bracket depth of words in the corpus
		returns:
			none'''
	print("Size\tAvg Length \t Avg MaxNestDepth \t Avg MaxBrackDist")
	print("{}\t${:3.2f}$ (${:3.2f}$) & ${:3.2f}$ (${:3.2f}$) & ${:3.2f}$ (${:3.2f}$)".format(size, avgLen, varLen, avgMaxND, varMaxND, avgMaxBD, varMaxBD))

def largerBD(corpus):
	'''Increases average bracket distance for a corpus by finding words with a low maximum bracket distance, deleting the lowest distance pair from the word and then wrapping the word in a matching pair of brackets.
		args:
			corpus: list of strings, list of Dyck words.
		returns:
			corpus: list of strings, list of Dyck words.'''
	# 'Word collectors' are initialized as lists to allow iteration through them
	small_BD = []
	big_BD = []
	
	# Find words with low maximum bracket distance
	for entry in corpus:
		word = entry[0][:-1]
		correct = entry[1]
		if correct:
			maxBD = maxBracketDistance(word)
			if maxBD < BD_CUTOFF:
				small_BD.append(word)
			elif maxBD >= BD_CUTOFF:
				big_BD.append(word)
	
	# Modify low maximum bracket distance words
	for word in small_BD:
		prev_ND = 0 # Nesting depth to compare to
		for i in range(len(word)):
			# Calculate nesting depth at each position of the word. Once it decreases, a closing bracket has been found
			ND = nestingDepthAtPosition(word, i)
			if ND < prev_ND:
				# Check if this position belongs to a bracket pair eligible for deletion - only {} and [] are eligible, since they have the shortest possible bracket distance
				char = word[i-1]
				prev_char = word[i-2]
				if samePair(prev_char, char):
					bracket = random.randint(0, 1)
					if bracket:
						word = '[' + word[:i-2] + word[i:] + ']'
						break
					else:
						word = '{' + word[:i-2] + word[i:] + '}'
						break
			prev_ND = ND # Update nesting depth
		big_BD.append(word) # Populate modified list

	BD_set = set(big_BD) # Fast deletion of duplicates.
	BD_list = list(BD_set)
	incorrect = [entry for entry in corpus if entry[1]==0]
	BD = [[word+'$',1] for word in BD_list] # Complete newly created and old correct words with EOW and class
	BD = BD + incorrect
	random.shuffle(BD)
	
	return BD

def smallerBD(corpus):
	'''Decreases average bracket distance for a corpus by finding words with a high maximum bracket distance. In those words, the pair with the highest maximum bracket distance is found. The opening bracket is then moved right in front of the closing bracket.
		args:
			corpus: list of strings, list of Dyck words.
		returns:
			corpus: list of strings, list of Dyck words.'''
	# 'Word collectors' are initialized as lists to allow iteration through them
	big_BD = []
	small_BD = []
	
	# Find words with high maximum bracket distance
	for entry in corpus:
		word = entry[0][:-1]
		correct = entry[1]
		if correct:
			maxBD = maxBracketDistance(word)
			if maxBD > MAX_BD_CUTOFF:
				big_BD.append(word)
			elif maxBD <= MAX_BD_CUTOFF:
				small_BD.append(word) # big_BD only features word entries, since all big_BD entries are correct
	
	for word in big_BD:
		max_BD = 0
		max_pos = 0
		for i in range(len(word)):
			# Calculate bracket distance at each position of the word. Once the maximum bracket distance has been found, the position of the closing bracket is recorded
			BD = bracketDistanceAtPosition(word, i)
			if BD > max_BD:
				max_pos = i
				max_BD = BD
		closer = max_pos # Position of longest distance closing bracket
		opener = max_pos - max_BD - 1 # Fix off by one return of bd@pos
		# New word is created by deleting the opener from its original position and moving it right in front of the closing bracket
		# This ensures grammaticality and reduces maxBracketDistance
		new_word = word[:opener] + word[opener+1:closer] + word[opener] + word[closer:]
		small_BD.append(new_word)
	
	BD_list = list(set(small_BD))
	incorrect = [entry for entry in corpus if entry[1]==0]
	BD = [[word+'$',1] for word in BD_list] # Complete newly created and old correct words with EOW and class
	BD = BD + incorrect
	
	return BD

def corrupt_words(correct_corpus, mode):
	'''Turns correct Dyck_2 words into incorrect ones by replacing a random opening bracket with a random closing one or vice versa, depending on the mode.
		args:
			correct_corpus: list of correct word-class tuples
			mode: string. "open"/"close" - determines brackets being changed
		returns:
			incorrect: list of incorrect word-class tuples'''
	incorrect = set() # set ensures uniqueness
	if mode == "open":
		replace = ("{","[")
		replacement = ["]","}"]
	else:
		replace = ("}","]")
		replacement = ["{","["]
	
	for entry in correct_corpus:
		word = entry[0]
		changed = 0
		c = 0
		while not changed:
			c += 1
			id = random.randint(0, len(word)-1)
			if word[id] in replace:
				closing = random.choice(replacement)
				new_word = word[:id] + closing + word[id+1:]
				assert len(new_word)==len(word)
				incorrect.add(new_word)
				changed = 1
			elif c >= 1000000:
				break
	incorrect = [[word,0] for word in incorrect]
	
	return incorrect
	
def create_LRD(base, subword_length):
	'''Creates the dataset for the extreme long range dependency (LRD) experiment/experiment 1.
		args:
			base: list of correct word-class tuples
		returns:
			LRD: list of correct LRD word-class tuples'''
	LRD = set()
	base = [entry[0] for entry in base if len(entry[0])==subword_length]
	for i in range(len(base)*5):
		bracket = random.randint(0,1)
		w1 = random.sample(base, 1)[0][:-1]
		w2 = random.sample(base, 1)[0][:-1]
		if bracket:
			new_word = '[' + w1 + w2+ ']$'
		else:
			new_word = '{' + w1 + w2 + '}$'
		LRD.add(new_word)
	LRD = [[word,1] for word in LRD]
	
	return LRD
		
def create_ND(base, infix_length):
	'''Creates the dataset for the extreme new depths (ND) experiment/experiment 2.
		args:
			base: list of correct word-class tuples
		returns:
			ND: list of correct LRD word-class tuples'''
	ND = set()
	base = [entry[0] for entry in base if len(entry[0])==infix_length]
	for i in range(len(base)*5):
		new_word = random.sample(base, 1)[0][:-1]
		for i in range(5):
			bracket = random.randint(0,1)
			if bracket:
				new_word = '[' + new_word + ']'
			else:
				new_word = '{' + new_word + '}'
		ND.add(new_word + '$')
	ND = [[word,1] for word in ND]
	
	return ND

def create_corpus(data, type):
	'''Creates full corpora for a classification task. Increases or lowers bracket distance for the high/low LRD training corpora.
		Prints corpus stats to the console, with the values in LaTeX formatting.
			args:
				data: list of [word,correct-bool] pairs, filled with correct generated D_2 words
				type: string, high/low/[misc], triggers increasing/decreasing/disregarding average bracket distance across the corpus
			returns:
				corpus: list of [word,correct-bool] pairs, filled with a 1:0.5:0.5 ratio of correct:incorrect_open:incorrect_closed words'''
	if type == 'high':
		corpus_correct = largerBD(data)
	elif type == 'low':
		corpus_correct = smallerBD(data)
	else:
		corpus_correct = data
	corpus_incorrect_open = corrupt_words(corpus_correct, 'open')
	corpus_incorrect_closed = corrupt_words(corpus_correct, 'closed')
	# Debug print
	#print(" === {} ===\nCorrect\tIncorrect O\tIncorrect C\n{}\t{}\t{}".format(type.upper(), len(data), len(corpus_incorrect_open), len(corpus_incorrect_closed)))
	corpus = data[:int(POSITIVE_RATIO*SIZE)] + corpus_incorrect_open[:int(NEGATIVE_RATIO/2.*SIZE)] + corpus_incorrect_closed[:int(NEGATIVE_RATIO/2.*SIZE)]
	random.shuffle(corpus)
	(avLen, varLen), (avMaxND, varMaxND), (avMaxBD, varMaxBD) = evaluateCorpus(corpus)
	printStats(len(corpus), avLen, varLen, avMaxND, varMaxND, avMaxBD, varMaxBD)
	
	return corpus
		

def save2file(outpath, corpus):
	'''Saves the corpus as a .csv file to the specified path.
			args:
				outpath: filepath of the output file
				corpus: list of [word,correct-bool] pairs, to be used for training/experiments on RNNs
			returns:
				none'''
	outfile = open(outpath, 'w')
	outfile.write('word,value\n')
	for entry in corpus:
		outfile.write('{},{}\n'.format(entry[0],entry[1]))

def create_corpora():
	'''Creates all datasets needed for the classification tasks from the input file.
			args:
				none
			returns:
				none'''
	file = open(INPUT_PATH, 'r')
	EOW = '$'
	raw_text = file.read()

	print("Creating Base...")
	raw_classified = [[word+EOW,1] for word in raw_text.split(EOW)]

	print("Corrupting Base...")
	base = create_corpus(raw_classified, 'base')
	save2file(OUTPUT_TRAINING, base)

	print("Corrupting High LRD...")
	highLRD = create_corpus(raw_classified, 'high')
	save2file(OUTPUT_HIGH_LRD, highLRD)

	print("Corrupting Low LRD...")
	lowLRD = create_corpus(raw_classified, 'low')
	save2file(OUTPUT_LOW_LRD, lowLRD)

	print("Creating LRD...")
	LRD_correct = create_LRD(raw_classified, LENGTH-2+1)
	LRD = create_corpus(LRD_correct, 'LRD')
	save2file(OUTPUT_LRD, LRD)

	print("Creating ND...")
	ND_correct = create_ND(raw_classified, LENGTH+1)
	ND = create_corpus(ND_correct, 'ND')
	save2file(OUTPUT_ND, ND)