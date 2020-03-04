import sys
import random
import pickle
import numpy as np

INPUT_FILE = 'cumlen20_1180000.txt' #sys.argv[1]
SIZE = 1000000 # Training Corpus Size Target
LENGTH = int(INPUT_FILE[6:8]) # max length of word in set
POSITIVE_RATIO = 0.5
NEGATIVE_RATIO = 1-POSITIVE_RATIO
BD_CUTOFF = 19
MAX_BD_CUTOFF = 10
OUTPUT_TRAINING = '../training/base.csv'
OUTPUT_HIGH_LRD = '../training/high.csv'
OUTPUT_LOW_LRD = '../training/low.csv'
OUTPUT_LRD = '../experiment/LRD.csv'
OUTPUT_ND = '../experiment/ND.csv'

def samePair(char1, char2):
	'''Helper function to check whether the two characters are a valid bracket pair.
		Args:
			char1: string, character in word
			char2: string, character in word
		Returns:
			bool
	'''
	if char1 == '{' and char2 == '}':
		return True
	elif char1 == '[' and char2 == ']':
		return True
	else:
		return False

def maxNestingDepth(word):
	'''Calculates the maximum nesting depth within a D_2 word.
			Args:
				word: string, Dyck word consisting of [, {, }, ] as brackets.
			Returns:
				max_depth: int
	'''
	max_depth = 0
	depth = 0
	for character in word:
		if character == "[" or character == "{":
			depth += 1
		else: # Any other character must be a closing bracket and thus reduce depth.
			depth -= 1
		if depth < 0:
			raise ValueError('Invalid Dyck word detected; negative nesting depth.\n{}'.format(word))
			
		if depth > max_depth:
			max_depth = depth
	return max_depth

def nestingDepth(word, position):
	'''Calculates the nesting depth of the character at word[position].
			Args:
				word: string, Dyck word consisting of [, {, }, ] as brackets.
				position: int, index for word.
			Returns:
				depth: int
	'''
	depth = 0
	for i in range(position):
		if word[i] == "[" or word[i] == "{":
			depth += 1
		else:
			depth -= 1
	return depth

def maxBracketDistance(word):
	'''Returns maximum distance in characters between an opening and its corresponding closing bracket, i.e. maxBracketDistance('[{[]}]') is 4.
		Args:
			word: string, Dyck word consisting of [, {, }, ] as brackets.
		Returns:
			max(max_distance_square,max_distance_curly): int, maximum distance for either of the two bracket pair types = maximum distance in the word.
	'''		
	distance_square = 0
	max_distance_square = 0
	distance_curly = 0
	max_distance_curly = 0
	square_stack = []
	curly_stack = []
	
	# Counts 'seen' characters between two corresponding brackets.
	for character in word:
		# Fill/empty the two stacks to keep count of unclosed brackets.
		if character == "[":
			square_stack.append(character)
		elif character == "{":
			curly_stack.append(character)
		elif character == "]":
			square_stack.pop()
		else:
			curly_stack.pop()
			
		 # Check if stack is empty - means all opening brackets have been closed.
		if not square_stack:
			# Save new max_distance_square and reset 'seen' characters counter.
			if distance_square > max_distance_square:
				max_distance_square = distance_square - 1 # Adjust for first character being counted.
			distance_square = 0
		else: # Increment 'seen' characters counter as long as there is still an unclosed bracket on the stack.
			distance_square += 1
		
		# Same process as for square_stack.
		if not curly_stack:
			if distance_curly > max_distance_curly:
				max_distance_curly = distance_curly - 1
			distance_curly = 0
		else:
			distance_curly += 1
	return max(max_distance_square, max_distance_curly)
	
def bracketDistanceAtPosition(word, position):
	'''Returns distance from closing bracket at a given position to
	its corresponding opening bracket.
		Args:
			word: string, Dyck word consisting of [, {, }, ] as brackets.
			position: position: int, index for word.
		Returns:
			distance: int, number of characters between word[position] and the corresponding opening bracket.
	'''
		
	distance = 0
	stack = []
	character = word[position]
	
	# Make sure a closing bracket is at the position. Otherwise, it's an opening bracket, for which the measure does not make sense.
	if character == "]":
		opener = "["
	elif character == "}":
		opener = "{"
	else:
		return 0
	
	# Go backwards through the word, starting at position. Same logic applies as in maxBracketDistance(word).
	for i in range(position, -1, -1):
		if word[i] == opener:
			stack.pop()
		elif word[i] == character:
			stack.append(character)
			
		if not stack:
			return distance-1
		else:
			distance += 1
	# If the stack is not empty after seeing the whole word, the word must be invalid.
	raise ValueError('Invalid Dyck word detected; no closing bracket found.')
		
def measureLength(corpus):
	'''Calculates average length and variance thereof for all words in the corpus.
		Args:
			corpus: list of strings, list of Dyck words.
		Returns:
			avg: float, average length of words in corpus.
			var: float, variance of length of words in corpus.
	'''
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
		Args:
			corpus: list of strings, list of Dyck words.
		Returns:
			avg: float, average maximum nesting depth of words in corpus.
			var: float, variance of maximum nesting depth of words in corpus.
	'''
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
		Args:
			corpus: list of strings, list of Dyck words.
		Returns:
			avg: float, average maximum bracket distance of words in corpus.
			var: float, variance of maximum bracket distance of words in corpus.
	'''
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
		Args:
			corpus: 
		Returns:
			(avg, var): 3 tuples of floats, average and variance for the respective measure.
	'''
	corpus = [entry[0][:-1] for entry in corpus if entry[1]] # Removing EOW.
	return measureLength(corpus), measureMaxNestingDepth(corpus), measureMaxBracketDistance(corpus)

def printStats(size, avgLen, varLen, avgMaxND, varMaxND, avgMaxBD, varMaxBD):
	'''Prints a table with all calculated corpus stats to the console. Table is for copy-pasting into .tex file.
		Args:
			size: int, number of words in the corpus
			avgLen: float, average length of word in the corpus
			varLen: float, variation of length of words in the corpus
			avgMaxND: float, average maximum nesting depth of word in the corpus
			varMaxND: float, variation of maximum nesting depth of words in the corpus
			avgMaxBD: float, average maximum bracket depth of word in the corpus
			varMaxBD: float, variation of maximum bracket depth of words in the corpus
		Returns:
			none
	'''
	print("Size\tAvg Length \t Avg MaxNestDepth \t Avg MaxBrackDist")
	print("{}\t${:3.2f}$ (${:3.2f}$) & ${:3.2f}$ (${:3.2f}$) & ${:3.2f}$ (${:3.2f}$)".format(size, avgLen, varLen, avgMaxND, varMaxND, avgMaxBD, varMaxBD))

def largerBD(corpus):
	'''Increases average bracket distance for a corpus by finding words with a low maximum bracket distance, deleting the lowest distance pair from the word and then wrapping the word in a matching pair of brackets.
		Args:
			corpus: list of strings, list of Dyck words.
		Returns:
			corpus: list of strings, list of Dyck words.
	'''
	# 'Word collectors' are initialized as lists to allow iteration through them.
	small_BD = []
	big_BD = []
	
	# Find words with low maximum bracket distance.
	for entry in corpus:
		word = entry[0][:-1]
		correct = entry[1]
		if correct:
			maxBD = maxBracketDistance(word)
			if maxBD < BD_CUTOFF:
				small_BD.append(word)
			elif maxBD >= BD_CUTOFF:
				big_BD.append(word)
	
	# Modify low maximum bracket distance words.
	for word in small_BD:
		prev_ND = 0 # Nesting depth to compare to.
		for i in range(len(word)):
			# Calculate nesting depth at each position of the word. Once it decreases, a closing bracket has been found.
			ND = nestingDepth(word, i)
			if ND < prev_ND:
				# Check if this position belongs to a bracket pair eligible for deletion - only {} and [] are eligible, since they have the shortest possible bracket distance.
				char = word[i-1]
				prev_char = word[i-2]
				if samePair(prev_char, char):
					#print("Replacing.")
					bracket = random.randint(0, 1)
					if bracket:
						word = '[' + word[:i-2] + word[i:] + ']'
						break
					else:
						word = '{' + word[:i-2] + word[i:] + '}'
						break
			prev_ND = ND # Update nesting depth.
		big_BD.append(word) # Populate modified list.

	BD_set = set(big_BD) # Fast deletion of duplicates.	
	BD_list = list(BD_set)
	incorrect = [entry for entry in corpus if entry[1]==0]
	BD = [[word+'$',1] for word in BD_list] # Complete newly created and old correct words with EOW and class.
	BD = BD + incorrect
	random.shuffle(BD)
	return BD

def smallerBD(corpus):
	'''Decreases average bracket distance for a corpus by finding words with a high maximum bracket distance. In those words, the pair with the highest maximum bracket distance is found. The opening bracket is then moved right in front of the closing bracket.
		Args:
			corpus: list of strings, list of Dyck words.
		Returns:
			corpus: list of strings, list of Dyck words.
	'''
	# 'Word collectors' are initialized as lists to allow iteration through them.
	big_BD = []
	small_BD = []
	
	# Find words with high maximum bracket distance.
	for entry in corpus:
		word = entry[0][:-1]
		correct = entry[1]
		if correct:
			maxBD = maxBracketDistance(word)
			if maxBD > MAX_BD_CUTOFF:
				big_BD.append(word)
			elif maxBD <= MAX_BD_CUTOFF:
				small_BD.append(word) # big_BD only features word entries, since all big_BD entries are correct.
	
	for word in big_BD:
		max_BD = 0
		max_pos = 0
		for i in range(len(word)):
			# Calculate bracket distance at each position of the word. Once the maximum bracket distance has been found, the position of the closing bracket is recorded.
			BD = bracketDistanceAtPosition(word, i)
			if BD > max_BD:
				max_pos = i
				max_BD = BD
		closer = max_pos # Position of longest distance closing bracket.
		opener = max_pos - max_BD - 1 # Fix off by one return of bd@pos
		# New word is created by deleting the opener from its original position and moving it right in front of the closing bracket.
		# This ensures grammaticality and reduces maxBracketDistance.
		new_word = word[:opener] + word[opener+1:closer] + word[opener] + word[closer:]
		small_BD.append(new_word)
	
	BD_list = list(set(small_BD))
	incorrect = [entry for entry in corpus if entry[1]==0]
	BD = [[word+'$',1] for word in BD_list] # Complete newly created and old correct words with EOW and class.
	BD = BD + incorrect
	return BD

def corrupt_words(correct_corpus, mode):
	''' Args:
			correct_corpus: list of correct word-class tuples
			mode: string. "open"/"close" - determines brackets being changed
		Returns:
			incorrect: list of incorrect word-class tuples
	'''
	incorrect = set() # set to ensure uniqueness
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
	''' Args:
			base: list of correct word-class tuples
		Returns:
			LRD: list of correct LRD word-class tuples
	'''
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
	''' Args:
			base: list of correct word-class tuples
		Returns:
			ND: list of correct LRD word-class tuples
	'''
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

def save2file(outpath, corpus):
	outfile = open(outpath, 'w')
	outfile.write('word,value\n')
	for entry in corpus:
		outfile.write('{},{}\n'.format(entry[0],entry[1]))

file = open(INPUT_FILE, 'r')
EOW = '$'
raw_text = file.read()

# print("Creating Base...")
# base_correct = [[word+EOW,1] for word in raw_text.split(EOW)]
# print("Corrupting...")
# base_incorrect_open = corrupt_words(base_correct, "open")
# base_incorrect_closed = corrupt_words(base_correct, "closed")
# print(" === BASE ===\nCorrect\tIncorrect O\tIncorrect C\n{}\t{}\t{}".format(len(base_correct), len(base_incorrect_open), len(base_incorrect_closed)))
# base = base_correct[:int(POSITIVE_RATIO*SIZE)] + base_incorrect_open[:int(NEGATIVE_RATIO/2.*SIZE)] + base_incorrect_closed[:int(NEGATIVE_RATIO/2.*SIZE)]
# random.shuffle(base)
# (avLen, varLen), (avMaxND, varMaxND), (avMaxBD, varMaxBD) = evaluateCorpus(base)
# printStats(len(base), avLen, varLen, avMaxND, varMaxND, avMaxBD, varMaxBD)
# save2file(OUTPUT_TRAINING, base)

# print("High LRD...")
# highLRD_correct = largerBD(base_correct)
# print("Corrupting...")
# highLRD_incorrect_open = corrupt_words(highLRD_correct, "open")
# highLRD_incorrect_closed = corrupt_words(highLRD_correct, "closed")
# print(" === HIGH LRD ===\nCorrect\tIncorrect O\tIncorrect C\n{}\t{}\t{}".format(len(highLRD_correct), len(highLRD_incorrect_open), len(highLRD_incorrect_closed)))
# highLRD = highLRD_correct[:int(POSITIVE_RATIO*SIZE)] + highLRD_incorrect_open[:int(NEGATIVE_RATIO/2.*SIZE)] + highLRD_incorrect_closed[:int(NEGATIVE_RATIO/2.*SIZE)]
# random.shuffle(highLRD)
# (avLen, varLen), (avMaxND, varMaxND), (avMaxBD, varMaxBD) = evaluateCorpus(highLRD)
# printStats(len(highLRD), avLen, varLen, avMaxND, varMaxND, avMaxBD, varMaxBD)
# save2file(OUTPUT_HIGH_LRD, highLRD)

# print("Low LRD...")
# lowLRD_correct = smallerBD(base_correct)
# print("Corrupting...")
# lowLRD_incorrect_open = corrupt_words(lowLRD_correct, "open")
# lowLRD_incorrect_closed = corrupt_words(lowLRD_correct, "closed")
# print(" === LOW LRD ===\nCorrect\tIncorrect O\tIncorrect C\n{}\t{}\t{}".format(len(lowLRD_correct), len(lowLRD_incorrect_open), len(lowLRD_incorrect_closed)))
# lowLRD = lowLRD_correct[:int(POSITIVE_RATIO*SIZE)] + lowLRD_incorrect_open[:int(NEGATIVE_RATIO/2.*SIZE)] + lowLRD_incorrect_closed[:int(NEGATIVE_RATIO/2.*SIZE)]
# random.shuffle(lowLRD)
# (avLen, varLen), (avMaxND, varMaxND), (avMaxBD, varMaxBD) = evaluateCorpus(lowLRD)
# printStats(len(lowLRD), avLen, varLen, avMaxND, varMaxND, avMaxBD, varMaxBD)
# save2file(OUTPUT_LOW_LRD, lowLRD)

# TODO import base corpus from file to create shorter LRD/ND training batches (in total just len 22)
print("Creating LRD...")
LRD_correct = create_LRD(base_correct, LENGTH-2+1)
LRD_incorrect_open = corrupt_words(LRD_correct, "open")
LRD_incorrect_closed = corrupt_words(LRD_correct, "closed")
print(" === LRD ===\nCorrect\tIncorrect O\tIncorrect C\n{}\t{}\t{}".format(len(LRD_correct), len(LRD_incorrect_open), len(LRD_incorrect_closed)))
LRD = LRD_correct[:int(POSITIVE_RATIO*SIZE)] + LRD_incorrect_open[:int(NEGATIVE_RATIO/2.*SIZE)] + LRD_incorrect_closed[:int(NEGATIVE_RATIO/2.*SIZE)]
random.shuffle(LRD)
save2file(OUTPUT_LRD, LRD)

print("Creating ND...")
ND_correct = create_ND(base_correct, LENGTH+1)
ND_incorrect_open = corrupt_words(ND_correct, "open")
ND_incorrect_closed = corrupt_words(ND_correct, "closed")
print(" === ND ===\nCorrect\tIncorrect O\tIncorrect C\n{}\t{}\t{}".format(len(ND_correct), len(ND_incorrect_open), len(ND_incorrect_closed)))
ND = ND_correct[:int(POSITIVE_RATIO*SIZE)] + ND_incorrect_open[:int(NEGATIVE_RATIO/2.*SIZE)] + ND_incorrect_closed[:int(NEGATIVE_RATIO/2.*SIZE)]
random.shuffle(ND)
save2file(OUTPUT_ND, ND)