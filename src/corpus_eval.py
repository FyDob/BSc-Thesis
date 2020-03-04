# Generate three different training corpora based on a previously generated text file containing Dyck words.

import numpy as np

infile = open('./cum_gen/cumlen20_140000_AvMaxNestDepth4.0_AvMaxBrackDist13.0.txt', 'r')
out_NOMOD = 'cumlen20_nomod.txt'
out_LONG = 'cumlen20_long.txt'
out_SHORT = 'cumlen20_short.txt'
BD_CUTOFF = 18
MAX_BD_CUTOFF = 14

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
			raise ValueError('Invalid Dyck word detected; negative nesting depth.')
			
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
			corpus: list of strings, list of Dyck words.
		Returns:
			(avg, var): 3 tuples of floats, average and variance for the respective measure.
	'''
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
	for word in words:
		if maxBracketDistance(word) < BD_CUTOFF:
			small_BD.append(word)
	
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
					random_float = np.random.uniform(0.0, 1.0)
					if random_float < 0.5:
						word = '[' + word[:i-2] + word[i:] + ']'
						break
					else:
						word = '{' + word[:i-2] + word[i:] + '}'
						break
			prev_ND = ND # Update nesting depth.
		big_BD.append(word) # Populate modified list.

	# Fill up corpus with remaining unmodified words.
	for word in words:
		if maxBracketDistance(word) >= BD_CUTOFF:
			big_BD.append(word)

	BD_set = set(big_BD) # Fast deletion of duplicates.	
	return list(BD_set)

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
	for word in words:
		if maxBracketDistance(word) > MAX_BD_CUTOFF:
			big_BD.append(word)
	
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
	
	# Fill up corpus with remaining unmodified words.
	for word in words:
		if maxBracketDistance(word) <= MAX_BD_CUTOFF:
			small_BD.append(word)

	BD_set = set(small_BD) # Fast deletion of duplicates.
	return list(BD_set)

# Reading input file.
# words = []
# for line in infile:
	# raw_words = line.split('$')
	# for word in raw_words:
		# words.append(word)
# infile.close()
# words = words[:-1] # Remove empty word.

# # Due to both the random nature of generating words in the initial corpus as well as the interim existence of the other two corpora as sets, simply slicing the lists actually removes words at random.
# print("LONG")
# bigBD = largerBD(words)
# bigBD = bigBD[:512*170]
# (avLen, varLen), (avMaxND, varMaxND), (avMaxBD, varMaxBD) = evaluateCorpus(bigBD)
# printStats(len(bigBD), avLen, varLen, avMaxND, varMaxND, avMaxBD, varMaxBD)

# print("SHORT")
# smallBD = smallerBD(words)
# smallBD = smallBD[:512*170]
# (avLen, varLen), (avMaxND, varMaxND), (avMaxBD, varMaxBD) = evaluateCorpus(smallBD)
# printStats(len(smallBD), avLen, varLen, avMaxND, varMaxND, avMaxBD, varMaxBD)

# print("NOMOD")
# words = words[:512*170]
# (avLen, varLen), (avMaxND, varMaxND), (avMaxBD, varMaxBD) = evaluateCorpus(words)
# printStats(len(words), avLen, varLen, avMaxND, varMaxND, avMaxBD, varMaxBD)

# # Save corpora for training.
# out = open(out_NOMOD, 'w')
# for word in words:
	# out.write(word+'$')
# out.close()

# out = open(out_LONG, 'w')
# for word in bigBD:
	# out.write(word+'$')
# out.close()

# out = open(out_SHORT, 'w')
# for word in smallBD:
	# out.write(word+'$')
# out.close()