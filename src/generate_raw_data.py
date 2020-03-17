# generate_raw_data.py
# Script to sample words from Dyck(2) according to Sennhauser & Berwick 2018.
#
# Grammar:
# S -> Z S | Z
# Z -> B | T
# B -> [ S ] | { S }
# T -> [] | {}
#
# branch = Z -> B
# concat = S -> Z S
# s(l) = min(1, -3 * (l/n) + 3)
# P_branch = r_b * s(l); r_b ~ U(0.4, 0.8)
# P_concat = r_c * s(l); r_c ~ U(0.4, 0.8)
# l = number of already generated characters in the sentence, n = total goal length
# https://en.wikipedia.org/wiki/Uniform_distribution_(continuous) == U(0.4, 0.8)

import sys
import numpy as np
import os

LIMIT = int(sys.argv[1])
GOAL_CORPUS_SIZE = int(sys.argv[2])

def countCharacters(word):
	'''Counts the number of generated terminal symbols in a string.
			args:
				word: string, string of terminal and non-terminal symbols.
			returns:
				generated_characters: int, number of generated terminal symbols in word.'''
	generated_characters = 0
	for character in word:
		if character in {"[","]","{","}"}:
			generated_characters +=  1
			
	# The beginning 'S' is counted as 2 generated terminal symbols here - if the word contains an 'S', the minimum amount of terminal symbols in the fully derived word is 2
	return generated_characters + 2

def findNonterminals(word):
	'''Assesses whether the input string contains any non-terminal symbols.
			args:
				word: string, string of terminal and non-terminal symbols.
			returns:
				bool'''
	if "S" in word:
		return True
	elif "Z" in word:
		return True
	elif "T" in word:
		return True
	elif "V" in word:
		return True
	elif "B" in word:
		return True
	else:
		return False

def expand(word, limit):
	'''Expands the input word by applying probabilistic grammar rules. The generated word cannot be longer than the given limit.
			args: 
				word: string, string of terminal and non-terminal symbols.
				limit: int, maximum length the word is allowed to have.
			returns:
				word: string, string of terminal and non-terminal symbols.'''
	# Sampling probabilities as per Sennhauser & Berwick 2018, once per word
	prob_b = np.random.uniform(0.6, 0.9)
	prob_c = np.random.uniform(0.6, 0.9)
	for i in range(len(word)):
		# Once word length has exceeded the limit, the candidate is invalid and will not be processed further
		if len(word) > limit:
			return word
		else:
			
			s_l = min(1, -3 * countCharacters(word)/float(limit) + 3)
			P_branch = prob_b * s_l
			P_concat = prob_c * s_l
			random = np.random.uniform(0.0, 1.0)
			
			# Applying probabilistic Dyck2 grammar by replacing left-hand non-terminals with their right-hand production
			if word[i] == 'S':
				if P_concat <= 0.0:
					word = word[:i] + 'Z' + word[i+1:]
				elif random < P_concat:
					word = word[:i] + 'ZS' + word[i+1:]
				else:
					word = word[:i] + 'Z' + word[i+1:]
			elif word[i] == 'Z':
				if P_branch <= 0.0:
					word = word[:i] + 'T' + word[i+1:]
				elif random < P_branch:
					word = word[:i] + 'B' + word[i+1:]
				else:
					word = word[:i] + 'T' + word[i+1:]
			elif word[i] == 'B':
				if random < 0.5:
					word = word[:i] + '[S]' + word[i+1:]
				else:
					word = word[:i] + '{S}' + word[i+1:]
			elif word[i] == 'T':
				if random < 0.5:
					word = word[:i] + '[]' + word[i+1:]
				else:
					word = word[:i] + '{}' + word[i+1:]
			else:
				continue
				
	return word

def createCorpus(limit, goal_corpus_size):
	'''Generates a corpus of size goal_corpus_size of words with len <= limit. Saves the result in steps of 10000 generated words as a txt file. Saves a final corpus as a txt file.
			args:
				limit: int, maximum length for a word in the corpus.
				goal_corpus_size: int, number of words in the full corpus.
			returns:
				none'''
	counter = 0
	corpus = set()
	# Populates the corpus
	while len(corpus) < goal_corpus_size:
		prev_save_len = 0
		counter += 1
		word = 'S'
		# Expands the word until it either is fully derived or exceeds the desired length
		while findNonterminals(word) and len(word) <= limit:
			word = expand(word, limit)
		# If the word is fully derived and fulfills the length requirement, it is added to the corpus
		if not findNonterminals(word):
			if len(word) <= limit:
				old_length = len(corpus)
				corpus.add(word)
				
		# Print occasional update on the generation process to the console
		if counter % 50000 == 0:
			print("Corp Size: {}\tGen words: {}".format(len(corpus), counter))
		
		# Save interim results
		if len(corpus) % 10000 == 0 and len(corpus) != prev_save_len:
			print("Saving corpus, length {} ...".format(len(corpus)))
			filename = os.path.join('..', 'corpus', "cumlen{}_{}.txt".format(limit, len(corpus)))
			outfile = open(filename, 'w')
			for item in corpus:
				outfile.write(item)
				outfile.write('$')
			outfile.close()

	# Save full corpus
	filename = os.path.join('..', 'corpus', "cumlen{}_{}.txt".format(limit, len(corpus)))
	outfile = open(filename, 'w')
	for item in corpus:
		outfile.write(item)
		outfile.write('$')
	outfile.close()

createCorpus(LIMIT, GOAL_CORPUS_SIZE)