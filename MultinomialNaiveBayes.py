import os
import sqlite3
from MySQLdb import escape_string
from collections import Counter
from math import log


class MultinomialNaiveBayes():
	"""
	Multinomimal Naive Bayesian Classifier for document classification.
	"""

	#CONSTRUCTORS:

	def __init__(self, db_name, classes=[]):
		self.conn = sqlite3.connect(db_name)
		self.isSetup = False
		if len(classes) > 0:
			for clas in classes:
				self._setup_db(clas)

	#AUX METHODS

	def _setup_db(self, clas):
		#Checks table exists. Creates it if not exist.
		self.conn.execute("CREATE TABLE IF NOT EXISTS {} (term text, count number);".format(clas))
		#Table for holding the num of docs of each class.
		self.conn.execute("CREATE TABLE IF NOT EXISTS doc_nums (class text, count number);")
		#Add class to doc_nums if not already in it.
		self.conn.execute("INSERT INTO doc_nums(class, count) SELECT ?, 0 WHERE NOT EXISTS(SELECT 1 FROM doc_nums WHERE class = ?);", (clas, clas))
		self.conn.commit()	
		self.isSetup = True

	#METHODS

	def train(self, clas, docs):
		"""
		Trains the Bayesian Classifier with the given input data.

		:param clas: string representing the class
		:param docs: list of docs, each of which is a list of words (w/ repeats)
		"""
		cur = self.conn.cursor()
		clas = escape_string(clas)

		self._setup_db(clas)

		#Adds documents to database.
		for doc in docs:
			counts = Counter(doc)
			for term, count in counts.iteritems():
				cur.execute("SELECT count from {} WHERE term = ?;".format(clas), (escape_string(term),))
				currCount = cur.fetchone()
				if currCount != None:
					count = currCount[0] + count
					self.conn.execute("UPDATE {} SET count = ? WHERE term = ?;".format(clas), (count, escape_string(term)))
				else:
					self.conn.execute("INSERT INTO {} VALUES(?, ?);".format(clas), (escape_string(term), count))

			#Update doc nums.
			cur.execute("SELECT count FROM doc_nums WHERE class = ?", (clas,))
			num = cur.fetchone()[0]
			self.conn.execute("UPDATE doc_nums SET count = ? WHERE class = ?;", (num+1, clas))
			self.conn.commit()


	def classify(self, doc):
		"""
		Returns dict of proportional probability of a doc being of each class.

		:param doc: list of words

		Based on Bayes's Theorem:
			P(C|D) = [P(C) * P(D|C)] / P(D)
			P(C|D) : posterior : probability of getting class C from doc D
			P(C) : prior : probability of C regardless of doc
			P(D|C) : likelihood : probability of doc D arising given that it is of class C
			P(D) : evidence : probability of D regardless of class (total num. evidence)

		In application to NLP, we can categorize a given document into a
		class by the words it contains. If we assume that all words are
		independent of each other in a document (an assumption which is
		typically false, but in reality does not affect the test results
		very much), then we can rewrite Bayes's Theorem as:

			P(C|w0, w1, ... wn) = P(C) * [PI i=0->n of P(wi|C)] / [PI i=0->n of P(wi)]

		where w is a given word of the doc, n is the number of words in the
		doc PI refers to the product notation.

		This can result in underflow error, so we take the ln, converting
		the product into a sum. This is okay since ln monotonically
		increases. In addition, we typically remove the denominator, since
		for all class calculations, there will be the same denominator.

			P(C|w0, w1, ... wn) = ln(P(C)) + [SIGMA i=0->n of ln(P(wi|C))]

		where "=" should really be a proportion symbol. To solve the problem
		where a 0 may be contained in the database (which would cause an error)
		with ln(0)), we carry out add-one laplace smoothing:
			P(wi|C) = COUNT()
		"""
		if not self.isSetup:
			return {}

		cur = self.conn.cursor()
		report = {}

		#get classes from db
		classes = zip(*list(cur.execute("SELECT class FROM doc_nums")))[0]

		#total number of docs in db
		cur.execute("SELECT SUM(count) FROM doc_nums")
		total_docs = cur.fetchone()
		total_docs = float(total_docs[0]) if total_docs != None else 0.0
		#DEBUG print("total num docs: "+str(total_docs))
		if total_docs == 0.0:
			return None


		for clas in classes:
			#DEBUG print clas
			#P(C): num of C docs over total num docs.
			cur.execute("SELECT count from doc_nums WHERE class = ?;", (clas,))
			num_docs = cur.fetchone()
			num_docs = float(num_docs[0]) if num_docs != None else 0.0
			#DEBUG print("num docs in class: "+str(num_docs))
			#class not yet fitted, then return 0 prob
			if num_docs == 0.0:
				return None

			PC = num_docs / total_docs if total_docs != 0.0 else 0.0
			#DEBUG print("PC: "+str(PC))
			probs = []
			probs.append(PC)

			#P(wi|C): num of wi occurances in C docs over total words C docs.
			for word in doc:
				#DEBUG print("term: "+word)
				#num occurances word in class
				cur.execute("SELECT count FROM {} WHERE term = ? ".format(clas), (escape_string(word),))
				word_freq = cur.fetchone()
				word_freq = float(word_freq[0]) if word_freq != None else 0.0
				#DEBUG print("word frequency: "+str(word_freq))

				#total number of words all docs in class
				cur.execute("SELECT SUM(count) FROM {}".format(clas))
				total_words = cur.fetchone()
				total_words = float(total_words[0]) if total_words[0] != None else 0.0
				#DEBUG print("total words in class: "+str(total_words))

				#num unique words in class
				cur.execute("SELECT COUNT(DISTINCT term) FROM {}".format(clas))
				vocab_size = cur.fetchone()
				vocab_size = float(vocab_size[0]) if vocab_size != None else 0.0
				#DEBUG print("vocab size: "+str(vocab_size))

				PwiC = (word_freq+1) / (total_words+vocab_size)
				#DEBUG print("PwiC: "+str(PwiC))
				probs.append(PwiC)

			#Take log of each term and sum:
			logs = [log(x) for x in probs]
			score = sum(logs)
			report[clas] = score

		return report
