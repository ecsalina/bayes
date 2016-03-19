import unittest2
from MultinomialNaiveBayes import MultinomialNaiveBayes as mnb

my_bayes = mnb("test.db")

#If no db, uncomment and run once, then recomment (so nums are not duplicated).

my_bayes.train("Abracadabra", [
["super", "awesome", "mega"],
["a", "a", "a", "a"],
["a", "b", "c"]])

my_bayes.train("Battery", [
["sad", "sad", "negative"],
["my", "apple", "is", "red"],
["my", "my", "my", "what", "teeth", "you", "awesome"]])

my_bayes.train("Jesus", [
["sad", "sad", "negative"],
["my", "apple", "is", "red"],
["my", "my", "my", "what", "teeth", "you", "awesome"]])

class test(unittest2.TestCase):

	def testUniqueWords1(self):
		cur = my_bayes.conn.cursor()
		cur.execute("SELECT COUNT(*) FROM Battery")
		self.failUnless(cur.fetchone()[0], 10)

	def testUniqueWords2(self):
		cur = my_bayes.conn.cursor()
		cur.execute("SELECT COUNT(*) FROM Abracadabra")
		self.failUnless(cur.fetchone()[0], 6)

	def testWordCount1Doc(self):
		cur = my_bayes.conn.cursor()
		cur.execute("SELECT count FROM Abracadabra WHERE term = 'super' ")
		self.failUnless(cur.fetchone()[0], 1)

	def testWordCountManyDocs(self):
		cur = my_bayes.conn.cursor()
		cur.execute("SELECT count FROM Jesus WHERE term = 'my' ")
		self.failUnless(cur.fetchone()[0], 4)

	def testTotNumDocs(self):
		cur = my_bayes.conn.cursor()
		cur.execute("SELECT SUM(count) FROM doc_nums")
		self.failUnless(cur.fetchone()[0], 9)

	def testNumDocsClass(self):
		cur = my_bayes.conn.cursor()
		cur.execute("SELECT count FROM doc_nums WHERE class = 'Abracadabra'")
		self.failUnless(cur.fetchone()[0], 3)

	def test1Word(self):
		results = my_bayes.classify(["awesome"])
		print(results)
		self.failUnless(results["Abracadabra"] > results["Battery"])

	def test1WordRepeat(self):
		results = my_bayes.classify(["awesome", "awesome", "awesome", "awesome"])
		print(results)
		self.failUnless(results["Abracadabra"] > results["Battery"])

	def test1WordRepeat2(self):
		results = my_bayes.classify(["my"])
		print(results)
		self.failUnless(results["Battery"] > results["Abracadabra"])


	def testEqualClasses(self):
		results = my_bayes.classify(["my"])
		print(results)
		self.failUnlessAlmostEqual(results["Battery"], results["Jesus"])






if __name__ == "__main__":
	unittest2.main()
