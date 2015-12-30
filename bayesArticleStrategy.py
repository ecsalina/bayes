import math
import csv
import datetime
import codecs
import urllib
import urllib2
import json
import nltk
import re
import yahoo_finance as yahoo
import _utils as _utils
from MultinomialNaiveBayes import MultinomialNaiveBayes as mnb

"""
For a given stock ticker in a given time period:
All articles released on a given day (starting on the open of the market to the
next open on the following day) are associated with the returns over the next
period. To predict, all articles before the open are collected, parsed into a
list of words, and then run through naive Bayesian classifier to classify the
docs as positive, negative, or neutral returns. The pseudo-probabilities for
each class are summed for all docs. The class with the highest probability is
then the prediction of the return for the following day. We then long x shares
for the next day if there is a positive prediction, short, if there is a
negative prediction, and nothing if there is a neutral prediction.

NOTES:
	1. test with different bars?
	2. hypothesis test to determine whether significant difference between each
	class probability.
	3. find the bar with the highest returns (hour, day, week?) --> need more
	specific stock data.
	4. end the long/short on the peak/trough (once begins to revert)
	5. change discrete +-neutral to continuous prediction of actual stock
	return values.
	6. negative more effect than positive?
	7. less important articles have effet later from now. More important more
	immediate?
	8. google trends to weight the amount of the return (on same day works well,
	but what about prediction for next day? more specific than day granularity?)

To backtest:
	1. for each day, before open, make a prediction of the return with the docs
	from the past day.
	2. buy/sell/nothing based on prediction. Hold for one day.
	3. At next open, reverse action. Also, update db with the actuall occurance
	of the past day (positive/negative/neutral returns)
"""

##AUXILIARY FUNCTIONS:
def _dayIter(start, end, delta):
	"""
	Returns datetimes from [start, end) for use in iteration.
	"""
	current = start
	while current < end:
		yield current
		current += delta

def _getWords(body):
	#split body text into individual words
	tokens = nltk.word_tokenize(body)
	#Removes stop words and non alphabetic tokens.
	#TODO: does not seem to remove possessive "'s" and contractions "n't"
	tokens = [e for e in tokens if e not in nltk.corpus.stopwords.words("english")]
	tokens = [e for e in tokens if re.search("[A-Za-z]", e)]
	tokens = [e for e in tokens if "'" not in e]
	#lemmatize below
	#stemmer = nltk.stem.snowball.SnowballStemmer("english", ignore_stopwords=True)
	#tokens = [stemmer.stem(e) for e in tokens]
	return tokens




##CONSTANTS & SETUP:

tickers = ["MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", "DD",
			 "XOM", "GE", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK",
			 "MSFT", "NKE", "PFE", "PG", "TRV", "UTX", "UNH", "VZ", "V", "WMT"]
tickers = ["AAPL"]

start = datetime.datetime(2015, 12, 22, 9, 30, 0)
end = datetime.datetime(2015, 12, 29, 9, 30, 0)

MIN_POS = 0.01
MAX_NEG = -0.01

#convert to dict for easier article/stock assignment
tickerData = {}
for ticker in tickers:
	tickerData[ticker] = {}
	for day in _dayIter(start, end, datetime.timedelta(days=1)):
		tickerData[ticker][day] = {}
		tickerData[ticker][day]["arts"] = []
		tickerData[ticker][day]["prices"] = {}


##MAIN CODE:

#1. Collect article body data, and stock data. Save to csv.
for ticker in tickers:
	arts = _utils.collectArticles(ticker)
	stocks = yahoo.Share(ticker).get_historical(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

	#convert stock data from string to correct types:
	for stock in stocks:
		stock["Adj_Close"] = float(stock["Adj_Close"])
		stock["Close"] = float(stock["Close"])
		stock["Date"] = datetime.datetime.strptime(stock["Date"], "%Y-%m-%d").replace(hour=9, minute=30, second=0)
		stock["High"] = float(stock["High"])
		stock["Low"] = float(stock["Low"])
		stock["Open"] = float(stock["Open"])
		stock["Volume"] = int(stock["Volume"])


	#group by day and add to tickerData, where day: [open, open)
	#articles assigned to date of the BEGINNING of the collection period
	for day in _dayIter(start, end, datetime.timedelta(days=1)):
		nextDay = day + datetime.timedelta(days=1)
		for art in arts:
			if day <= art[0] < nextDay:
				date = art[0]
				title = art[1]
				link = art[2]
				text = _getWords(art[3])
				tickerData[ticker][day]["arts"].append([date, title, link, text])
		for stock in stocks:
			if day.date() == stock["Date"].date():
				tickerData[ticker][day]["prices"] = stock


def train(data, bayes, start, end):
	"""
	Setup the classifier with some beginning data.
	"""
	print("Training classifier:")
	for day in _dayIter(start, datetime.datetime(2015, 12, 24, 9, 30, 0), datetime.timedelta(days=1)):
		try:
			prevArts = data[day - datetime.timedelta(days=1)]["arts"]
			currPrice = data[day]["prices"]["Open"]
			nextPrice = data[day + datetime.timedelta(days=1)]["prices"]["Open"]
			returns = math.log(nextPrice / currPrice)
			if returns >= MIN_POS:
				clas = "POS"
			elif returns <= MAX_NEG:
				clas = "NEG"
			elif MAX_NEG < returns < MIN_POS:
				clas = "NEUT"
			print(day)
			print("    returns: "+str(returns))
			print("    class: "+clas)
			bayes.train(clas, zip(*prevArts)[3])
		except KeyError:
			print "key error somewhere"


def backtest(data, bayes, start, end):
	"""
	Backtests a given ticker on the period supplied by the data.

	:param data: dict of dates in period, which are also dicts containing
		a list of arts for that given day, and a list of stock prices on day
	:param start: datetime for start of backtest
	:param end: datetime for end of backtest
	"""
	print("Testing classifier: ")
	profit = 0.0
	for day in _dayIter(start, end, datetime.timedelta(days=1)):
		print(day)
		try:
			prevArts = data[day - datetime.timedelta(days=1)]["arts"]
			currPrice = data[day]["prices"]["Open"]
			nextPrice = data[day + datetime.timedelta(days=1)]["prices"]["Open"]

			#1. Make prediction based on past day's performance.
			predic = {"POS": 0.0, "NEG": 0.0, "NEUT": 0.0}
			for art in prevArts:
				results = bayes.classify(art[3])
				print("Bayes classify results: "+str(results))
				if results == None:
					continue
				predic["POS"] += results["POS"]
				predic["NEG"] += results["NEG"]
				predic["NEUT"] += results["NEUT"]

			#2. Buy/sell/nothing based on prediction.
			if predic["POS"] > predic ["NEG"]  and predic["POS"] > predic["NEUT"]:
				print("    BUY")
				profit += (-currPrice + nextPrice)
			elif predic["NEG"] > predic ["POS"]  and predic["NEG"] > predic["NEUT"]:
				print("    SELL")
				profit += (+currPrice - nextPrice)
			if predic["NEUT"] > predic ["POS"]  and predic["NEUT"] > predic["NEG"]:
				print("    NOTHING")
				pass
			print("profit: "+str(profit))

			#3. Update Bayesian classifier.
			returns = math.log(nextPrice / currPrice)
			if returns >= MIN_POS:
				clas = "POS"
			elif returns <= MAX_NEG:
				clas = "NEG"
			elif MAX_NEG < returns < MIN_POS:
				clas = "NEUT"
			bayes.train(clas, zip(*prevArts)[3])
		except KeyError:
			print("KeyError, probably because of beginning or ending of time range")



bayes = mnb("backtest.db", ["POS", "NEG", "NEUT"])
train(tickerData[ticker], bayes, start, datetime.datetime(2015, 12, 24, 9, 30, 0))
backtest(tickerData[ticker], bayes, datetime.datetime(2015, 12, 22, 9, 30, 0), end)
