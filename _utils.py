import os
import math
import csv
import datetime
import pytz
import codecs
import urllib
import urllib2
import json
import nltk
import re



def collectArticles(ticker):
	"""
	Returns a list of articles for given ticker in format: [dt, title, link, text].
	"""
	artData = []

	#already on file (no need to recall diffbot):
	#DEBUG CHANGE FOLLOWING LINE
	if os.path.isfile("data/{}_article_text - Copy.csv".format(ticker)): #DEBUG
		f = open("data/{}_article_text - Copy.csv".format(ticker), "rb")
		reader = csv.reader(f)
		for line in reader:
			dt = datetime.datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc)
			title = line[1]
			link = line[2]
			text = line[3]
			artData.append([dt, title, link, text])
		return artData

	#otherwise, need to call diffbot
	else:
		f = open("../RSS Collector/sample data/{}.csv".format(ticker), "rb")
		reader = csv.reader(f)
		
		fout = open("data/{}_article_text.csv".format(ticker), "wb")
		writer = csv.writer(fout)

		#gets rid of header: [datetime, title, link, id]
		reader.next()
		#read article data, get body text
		for line in reader:
			dt = datetime.datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S")
			title = line[1]
			link = line[2]

			try:
				print(link[link.index('*')+1:])
				query = "http://api.diffbot.com/v3/article?token=15feb246f969151e48a7171bae648d92&url="+urllib.quote(link)
				response = urllib2.urlopen(query)
				data = json.loads(response.read())

				if "error" in data:
					print("Error: Diffbot was unable to scrape this webpage.")
					continue
					
				text = data["objects"][0]["text"]
				try: #preffered, but sometimes breaks
					text = codecs.encode(text, "translit/long/ascii")
				except:
					text = text.encode("ascii", "ignore")
				if not text: #if empty body text, skip
					continue
				else:
					newLine = [dt, title, link, text]
					artData.append(newLine)
					writeLine = [dt.strftime("%Y-%m-%d %H:%M:%S"), title, link, text]
					writer.writerow(writeLine)

			except:
				print("Error has occurred.")
		return artData
