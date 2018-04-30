# A multithreading script which scrapes mashable
import pandas as pd
from bs4 import BeautifulSoup
import requests
import urllib.request as ur
import csv
from multiprocessing.dummy import Pool  # This is a thread-based Pool
from multiprocessing import cpu_count

def crawlToCSV(URLrecord):
    placeHolder = []
    print(URLrecord)
    placeHolder.append(URLrecord)
    try:
        page = requests.get(URLrecord)
        data = page.text
        soup = BeautifulSoup(data, "html5lib")
        if soup:
            title = soup.find('h1', 'title')
            # content = soup.find('section','article-content').get_text()
            if title:
                title = title.get_text()
                print(title)
                # print(content.strip())
                placeHolder.append(title)
            content = soup.find('section', 'article-content')
            if content:
                print(content.text)
                placeHolder.append(content.text)
            with open("news.csv", "a") as f:
                writeFile = csv.writer(f)
                writeFile.writerow(placeHolder)
                f.close()
    except ValueError:
        print("Couldn't get anything")


i = 0
fileName = "data/data.csv"
df = pd.read_csv("data/data.csv")
urls = list(df['url'])
pool = Pool(cpu_count() * 2)  # Creates a Pool with cpu_count * 2 threads.
pool.map(crawlToCSV, urls)  # results is a list of all the placeHolder lists returned from each call to crawlToCSV






