import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import re

list_of_speeches = []



res = requests.get('http://www.historyplace.com/speeches/previous.htm')

soup = bs(res.content, 'lxml')
speeches = soup.find('table')


speech_names = []
new_speech = []
speech_href = []
links_for_sp = []
list_of_links = []


for link in speeches.find_all('li'):
    speech_names.append(link.text)
    speech_href.append(link('a'))

for item in speech_href:
    new_item = str(item)
    new_speech.append(new_item)


for row in new_speech:
    link = re.findall(r'"(.*?)"', row)
    links_for_sp.append(link[0])



for link in links_for_sp:
    list_of_links.append('http://www.historyplace.com/speeches/' + link)


for slug in list_of_links:
    res = requests.get(slug)
    soup2 = bs(res.content, 'lxml')
    text = soup2.find_all('strong')
    list_of_speeches.append(text)
