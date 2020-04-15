from bs4 import BeautifulSoup
import requests
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')


class WebScraper:
    def __init__(self, url):
        headers = requests.utils.default_headers()
        headers.update({'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
        req = requests.get(url, headers)
        self.soup = BeautifulSoup(req.content, 'html.parser')

    def scrape_all_rally(self, tag, tokenize=None):
        [s.extract() for s in self.soup('a')]
        full_speech = ""

        for p in self.soup.find_all(tag):
            speech_parts = p.get_text().split("\n")
            first_line = speech_parts[0]

            # only include speech from Donald Trump, exclude crowd and other speakers
            if first_line == "Donald Trump: ()" or first_line == "President Trump: ()":
                full_speech = full_speech + " " + speech_parts[1]

        full_speech = self._clean_transcript(full_speech)
        if tokenize:
            full_speech = self.tokenize(full_speech)
            return full_speech

        return full_speech

    def scrape_all_union(self, tag, tokenize=None):
        speech_parts = []

        for p in self.soup.find_all(tag):
            speech_parts.append(p.get_text())

        # remove the irrelevant first 3 p tags and last 6 p tags
        speech_parts = speech_parts[3:len(speech_parts)-6]
        full_speech = ''.join(speech_parts)

        full_speech = self._clean_transcript(full_speech)
        if tokenize:
            full_speech = self.tokenize(full_speech)
            return full_speech

        return full_speech

    def _clean_transcript(self, text):
        # remove bracketed text
        text = re.sub('\[.*?\]', '', text)
        # remove non-word and non-space characters (i.e. punctuation)
        text = re.sub('[^\w\s]', '', text)
        # remove numbers
        text = ''.join([i for i in text if not i.isdigit()])
        # change multiple spaces back to single space
        text = re.sub(' +', ' ', text)
        # lowercase
        text = text.lower()
        # strip leading and trailing spaces
        text = text.strip()

        return text

    def tokenize(self, content, amount_features=None, regex='\w+'):
        tokenized = word_tokenize(content)
        return tokenized