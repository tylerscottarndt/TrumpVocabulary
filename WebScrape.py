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

    def scrape_trump_rally(self, tag, tokenize=None):

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

    # def scrape_obama_rally(self, tag, tokenize=None):
    #     full_speech = ""
    #
    #     for count, p in enumerate(self.soup.find_all(tag, attrs={'size': '3'})):
    #         # don't include first two irrelevant tags
    #         if count > 1:
    #             full_speech = full_speech + p.get_text()
    #
    #     full_speech = self._clean_transcript(full_speech)
    #     if tokenize:
    #         full_speech = self.tokenize(full_speech)
    #         return full_speech
    #
    #     return full_speech

    def scrape_obama_rally(self, tag, tokenize=None):
        # get speech body
        text_body = self.soup.find_all('div', attrs={'class': 'field-docs-content'})[0]
        # remove i tags with audience laughter
        [s.extract() for s in text_body.find_all('i') if s.get_text() == "Laughter"]
        full_speech = ""

        # extract all p tags from speech body
        for p in text_body.find_all(tag):
            text = p.get_text()
            # <p> tag beginning with <i> tags signify new speaker
            i_tags = p.find_all('i')
            # add all text not beginning with <i> tag (i.e. no new speaker)
            if len(i_tags) == 0:
                full_speech += text
            # add all text beginning with <i> tag of "the president" (i.e. Obama speaking)
            else:
                if i_tags[0].get_text() == "The President.":
                    # remove "The President" signifying text
                    full_speech += ' '.join(text.split()[2:])

        full_speech = self._clean_transcript(full_speech)
        if tokenize:
            full_speech = self.tokenize(full_speech)
            return full_speech

        return full_speech

    def scrape_all_union(self, tag, tokenize=None):
        speech_parts = []

        for p in self.soup.find_all(tag):
            speech_parts.append(p.get_text() + " ")
            speech_parts.append(p.get_text())
        # remove the irrelevant first 3 p tags and last 6 p tags
        speech_parts = speech_parts[4:len(speech_parts) - 13]
        full_speech = ''.join(speech_parts)

        full_speech = self._clean_transcript(full_speech)
        if tokenize:
            full_speech = self.tokenize(full_speech)
            return full_speech

        return full_speech

    def _clean_transcript(self, text):
        # remove bracketed text
        text = re.sub('\[.*?\]', '', text)
        # replace apostraphe with no space
        text = re.sub("[‘’']", '', text)
        # replace remaining punctuation with a space
        text = re.sub('[^\w\s]', ' ', text)
        # replace newline character with space
        text = text.replace('\n', ' ')
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
