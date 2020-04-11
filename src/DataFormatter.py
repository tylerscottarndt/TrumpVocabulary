from bs4 import BeautifulSoup
import requests
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
cv = CountVectorizer(stop_words = 'english')

class DataFormatter:
    @staticmethod
    def rally_url_to_transcript(url):
        headers = requests.utils.default_headers()
        headers.update({'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
        req = requests.get(url, headers)
        soup = BeautifulSoup(req.content, 'html.parser')

        # remove the <a> tags from the HTML
        [s.extract() for s in soup('a')]

        full_speech = ""

        # <p> tags contain the entire speech
        for p in soup.find_all("p"):
            # split <p> tag into [speaker, speech]
            speech_parts = p.get_text().split("\n")
            first_line = speech_parts[0]

            # only include speech from Donald Trump, exclude crowd and other speakers
            if first_line == "Donald Trump: ()" or first_line == "President Trump: ()":
                full_speech = full_speech + " " + speech_parts[1]

        return full_speech

    @staticmethod
    def clean_transcript(text):
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

    @staticmethod
    def pickle_object(object_to_pickle, file_name):
        pickle_out = open(file_name, "wb")
        pickle.dump(object_to_pickle, pickle_out)
        pickle_out.close()


