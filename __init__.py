import pandas as pd
import pickle
from WebScrape import WebScraper
import nltk
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))
nltk.download('stopwords')


def addToDataFrame(content, label=None):
    rally_df = pd.DataFrame(content)
    rally_df.columns = ['transcript']
    rally_df['type'] = label
    print("Added to dataset!")
    return rally_df


def generate_speech_snippets(speech_list, snippet_length=100):
    print("Generating Snippets...")
    snippets = []
    for speech in speech_list:
        print("a speech...")
        speech = speech.split()
        for i in range(0, len(speech), snippet_length):
            current_snippet = speech[i:i+snippet_length]
            current_snippet = [word for word in current_snippet if word not in stopwords]
            current_snippet = ' '.join(current_snippet)
            snippets.append(current_snippet)

    return snippets


def extract_urls_from(txt_file):
    url_files = open(txt_file, 'r')

    for whole_file in url_files:
        current_url = whole_file.replace('\n', '')
        print("Scraping:", current_url)
        yield current_url
    url_files.close()


def pickle_item(item, file_name):
    pickle_out = open(file_name, "wb")
    pickle.dump(item, pickle_out)
    pickle_out.close()


# Python File to setup and pickle our dataframe we plan on working with    
if __name__ == '__main__':
    SNIPPET_LENGTH = 100

    OBAMA_RALLY_TRANSCRIPTS, OBAMA_UNION_TRANSCRIPTS = [], []
    TRUMP_RALLY_TRANSCRIPTS, TRUMP_UNION_TRANSCRIPTS = [], []
    for current_url in extract_urls_from('trump_rally_urls.txt'):
        full_speech = WebScraper(current_url).scrape_trump_rally('p', tokenize=False)
        TRUMP_RALLY_TRANSCRIPTS.append(full_speech)

    for current_url in extract_urls_from('trump_union_urls.txt'):
        full_speech = WebScraper(current_url).scrape_all_union('p', tokenize=False)
        TRUMP_UNION_TRANSCRIPTS.append(full_speech)

    for current_url in extract_urls_from('obama_rally_urls.txt'):
        full_speech = WebScraper(current_url).scrape_obama_rally('p', tokenize=False)
        OBAMA_RALLY_TRANSCRIPTS.append(full_speech)

    for current_url in extract_urls_from('obama_union_urls.txt'):
        full_speech = WebScraper(current_url).scrape_all_union('p', tokenize=False)
        OBAMA_UNION_TRANSCRIPTS.append(full_speech)

    # save individual president dataframes if we want to adjust snippet size later
    trump_rally_df = addToDataFrame(TRUMP_RALLY_TRANSCRIPTS, label=1)
    trump_union_df = addToDataFrame(TRUMP_UNION_TRANSCRIPTS, label=0)
    trump_speeches_df = pd.concat([trump_rally_df, trump_union_df], ignore_index=True)
    pickle_item(trump_speeches_df, "trump_speeches_df.pickle")

    obama_rally_df = addToDataFrame(OBAMA_RALLY_TRANSCRIPTS, label=1)
    obama_union_df = addToDataFrame(OBAMA_UNION_TRANSCRIPTS, label=0)
    obama_speeches_df = pd.concat([obama_rally_df, obama_union_df], ignore_index=True)
    pickle_item(obama_speeches_df, "obama_speeches_df.pickle")

    print("Generating speech snippets of size " + str(SNIPPET_LENGTH))
    trump_rally_snippets = generate_speech_snippets(TRUMP_RALLY_TRANSCRIPTS, snippet_length=SNIPPET_LENGTH)
    trump_union_snippets = generate_speech_snippets(TRUMP_UNION_TRANSCRIPTS, snippet_length=SNIPPET_LENGTH)
    obama_rally_snippets = generate_speech_snippets(OBAMA_RALLY_TRANSCRIPTS, snippet_length=SNIPPET_LENGTH)
    obama_union_snippets = generate_speech_snippets(OBAMA_UNION_TRANSCRIPTS, snippet_length=SNIPPET_LENGTH)
    print('Done.')

    print("Labeling Trump snippets...")
    trump_rally_snippets_df = addToDataFrame(trump_rally_snippets, label=1)
    trump_union_snippets_df = addToDataFrame(trump_union_snippets, label=0)
    trump_speech_snippets_df = pd.concat([trump_rally_snippets_df, trump_union_snippets_df], ignore_index=True)
    pickle_item(trump_speech_snippets_df, "TRUMP_SNIPPETS_DF.pickle")
    print("Saved!")


    print("Labeling Obama snippets...")
    obama_rally_snippets_df = addToDataFrame(obama_rally_snippets, label=1)
    obama_union_snippets_df = addToDataFrame(obama_union_snippets, label=0)
    obama_speech_snippets_df = pd.concat([obama_rally_snippets_df, obama_union_snippets_df], ignore_index=True)
    pickle_item(obama_speech_snippets_df, "OBAMA_SNIPPETS_DF.pickle")
