import pandas as pd
import pickle
from WebScrape import WebScraper

def addToDataFrame(content, label=None):
    rally_df = pd.DataFrame(content)
    rally_df.columns = ['transcript']
    rally_df['type'] = label
    print("Added to dataset!")
    return rally_df


def extract_urls_from(txt_file):
    url_files = open(txt_file, 'r')

    for whole_file in url_files:
        current_url = whole_file.replace('\n', '')
        print("Scraping:", current_url)
        yield current_url
    url_files.close()


# Python File to setup and pickle our dataframe we plan on working with    
if __name__ == '__main__':
    TRUMP_RALLY_TRANSCRIPTS, TRUMP_UNION_TRANSCRIPTS = [], []
    for current_url in extract_urls_from('trump_rally_urls.txt'):
        full_speech = WebScraper(current_url).scrape_trump_rally('p', tokenize=False)
        TRUMP_RALLY_TRANSCRIPTS.append(full_speech)

    for current_url in extract_urls_from('trump_union_urls.txt'):
        full_speech = WebScraper(current_url).scrape_all_union('p', tokenize=False)
        TRUMP_UNION_TRANSCRIPTS.append(full_speech)

    OBAMA_RALLY_TRANSCRIPTS, OBAMA_UNION_TRANSCRIPTS = [], []
    for current_url in extract_urls_from('obama_rally_urls.txt'):
        full_speech = WebScraper(current_url).scrape_obama_rally('font', tokenize=False)
        OBAMA_RALLY_TRANSCRIPTS.append(full_speech)

    for current_url in extract_urls_from('obama_union_urls.txt'):
        full_speech = WebScraper(current_url).scrape_all_union('p', tokenize=False)
        OBAMA_UNION_TRANSCRIPTS.append(full_speech)

    trump_rally_df = addToDataFrame(TRUMP_RALLY_TRANSCRIPTS, label=1)
    trump_union_df = addToDataFrame(TRUMP_UNION_TRANSCRIPTS, label=0)
    trump_main_df = pd.concat([trump_rally_df, trump_union_df])

    obama_rally_df = addToDataFrame(OBAMA_RALLY_TRANSCRIPTS, label=1)
    obama_union_df = addToDataFrame(OBAMA_UNION_TRANSCRIPTS, label=0)
    obama_main_df = pd.concat([obama_rally_df, obama_union_df])

    # We can pickle this dataframe.
    pickle_out = open("trump_main_df.pickle", "wb")
    pickle.dump(trump_main_df, pickle_out)
    pickle_out.close()

    pickle_out = open("obama_main_df.pickle", "wb")
    pickle.dump(obama_main_df, pickle_out)
    pickle_out.close()
