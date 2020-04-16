import pandas as pd
import pickle
from WebScrape import WebScraper
from time import sleep


def addToDataFrame(content, label=None):
    rally_df = pd.DataFrame(content)
    rally_df.columns = ['transcript']
    rally_df['type'] = label
    print("Added to dataset!")
    return rally_df


def gather_chunks(main_data, transcript, chunk_length=100):
    chunks = []
    for each_transcript in transcript:
        current_tran = each_transcript
        current_tran = current_tran.split()
        for i in range(len(current_tran) - 1):
            word = current_tran[i]
            chunks.append(word)
            if len(chunks) == chunk_length:
                print(len(chunks))
                chunks = ' '.join(chunks)
                print(len(chunks))
                main_data.append(chunks)
                chunks = []

    return main_data


def extract_urls_from(txt_file):
    url_files = open(txt_file, 'r')

    for whole_file in url_files:
        current_url = whole_file.replace('\n', '')
        print("Scraping:", current_url)
        yield current_url
    url_files.close()


# Python File to setup and pickle our dataframe we plan on working with    
if __name__ == '__main__':
    RALLY_TRANSCRIPTS, UNION_TRANSCRIPTS = [], []
    total_rally_words = 0
    for current_url in extract_urls_from('rally_urls.txt'):
        full_speech = WebScraper(current_url).scrape_all_rally('p', tokenize=False)
        total_rally_words += len(full_speech)
        RALLY_TRANSCRIPTS.append(full_speech)

    total_union_words = 0
    for current_url in extract_urls_from('union.txt'):
        full_speech = WebScraper(current_url).scrape_all_union('p', tokenize=False)
        total_union_words += len(full_speech)
        UNION_TRANSCRIPTS.append(full_speech)

    rally_data, union_data = [], []
    rally_data = gather_chunks(rally_data, RALLY_TRANSCRIPTS, chunk_length=100)
    union_data = gather_chunks(union_data, UNION_TRANSCRIPTS, chunk_length=100)

    print('rally', len(rally_data))
    print('union', union_data[0],'\n', union_data[1])

    rally_df = addToDataFrame(rally_data, label=1)
    union_df = addToDataFrame(union_data, label=0)
    main_df = pd.concat([rally_df, union_df])

    print(main_df)

    # We can pickle this dataframe.
    pickle_out = open("main_df.pickle", "wb")
    pickle.dump(main_df, pickle_out)
    pickle_out.close()
