import pandas as pd
import pickle
from WebScrape import WebScraper
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


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
                filt_sen = [word for word in chunks if not word in stopwords.words()]
                filt_sen = " ".join(filt_sen)
                main_data.append(filt_sen)
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
    CHUNKS_LENGTH = 100  # Choose # if chunks

    OBAMA_RALLY_TRANSCRIPTS, OBAMA_UNION_TRANSCRIPTS = [], []
    TRUMP_RALLY_TRANSCRIPTS, TRUMP_UNION_TRANSCRIPTS = [], []
    for current_url in extract_urls_from('trump_rally_urls.txt'):
        full_speech = WebScraper(current_url).scrape_all_rally('p', tokenize=False)
        TRUMP_RALLY_TRANSCRIPTS.append(full_speech)

    for current_url in extract_urls_from('trump_union_urls.txt'):
        full_speech = WebScraper(current_url).scrape_all_union('p', tokenize=False)
        TRUMP_UNION_TRANSCRIPTS.append(full_speech)

    for current_url in extract_urls_from('obama_rally_urls.txt'):
        full_speech = WebScraper(current_url).scrape_obama_rally('font', tokenize=False)
        OBAMA_RALLY_TRANSCRIPTS.append(full_speech)

    for current_url in extract_urls_from('obama_union_urls.txt'):
        full_speech = WebScraper(current_url).scrape_all_union('p', tokenize=False)
        OBAMA_UNION_TRANSCRIPTS.append(full_speech)

    rally_data, union_data = [], []
    print("Gathering  Chunks...")
    for rally_trans in [TRUMP_RALLY_TRANSCRIPTS, OBAMA_RALLY_TRANSCRIPTS]:
        rally_data = gather_chunks(rally_data, rally_trans, chunk_length=CHUNKS_LENGTH)

    for union_trans in [TRUMP_UNION_TRANSCRIPTS, OBAMA_UNION_TRANSCRIPTS]:
        union_data = gather_chunks(union_data, union_trans, chunk_length=CHUNKS_LENGTH)
    print("Labeling...")
    rally_df = addToDataFrame(rally_data, label=1)
    union_df = addToDataFrame(union_data, label=0)
    main_df = pd.concat([rally_df, union_df], ignore_index=True)
    print("Saved!")
    # We can pickle this dataframe.
    pickle_out = open("trump_main_df.pickle", "wb")
    pickle.dump(main_df, pickle_out)
    pickle_out.close()
