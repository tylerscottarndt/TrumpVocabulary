import pandas as pd
from WebScrape import WebScrapper

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
        print("Scrapping:", current_url)
        yield current_url
    url_files.close()


# Python File to setup and pickle our dataframe we plan on working with    
if __name__ == '__main__':
    RALLY_TRANSCRIPTS, UNION_TRANSCRIPTS = [], []
    for current_url in extract_urls_from('rally_urls.txt'):
        full_speech = WebScrapper(current_url).scrape_all('p', tokenize=False)
        RALLY_TRANSCRIPTS.append(full_speech)

    for current_url in extract_urls_from('union.txt'):
        full_speech = WebScrapper(current_url).scrape_all('p', tokenize=False)
        UNION_TRANSCRIPTS.append(full_speech)

    rally_df = addToDataFrame(RALLY_TRANSCRIPTS, label=1)
    union_df = addToDataFrame(UNION_TRANSCRIPTS, label=0)
    main_df = pd.concat([rally_df, union_df])

    # We can pickle this dataframe.
    main_df.head()
