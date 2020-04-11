from DataFormatter import DataFormatter
import pickle
import pandas as pd
import sys

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

if __name__ == '__main__':
    rally_urls = []
    rally_url_file = open('rally_urls.txt')
    for line in rally_url_file:
        rally_urls.append(line.replace('\n', ''))
    rally_url_file.close()

    rally_location = ['Charlotte_NC', 'Charleston_SC', 'Las_Vegas_NV', 'Colorado_Springs_CO', 'Phoenix_AZ',
                  'Manchester_NH_01', 'Des_Moines, IA', 'Wildwood_NJ', 'Milwaukee_WS', 'Battle_Cree_MI',
                  'Hershey_PA', 'Lexington_KY', 'Tupelo_MS', 'Dallas_TX', 'Minneapolis_MN', 'Houston_TX',
                  'Albuquerque_NM', 'Fayetteville_NC', 'Manchester_NH_02', 'Cincinnati_OH', 'Greenville_NC']

    try:
        pickle_in = open('speech_df.pickle', 'rb')
        speech_df = pickle.load(pickle_in)
        pickle_in.close()
    except:
        speeches = []
        print("Loading and Cleaning URLs...")
        for url in rally_urls:
            transcript = DataFormatter.url_to_transcript(url)
            clean_transcript = DataFormatter.clean_transcript(transcript)
            speeches.append(clean_transcript)

        print("Creating Dictionary of {key: location, value:Speech}...")
        speech_dict = dict(zip(rally_location, speeches))
        speech_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in speech_dict.items()]))

        print("Saving DataFrame...")
        DataFormatter.pickle_object(speech_df, 'speech_df.pickle')

        print("Done.")
        sys.exit()

    print(speech_df.iloc[0:10, 0:10])
