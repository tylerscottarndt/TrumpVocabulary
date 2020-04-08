from DataFormatter import DataFormatter
import pickle
import pandas as pd
import sys

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

if __name__ == '__main__':
    urls = [
            "https://www.rev.com/blog/transcripts/donald-trump-charlotte-north-carolina-rally-transcript-trump-holds-rally-before-super-tuesday",
            "https://www.rev.com/blog/transcripts/donald-trump-charleston-south-carolina-rally-transcript-february-28-2020",
            "https://www.rev.com/blog/transcripts/donald-trump-las-vegas-nevada-rally-transcript",
            "https://www.rev.com/blog/transcripts/donald-trump-colorado-springs-co-rally-transcript",
            "https://www.rev.com/blog/transcripts/donald-trump-phoenix-arizona-rally-transcript",
            "https://www.rev.com/blog/transcripts/donald-trump-new-hampshire-rally-february-10-2020",
            "https://www.rev.com/blog/transcripts/donal-trump-iowa-rally-transcript-trump-holds-rally-in-des-moines-iowa",
            "https://www.rev.com/blog/transcripts/donald-trump-new-jersey-rally-speech-transcript-trump-holds-rally-in-wildwood-nj",
            "https://www.rev.com/blog/transcripts/donald-trump-milwaukee-rally-transcript-trump-holds-rally-during-iowa-democratic-debate",
            "https://www.rev.com/blog/transcripts/donald-trump-michigan-rally-transcript-trump-holds-a-rally-in-battle-creek-during-impeachment",
            "https://www.rev.com/blog/transcripts/donald-trump-hershey-pennsylvania-rally-transcript-december-10-2019",
            "https://www.rev.com/blog/transcripts/donald-trump-kentucky-rally-speech-transcript-lexington-kentucky-rally",
            "https://www.rev.com/blog/transcripts/donald-trump-mississippi-rally-speech-transcript-2019-rally-in-tupelo-mississippi",
            "https://www.rev.com/blog/transcripts/donald-trump-dallas-rally-speech-transcript-october-17-2019",
            "https://www.rev.com/blog/transcripts/donald-trump-minnesota-rally-speech-transcript-minneapolis-mn-rally-october-10-2019",
            "https://www.rev.com/blog/transcripts/donald-trump-narendra-modi-houston-tx-rally-transcript-trump-and-modi-hold-texas-rally",
            "https://www.rev.com/blog/transcripts/donald-trump-new-mexico-rally-transcript-full-speech-transcript",
            "https://www.rev.com/blog/transcripts/donald-trump-north-carolina-rally-transcript-in-fayetteville-nc-september-9-2019",
            "https://www.rev.com/blog/transcripts/donald-trump-new-hampshire-rally-transcript-august-15-2019",
            "https://www.rev.com/blog/transcripts/donald-trump-ohio-rally-speech-transcript-full-transcript-of-august-1-2019-rally-in-cincinnati",
            "https://www.rev.com/blog/transcripts/donald-trump-maga-event-speech-transcript-north-carolina-rally"
            ]

    speech_loc = ['Charlotte_NC', 'Charleston_SC', 'Las_Vegas_NV', 'Colorado_Springs_CO', 'Phoenix_AZ',
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
        for url in urls:
            transcript = DataFormatter.url_to_transcript(url)
            clean_transcript = DataFormatter.clean_transcript(transcript)
            speeches.append(clean_transcript)

        print("Creating Dictionary of {key: location, value:Speech}...")
        speech_dict = dict(zip(speech_loc, speeches))
        speech_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in speech_dict.items()]))

        print("Saving DataFrame...")
        DataFormatter.pickle_object(speech_df, 'speech_df.pickle')

        print("Done.")
        sys.exit()

    print(speech_df.iloc[0:10, 0:10])
