import numpy as np
import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)


def find_avg_word_length(speech_list):
    word_count = 0
    total_characters = 0
    for speech in speech_list:
        word_count += len(speech.split())
        total_characters += len(speech) - speech.count(' ')

    avg_word_length = total_characters / word_count
    return avg_word_length


def get_speeches_list_from_df(df, attr_val):
    df_speeches = df.loc[df['type'] == attr_val]
    speeches_list = df_speeches['transcript'].tolist()

    return speeches_list

class load_file:
    def __init__(self, file):
        self.file = file

    def get_data(self):
        hal = np.load(self.file)
        X_train, y_train, X_test, y_test = [hal[f] for f in hal.files]
        return X_train, y_train, X_test, y_test


# df = load_file('dataset_split_transcripts.npz')
# X_train, y_train, X_test, y_test = df.get_data()
# Load split data
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# find average word length for full trump speeches
trump_df = pd.read_pickle("trump_speeches_df.pickle")
trump_rallies = get_speeches_list_from_df(1)
trump_unions = get_speeches_list_from_df(0)

trump_avg_rally_word_len = round(find_avg_word_length(trump_rallies), 2)
trump_avg_union_word_len = round(find_avg_word_length(trump_unions), 2)
print("Trump average rally word length: " + str(trump_avg_rally_word_len))
print("Trump average union word length: " + str(trump_avg_union_word_len)+'\n')

# find average word length for trump speeches w/o stop words
trump_df = pd.read_pickle("TRUMP_SNIPPETS_DF.pickle")
trump_rallies = get_speeches_list_from_df(trump_df, 1)
trump_unions = get_speeches_list_from_df(trump_df, 0)

trump_avg_rally_word_len = round(find_avg_word_length(trump_rallies), 2)
trump_avg_union_word_len = round(find_avg_word_length(trump_unions), 2)
print("Removing common stop words...\n")
print("Trump average rally word length: " + str(trump_avg_rally_word_len))
print("Trump average union word length: " + str(trump_avg_union_word_len)+'\n')

# find average word length for full obama speeches
obama_df = pd.read_pickle("obama_speeches_df.pickle")
obama_rallies = get_speeches_list_from_df(obama_df, 1)
obama_unions = get_speeches_list_from_df(obama_df, 0)

obama_avg_rally_word_len = round(find_avg_word_length(obama_rallies), 2)
obama_avg_union_word_len = round(find_avg_word_length(obama_unions), 2)
print("Obama average rally word length: " + str(obama_avg_rally_word_len))
print("Obama average union word length: " + str(obama_avg_union_word_len)+'\n')

# find average word length for obama speeches w/o stop words
obama_df = pd.read_pickle("OBAMA_SNIPPETS_DF.pickle")
obama_rallies = get_speeches_list_from_df(obama_df, 1)
obama_unions = get_speeches_list_from_df(obama_df, 0)

obama_avg_rally_word_len = round(find_avg_word_length(obama_rallies), 2)
obama_avg_union_word_len = round(find_avg_word_length(obama_unions), 2)
print("Removing common stop words...\n")
print("Obama average rally word length: " + str(obama_avg_rally_word_len))
print("Obama average union word length: " + str(obama_avg_union_word_len)+'\n')
