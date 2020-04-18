import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def get_average_speech_length(speech_list):
    word_count = 0
    for speech in speech_list:
        word_count += len(speech.split())
    average_speech_length = word_count / len(speech_list)
    return average_speech_length


def get_speeches_list_from_df(df, attr_val):
    df_speeches = df.loc[df['type'] == attr_val]
    speeches_list = df_speeches['transcript'].tolist()
    return speeches_list


def get_word_length_freq(speech_list):
    new_dict = dict()
    word_count = 0
    for speech in speech_list:
        for word in speech.split():
            length = len(word)
            if length >= 10:
                new_dict[10] = new_dict.get(10, 0) + 1
            else:
                new_dict[length] = new_dict.get(length, 0) + 1
            word_count += 1
    for key in new_dict.keys():
        new_dict[key] = round(new_dict[key]/word_count*100, 2)
    value_list = list(dict(sorted(new_dict.items(), key=lambda x: x[0])).values())
    return value_list


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

# get full trump speeches
trump_df = pd.read_pickle("trump_speeches_df.pickle")
trump_rallies = get_speeches_list_from_df(trump_df, 1)
trump_unions = get_speeches_list_from_df(trump_df, 0)
# get clean trump speeches, i.e. no stop words
trump_clean_df = pd.read_pickle("TRUMP_SNIPPETS_DF.pickle")
trump_clean_rallies = get_speeches_list_from_df(trump_clean_df, 1)
trump_clean_unions = get_speeches_list_from_df(trump_clean_df, 0)

# get full obama speeches
obama_df = pd.read_pickle("obama_speeches_df.pickle")
obama_rallies = get_speeches_list_from_df(obama_df, 1)
obama_unions = get_speeches_list_from_df(obama_df, 0)
# get clean obama speeches, i.e. no stop words
obama_clean_df = pd.read_pickle("OBAMA_SNIPPETS_DF.pickle")
obama_clean_rallies = get_speeches_list_from_df(obama_clean_df, 1)
obama_clean_unions = get_speeches_list_from_df(obama_clean_df, 0)

print("DONALD TRUMP SPEECH STATISTICS: ")
print("================================================")
# find average length of trump speeches
trump_avg_rally_len = round(get_average_speech_length(trump_rallies))
trump_avg_union_len = round(get_average_speech_length(trump_unions))
print("Trump average length of rally speech: " + "{:,}".format(trump_avg_rally_len) + ' words')
print("Trump average length of SOU speech: " + "{:,}".format(trump_avg_union_len) + ' words\n')

# find average word length of trump speeches
trump_avg_rally_word_len = round(find_avg_word_length(trump_rallies), 2)
trump_avg_union_word_len = round(find_avg_word_length(trump_unions), 2)
print("Trump average rally word length: " + str(trump_avg_rally_word_len))
print("Trump average SOU word length: " + str(trump_avg_union_word_len)+'\n')

# find average word length of clean trump speeches
trump_avg_rally_word_len = round(find_avg_word_length(trump_clean_rallies), 2)
trump_avg_union_word_len = round(find_avg_word_length(trump_clean_unions), 2)
print("Removing common stop words...\n")
print("Trump average rally word length: " + str(trump_avg_rally_word_len))
print("Trump average SOU word length: " + str(trump_avg_union_word_len)+'\n')

print("BARACK OBAMA SPEECH STATISTICS: ")
print("================================================")
# find average length of obama speeches
obama_avg_rally_len = round(get_average_speech_length(obama_rallies))
obama_avg_union_len = round(get_average_speech_length(obama_unions))
print("Obama average length of rally speech: " + "{:,}".format(obama_avg_rally_len) + ' words')
print("Obama average length of SOU speech: " + "{:,}".format(obama_avg_union_len) + ' words\n')

# find average word length of obama speeches
obama_avg_rally_word_len = round(find_avg_word_length(obama_rallies), 2)
obama_avg_union_word_len = round(find_avg_word_length(obama_unions), 2)
print("Obama average rally word length: " + str(obama_avg_rally_word_len))
print("Obama average SOU word length: " + str(obama_avg_union_word_len)+'\n')

# find average word length of clean obama speeches
obama_avg_rally_word_len = round(find_avg_word_length(obama_clean_rallies), 2)
obama_avg_union_word_len = round(find_avg_word_length(obama_clean_unions), 2)
print("Removing common stop words...\n")
print("Obama average rally word length: " + str(obama_avg_rally_word_len))
print("Obama average SOU word length: " + str(obama_avg_union_word_len)+'\n')

# data to plot
n_groups = 10
# find discretized word lengths in trump speeches
trump_rally_word_freq = get_word_length_freq(trump_rallies)
trump_union_word_freq = get_word_length_freq(trump_unions)
# find discretized word lengths in obama speeches
obama_rally_word_freq = get_word_length_freq(obama_rallies)
obama_union_word_freq = get_word_length_freq(obama_unions)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, trump_rally_word_freq, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Trump_Rally')

rects2 = plt.bar(index + bar_width, trump_union_word_freq, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Trump_SOU')

plt.xlabel('Word Lengths')
plt.ylabel('Percentage of Speech')
plt.title('Frequency of Words by Length')
plt.xticks(index + bar_width, ('1', '2',  '3', '4', '5', '6', '7', '8', '9', '10+'))
plt.legend()

plt.tight_layout()
plt.show()
