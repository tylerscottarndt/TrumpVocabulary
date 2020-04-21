import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

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


def get_avg_vocabulary(speech_list):
    total_vocabulary = 0
    for speech in speech_list:
        curr_vocabulary = set()
        curr_vocabulary.update(speech.split())
        total_vocabulary += len(curr_vocabulary)
    avg_vocabulary = total_vocabulary / len(speech_list)
    return avg_vocabulary


def get_cleaned_speeches(speech_list):
    cleaned_speech_list = []
    for speech in speech_list:
        cleaned_speech = [word for word in speech.split() if word not in stopwords]
        cleaned_speech = ' '.join(cleaned_speech)
        cleaned_speech_list.append(cleaned_speech)
    return cleaned_speech_list


def generate_wordcloud(text):
    # Create and generate a word cloud image:
    wordcloud = WordCloud(background_color="white").generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def print_statistics(speech_list):
    avg_rally_len = round(get_average_speech_length(speech_list))
    avg_rally_vocab = round(get_avg_vocabulary(speech_list))
    avg_rally_word_len = round(find_avg_word_length(speech_list), 2)
    print("Average speech length: " + "{:,}".format(avg_rally_len) + ' words')
    print("Average speech vocabulary: " + "{:,}".format(avg_rally_vocab) + ' words')
    print("Average speech word length: " + str(avg_rally_word_len) + '\n')



class load_file:
    def __init__(self, file):
        self.file = file

    def get_data(self):
        hal = np.load(self.file)
        X_train, y_train, X_test, y_test = [hal[f] for f in hal.files]
        return X_train, y_train, X_test, y_test


# get full trump speeches
trump_df = pd.read_pickle("trump_speeches_df.pickle")
trump_rallies = get_speeches_list_from_df(trump_df, 1)
trump_unions = get_speeches_list_from_df(trump_df, 0)
# get clean trump speeches, i.e. no stop words
trump_clean_rallies = get_cleaned_speeches(trump_rallies)
trump_clean_unions = get_cleaned_speeches(trump_unions)

# get full obama speeches
obama_df = pd.read_pickle("obama_speeches_df.pickle")
obama_rallies = get_speeches_list_from_df(obama_df, 1)
obama_unions = get_speeches_list_from_df(obama_df, 0)
# get clean obama speeches, i.e. no stop words
obama_clean_rallies = get_cleaned_speeches(obama_rallies)
obama_clean_unions = get_cleaned_speeches(obama_unions)

print("DONALD TRUMP SPEECH STATISTICS: ")
print("================================================")
print("Pure Rallies:")
print_statistics(trump_rallies)

print("Cleaned Rallies:")
print_statistics(trump_clean_rallies)

print("Pure SOU:")
print_statistics(trump_unions)

print("Cleaned SOU:")
print_statistics(trump_clean_unions)

print("BARACK OBAMA SPEECH STATISTICS: ")
print("================================================")
print("Pure Rallies:")
print_statistics(obama_rallies)

print("Cleaned Rallies:")
print_statistics(obama_clean_rallies)

print("Pure SOU:")
print_statistics(obama_unions)

print("Cleaned SOU:")
print_statistics(obama_clean_unions)

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
bar_width = 0.15
opacity = 0.8

rects1 = plt.bar(index, trump_rally_word_freq, bar_width,
                 alpha=opacity,
                 color='#8c3f54',
                 label='Trump rally')

rects2 = plt.bar(index + bar_width, trump_union_word_freq, bar_width,
                 alpha=opacity,
                 color='#70878e',
                 label='Trump SOU')

rects3 = plt.bar(index + bar_width*2, obama_rally_word_freq, bar_width,
                 alpha=opacity,
                 color='#8fd1d9',
                 label='Obama rally')

rects4 = plt.bar(index + bar_width*3, obama_union_word_freq, bar_width,
                 alpha=opacity,
                 color='#8c8270',
                 label='Obama SOU')

plt.xlabel('Word Lengths')
plt.ylabel('Percentage of Speech')
plt.title('Frequency of Words by Length')
plt.xticks(index + bar_width, ('1', '2',  '3', '4', '5', '6', '7', '8', '9', '10+'))
plt.legend()

plt.tight_layout()
plt.show()

# generate wordclouds
generate_wordcloud(' '.join(trump_clean_rallies))
generate_wordcloud(' '.join(trump_clean_unions))
generate_wordcloud(' '.join(obama_clean_rallies))
generate_wordcloud(' '.join(obama_clean_unions))
