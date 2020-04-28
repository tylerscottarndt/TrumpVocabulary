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


def get_cleaned_speeches(speech_list):
    cleaned_speech_list = []
    for speech in speech_list:
        cleaned_speech = [word for word in speech.split() if word not in stopwords]
        cleaned_speech = ' '.join(cleaned_speech)
        cleaned_speech_list.append(cleaned_speech)
    return cleaned_speech_list


def get_avg_word_length(speech_list):
    word_count = 0
    total_characters = 0
    for speech in speech_list:
        word_count += len(speech.split())
        total_characters += len(speech) - speech.count(' ')

    avg_word_length = total_characters / word_count
    return avg_word_length


def get_avg_word_counts(speech_list):
    word_count = 0
    for speech in speech_list:
        word_count += len(speech.split())
    average_speech_length = word_count / len(speech_list)
    return average_speech_length


def get_speeches_list_from_df(df, attr_val):
    df_speeches = df.loc[df['type'] == attr_val]
    speeches_list = df_speeches['transcript'].tolist()
    return speeches_list


def _get_word_length_freq(speech_list):
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


def generate_wordcloud(text):
    # Create and generate a word cloud image:
    wordcloud = WordCloud(background_color="white").generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def generate_word_freq_graph(trump_speeches, obama_speeches):
    # data to plot
    n_groups = 10
    # find discretized word lengths in trump speeches
    trump_rally_word_freq = _get_word_length_freq(trump_speeches[0])
    trump_union_word_freq = _get_word_length_freq(trump_speeches[1])
    # find discretized word lengths in obama speeches
    obama_rally_word_freq = _get_word_length_freq(obama_speeches[0])
    obama_union_word_freq = _get_word_length_freq(obama_speeches[1])

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

    rects3 = plt.bar(index + bar_width * 2, obama_rally_word_freq, bar_width,
                     alpha=opacity,
                     color='#8fd1d9',
                     label='Obama rally')

    rects4 = plt.bar(index + bar_width * 3, obama_union_word_freq, bar_width,
                     alpha=opacity,
                     color='#8c8270',
                     label='Obama SOU')

    plt.xlabel('Word Lengths')
    plt.ylabel('Percentage of Speech')
    plt.title('Frequency of Words by Length')
    plt.xticks(index + bar_width, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10+'))
    plt.legend()

    plt.tight_layout()
    plt.show()


def generate_comparison_graph(trump_speeches, obama_speeches, title_name, y_label):
    # data to plot
    n_groups = 2

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = ax.bar(index, trump_speeches, bar_width,
                     alpha=opacity,
                     color='#8c3f54',
                     label='Trump')

    rects2 = ax.bar(index + bar_width, obama_speeches, bar_width,
                     alpha=opacity,
                     color='#70878e',
                     label='Obama')

    plt.xlabel('Speech Types')
    plt.ylabel(y_label)
    plt.title(title_name)
    plt.xticks(index + bar_width, ('Rallies', 'SOU'))
    plt.legend()

    _autolabel(rects1, ax)
    _autolabel(rects2, ax)

    fig.tight_layout()
    plt.show()


def _autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:,}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


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

# get full obama speeches
obama_df = pd.read_pickle("obama_speeches_df.pickle")
obama_rallies = get_speeches_list_from_df(obama_df, 1)
obama_unions = get_speeches_list_from_df(obama_df, 0)

# generate average word count visualizations
trump_word_counts = [get_avg_word_counts(trump_rallies), get_avg_word_counts(trump_unions)]
obama_word_counts = [get_avg_word_counts(obama_rallies), get_avg_word_counts(obama_unions)]
generate_comparison_graph(trump_word_counts, obama_word_counts, "Average Word Count", "Number of Words")

# generate average vocabulary visualizations
trump_vocabs = [get_avg_vocabulary(trump_rallies), get_avg_vocabulary(trump_unions)]
obama_vocabs = [get_avg_vocabulary(obama_rallies), get_avg_vocabulary(obama_unions)]
generate_comparison_graph(trump_vocabs, obama_vocabs, "Average Vocabulary", "Number of Words")

# generate average word length visualizations
trump_word_lengths = [get_avg_word_length(trump_rallies), get_avg_word_length(trump_unions)]
obama_word_lengths = [get_avg_word_length(obama_rallies), get_avg_word_length(obama_unions)]
generate_comparison_graph(trump_word_lengths, obama_word_lengths, "Average Word Length", "Number of Characters")

# generate word frequencies visualizations
trump_speeches = [trump_rallies, trump_unions]
obama_speeches = [obama_rallies, obama_unions]
generate_word_freq_graph(trump_speeches, obama_speeches)

# generate wordclouds
generate_wordcloud(' '.join(get_cleaned_speeches(trump_rallies)))
generate_wordcloud(' '.join(get_cleaned_speeches(trump_unions)))
generate_wordcloud(' '.join(get_cleaned_speeches(obama_rallies)))
generate_wordcloud(' '.join(get_cleaned_speeches(obama_unions)))

