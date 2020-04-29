import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))


def get_cleaned_speeches(speech_list):
    cleaned_speech_list = []
    for speech in speech_list:
        cleaned_speech = [word for word in speech.split() if word not in stopwords]
        cleaned_speech = ' '.join(cleaned_speech)
        cleaned_speech_list.append(cleaned_speech)
    return cleaned_speech_list


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_speeches_list_from_df(df, attr_val):
    df_speeches = df.loc[df['type'] == attr_val]
    speeches_list = df_speeches['transcript'].tolist()
    return speeches_list

def get_top_n_words(corpus, n):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


trump_data = pickle.load(open("DATA/TRUMP_SNIPPETS_DF.pickle", "rb"))
obama_data = pickle.load(open("DATA/OBAMA_SNIPPETS_DF.pickle", "rb"))
X = obama_data['transcript'].to_numpy()
y = obama_data['type'].to_numpy()

tf_idf = TfidfVectorizer(min_df = 1)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
sss.get_n_splits(X, y)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

X_train_tf = tf_idf.fit_transform(X_train)
X_test_tf = tf_idf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tf, y_train)

predictions = model.predict(X_test_tf)

cm = confusion_matrix(y_test, predictions)
labels = ['Union', 'Rally']
plot_confusion_matrix(cm, labels, title='Confusion Matrix')
print(classification_report(y_test, predictions))

trump_df = pd.read_pickle("DATA/trump_speeches_df.pickle")
obama_df = pd.read_pickle("DATA/obama_speeches_df.pickle")
trump_rallies = get_cleaned_speeches(get_speeches_list_from_df(trump_df, 1))
trump_unions = get_cleaned_speeches(get_speeches_list_from_df(trump_df, 0))
obama_rallies = get_cleaned_speeches(get_speeches_list_from_df(obama_df, 1))
obama_unions = get_cleaned_speeches(get_speeches_list_from_df(obama_df, 0))
