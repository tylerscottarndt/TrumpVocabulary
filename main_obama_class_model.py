import numpy as np

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.regularizers import l1, l2
from keras import optimizers
import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


import itertools
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


def build_train_view():

    maxlen = 100
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=maxlen)
    model.add(embedding_layer)
    model.add(Dense(32, activity_regularizer=l1(0.001), activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.adam(lr=.0001), loss='binary_crossentropy', metrics=['acc'])
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    print(model.summary())
    history = model.fit(X_train, y_train, batch_size=64, epochs=40, validation_data=(X_test, y_test),
                        callbacks=[es_callback])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    loss, accuracy = model.evaluate(X_train, y_train, verbose=1)
    print('Training Accuracy is {}'.format(accuracy * 100))

    loss, accuracy = model.evaluate(X_test, y_test)

    print('Testing Accuracy is {} '.format(accuracy * 100))

    yhat_classes = model.predict_classes(X_test)

    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)
    cm = confusion_matrix(y_test, yhat_classes)
    labels = ['Rally', 'Union']
    plot_confusion_matrix(cm, labels, title='Confusion Matrix')


class load_file:
    def __init__(self, file):
        self.file = file

    def get_data(self):
        hal = np.load(self.file)
        X_train, y_train, X_test, y_test,vocab_size, embedding_matrix = [hal[f] for f in hal.files]
        return X_train, y_train, X_test, y_test,vocab_size, embedding_matrix


df = load_file('ObamaSplitData.npz')
X_train, y_train, X_test, y_test,vocab_size, embedding_matrix = df.get_data()
build_train_view()