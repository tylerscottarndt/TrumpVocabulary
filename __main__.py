import numpy as np
import pandas as pd

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

obama_df = pd.read_pickle("OBAMA_SNIPPETS_DF.pickle")
print(obama_df.head())
print(obama_df.tail())

trump_df = pd.read_pickle("TRUMP_SNIPPETS_DF.pickle")
print(trump_df.head())
print(trump_df.tail())
