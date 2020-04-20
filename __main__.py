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

