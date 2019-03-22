import pandas as pd
from keras.preprocessing.text import text_to_word_sequence

def formatting(path):
    data = pd.read_csv(path)
    data["summary"] = data["summary"].astype(str)
    data['reviewText'] = data['reviewText'].astype(str)
    data['summary'] = data['summary'].apply(text_to_word_sequence)
    data['reviewText'] = data['reviewText'].apply(text_to_word_sequence)

    return data