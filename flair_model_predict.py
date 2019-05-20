import pandas as pd

movie_df = pd.read_csv("phase1_movie_reviews-test-hidden.csv")
movie_labels = pd.read_csv("true_labels/true_movie_labels.txt", header=None)
movies_test = pd.concat([movie_df, movie_labels], axis=1).drop('polarity', axis=1).rename(columns={0: "polarity"})

game_df = pd.read_csv("phase1_video_games-test-hidden.csv")
game_labels = pd.read_csv("true_labels/true_game_labels.txt", header=None)
games_test = pd.concat([game_df, game_labels], axis=1).drop('polarity', axis=1).rename(columns={0: "polarity"})

movies_X_test = movies_test[['reviewText']]
movies_y_test = movies_test[['polarity']]

games_X_test = games_test[['reviewText']]
games_y_test = games_test[['polarity']]

from flair.models import TextClassifier
from flair.data import Sentence

#Using pretrained model:
classifier = TextClassifier.load('best-model.pt')

def predict(x):
    sentence = Sentence(x)
    classifier.predict(sentence)
    for label in sentence.labels:        
        return(label.value)

#Note this takes a long time!!
movies_X_test['polarity'] = movies_X_test['reviewText'].apply(predict)
games_X_test['polarity'] = games_X_test['reviewText'].apply(predict)