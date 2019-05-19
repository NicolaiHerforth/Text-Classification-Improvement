import torch
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path

corpus = NLPTaskDataFetcher.load_classification_corpus(Path('flair_data/'), 
                                                       test_file = 'test.csv', 
                                                       dev_file = 'dev.csv', 
                                                       train_file = 'train.csv')

word_embeddings = [WordEmbeddings('glove'), 
                   FlairEmbeddings('news-forward'), 
                   FlairEmbeddings('news-backward')]

document_embeddings = DocumentLSTMEmbeddings(word_embeddings, 
                                            hidden_size = 256, 
                                            reproject_words = True,
                                            dropout = 0.2,
                                            rnn_layers = 2)

classifier = TextClassifier(document_embeddings, label_dictionary = corpus.make_label_dictionary(), multi_label = False)
trainer = ModelTrainer(classifier, corpus)

trainer.train('./', max_epochs = 20, 
              embeddings_in_memory = False, 
              mini_batch_size = 32,
              learning_rate = 0.05)