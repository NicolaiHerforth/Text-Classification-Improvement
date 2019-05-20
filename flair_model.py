from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
import torch

torch.cuda.is_available()
print(torch.__version__)
torch.cuda.get_device_name(0)
torch.cuda.is_available()


corpus = NLPTaskDataFetcher.load_classification_corpus(Path('flair_data/'), 
                                                       test_file = 'test.csv', 
                                                       dev_file = 'dev.csv', 
                                                       train_file = 'train.csv')

word_embeddings = [WordEmbeddings('glove'), 
                   FlairEmbeddings('news-forward'), 
                   FlairEmbeddings('news-backward')]

document_embeddings = DocumentRNNEmbeddings(word_embeddings, 
                                            hidden_size = 256, 
                                            reproject_words = True, 
                                            dropout = 0.2,
                                            rnn_layers = 2)

classifier = TextClassifier(document_embeddings, label_dictionary = corpus.make_label_dictionary(), multi_label = False)
trainer = ModelTrainer(classifier, corpus)

trainer.train('./', max_epochs = 10, 
              embeddings_in_memory = False, 
              mini_batch_size = 32,
              learning_rate = 0.1)