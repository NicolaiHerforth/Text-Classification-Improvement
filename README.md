# Group 2 exam project.

## Instructions for reproducing the resulsts in the paper:

- The Flair model can be trained running flair_model.py (Note that this takes a long time). The trained model used in the paper can be loaded via flair_model_predict.py (This also takes a long time).

- The LSTM model can be accessed in LSTM.py, the code trains and benchmarks the model against both test sets. Due to it being cuda optimized training is fast and takes short time. 

- Sentence level Model can be accessed in sentenceLevel_model.py, the code trains and benchmarks the model in a cross valuation setup, 1 group against all. 

- GRU All Features

- GRU Reviewtext & Topics
- GRU Topics & Afinn
- GRU ReviewText & Afinn
