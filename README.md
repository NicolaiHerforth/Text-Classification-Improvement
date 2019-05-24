# Group 2 exam project.

Link for the trained GRU models: https://drive.google.com/drive/folders/1S6CqGbuYQf4yR_ez5VkfwsmqPKJc9BmR?usp=sharing

## Instructions for reproducing the resulsts in the paper:

- The Flair model can be trained running flair_model.py (Note that this takes a long time). The trained model used in the paper can be loaded via flair_model_predict.py (This also takes a long time).

- The LSTM model can be accessed in LSTM.py, the code trains and benchmarks the model against both test sets. Due to it being cuda optimized training is fast and takes short time. 

- Sentence level Model can be accessed in sentenceLevel_model.py, the code trains and benchmarks the model in a cross valuation setup, 1 group against all. 

- The GRU All Features results can be accessed in the folder phase3 by running plotting_predicting_func_gru_model_h5.py, this model uses all features

- The GRU Reviewtext & Topics results can be accessed in the folder phase3 by running plotting_predicting_func_gru_model_ablation_no_afinn.py this model uses reviewText and topics as features

- The GRU Topics & Afinn results can be accessed in the folder phase3 by running  plotting_predicting_funct_GRU_model_ablation_no_text.py this model uses topics and afinn scores as features

- The GRU ReviewText & Afinn results can be accessed in the folder phase3 by running plotting_predicting_funct_GRU_model_ablation_no_topics.py this model uses reviewText and afinn scores as features

