# Improving Text Classification with Deep Neural Network and Word Embeddings

Link for the research paper: https://afribary.com/works/improving-text-classification-using-deep-neural-networks-and-word-embeddings

Link for the trained GRU models: https://drive.google.com/drive/folders/1S6CqGbuYQf4yR_ez5VkfwsmqPKJc9BmR?usp=sharing

## Abstract
Natural Language Processing is a flourishing aspect of Data Science in both academia and industry. This paper examines the technique of sentiment classification on text data using automated Machine learning approaches. It will delve into newly introduced embedding techniques and machine learning model pipelines. The models explored in this paper consist of a simple Naive Bayes classifier, as well as several Deep Neural Models, including a Gated Recurrent Unit (GRU) and a Long Short Term Memory network. Several different forms of numerical representations of words (word embeddings) are introduced to supplement the neural models in an attempt to build a solid sentiment predictor.
Overall, the Neural methods outperformed Naive Bayes. Furthermore, the neural methods performed very similarly; the word embeddings and suggested feature extraction yielded very little gain, leading to the conclusion that in the case of these experiments, a simple model would be preferred. Perhaps in the future for a stronger model, it would be beneficial to optimize the parameters with grid search, or by acquiring a larger variety of data for transfer learning.


### Instructions for reproducing the resulsts in the paper:

- The Flair model can be trained running flair_model.py (Note that this takes a long time). The trained model used in the paper can be loaded via flair_model_predict.py (This also takes a long time).

- The LSTM model can be accessed in LSTM.py, the code trains and benchmarks the model against both test sets. Due to it being cuda optimized training is fast and takes short time. 

- Sentence level Model can be accessed in sentenceLevel_model.py, the code trains and benchmarks the model in a cross valuation setup, 1 group against all. 

- The GRU All Features results can be accessed in the folder phase3 by running plotting_predicting_func_gru_model_h5.py, this model uses all features

- The GRU Reviewtext & Topics results can be accessed in the folder phase3 by running plotting_predicting_func_gru_model_ablation_no_afinn.py this model uses reviewText and topics as features

- The GRU Topics & Afinn results can be accessed in the folder phase3 by running  plotting_predicting_funct_GRU_model_ablation_no_text.py this model uses topics and afinn scores as features

- The GRU ReviewText & Afinn results can be accessed in the folder phase3 by running plotting_predicting_funct_GRU_model_ablation_no_topics.py this model uses reviewText and afinn scores as features

