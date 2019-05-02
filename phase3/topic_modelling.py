import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import wordnet as wn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

pd.set_option("display.max_colwidth", 200)

def add_topic_features(data, test=False):
    if not test:
        adjectives = set([synset.name().split('.')[0]
                        for synset in list(wn.all_synsets(wn.ADJ))])

        dataset = data
        documents = dataset['reviewText']
        news_df = pd.DataFrame({'document': documents})
        news_df = news_df.fillna('')

        # removing everything except alphabets`
        news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
        # removing short words
        news_df['clean_doc'] = news_df['clean_doc'].apply(
            lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
        # make all text lowercase
        news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
        # tokenization
        tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
        # remove stop-words
        tokenized_doc = tokenized_doc.apply(
            lambda x: [item for item in x if item in adjectives])
        # de-tokenization
        detokenized_doc = []
        for i in range(len(news_df)):
            t = ' '.join(tokenized_doc[i])
            detokenized_doc.append(t)

        news_df['clean_doc'] = detokenized_doc
        vectorizer = TfidfVectorizer(stop_words='english',
                                    max_features=1000,  # keep top 1000 terms
                                    max_df=0.5,
                                    smooth_idf=True)

        X = vectorizer.fit_transform(news_df['clean_doc'])
        # SVD represent documents and terms in vectors
        svd_model = TruncatedSVD(
            n_components=25, algorithm='randomized', n_iter=100, random_state=122)

        svd_model.fit(X)
        terms = vectorizer.get_feature_names()
        topics = []
        for i, comp in enumerate(svd_model.components_):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
            topics.append(sorted_terms[0][0])
        cleaned_topics = list(set(topics))
        
        with open('topics.txt', 'w') as topic_file:
            for item in cleaned_topics:
                topic_file.write(item + '\n')
        #add features to df
        df = data
        df = pd.concat([df,pd.DataFrame(columns = cleaned_topics)],sort=False)
        df = df.fillna(int(0))
        for i, row in df.iterrows():
            intersect = set(row['reviewText']) & set(cleaned_topics)
        for word in intersect:
            df[word][i] = 1
        return df

    elif test == True:
        cleaned_topics = []
        with open('topics.txt', 'r') as topic_file:
            for line in topic_file.readlines():
                cleaned_topics.append(line)
        df = pd.read_csv(data)
        df = pd.concat([df,pd.DataFrame(columns = cleaned_topics)],sort=False)
        df = df.fillna(int(0))
        for i, row in df.iterrows():
            intersect = set(row['reviewText']) & set(cleaned_topics)
        for word in intersect:
            df[word][i] = 1
        return df


