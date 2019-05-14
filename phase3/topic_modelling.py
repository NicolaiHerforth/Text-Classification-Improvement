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
    print("Adding topic features")
    if not test:
        print("Train set registered, computing sentiment and topics")
        adjectives = set([synset.name().split('.')[0]
                        for synset in list(wn.all_synsets(wn.ADJ))])

        dataset = data
        documents = dataset['reviewText']
        new_df = pd.DataFrame({'document': documents})
        tokenized_doc = new_df['document']
        tokenized_doc = tokenized_doc.apply(
            lambda x: [item for item in x if item in adjectives])

        # de-tokenization
        detokenized_doc = []
        for i in range(len(new_df)):
            t = ' '.join(tokenized_doc[i])
            detokenized_doc.append(t)

        new_df['document'] = detokenized_doc
        vectorizer = TfidfVectorizer(stop_words='english',
                                    max_features=100,  # keep top 1000 terms
                                    max_df=0.9,
                                    smooth_idf=True)

        X = vectorizer.fit_transform(new_df['document'])
        # SVD represent documents and terms in vectors
        svd_model = TruncatedSVD(n_components=25, algorithm='randomized', n_iter=100, random_state=122)

        svd_model.fit(X)
        terms = vectorizer.get_feature_names()
        topics = []
        for i, comp in enumerate(svd_model.components_):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
            topics.append(sorted_terms[0][0])

        cleaned_topics = list(set(topics))
        print("Writing topics")
        print(cleaned_topics)
        with open('topics.txt', 'w') as topic_file:
            for item in cleaned_topics:
                topic_file.write(item + '\n')
        #add features to df
        df = data
        df = pd.concat([df,pd.DataFrame(columns = cleaned_topics)],sort=False)
        df = df.fillna(int(0))
        print("Adding topics to train")
        print(cleaned_topics)
        for i, row in df.iterrows():
            intersect = set(row['reviewText']) & set(cleaned_topics)
            for word in intersect:
                df.at[i,word] = 1
        return df

    else:
        print("Test registered, writing topics to dataframe")
        cleaned_topics = []
        print("Opening topic file")
        with open('topics.txt', 'r') as topic_file:
            for line in topic_file.readlines():
                cleaned_topics.append(line.strip("\n"))
        #print(data.head(2))
        print("Adding test topics")
        print(cleaned_topics)
        
        df = data
        df = pd.concat([df,pd.DataFrame(columns = cleaned_topics)],sort=False)
        df = df.fillna(int(0)) 
        for i, row in df.iterrows():
            intersect = set(row['reviewText']) & set(cleaned_topics)
            for word in intersect:
                df.at[i,word] = 1
        return df


