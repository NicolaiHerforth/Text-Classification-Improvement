from afinn import Afinn

def afinn_sentiment(sentence):
    af = Afinn()
    negations = set(['not', 'dont'])
    first = True
    for i in range(len(sentence)):
        if first:
            sentence[i] = af.score(sentence[i])
            first = False
        else:
            sentence[i] = af.score(sentence[i])
            if sentence[i-1] in set(['not' or :
                sentence[i] = -sentence[i]
    
    return sum(sentence)/len(sentence)



