from afinn import Afinn


def avg_afinn_sentiment(sentence, af):
    negations = set(['not', 'dont'])
    first = True

    new_sent = []

    for i,w in enumerate(sentence):
        if first:
            new_sent.append(af.score(w))
            first = False
        else:
            new_sent.append(af.score(w))
            if sentence[i-1] in negations:
                new_sent[-1] = -new_sent[-1]

    
    if len(new_sent) == 0:
        return 0
    else:
        return sum(new_sent)/len(new_sent)



