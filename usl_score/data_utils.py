import random
import nltk
from nltk.tokenize import RegexpTokenizer


def apply_word_order(tokens):
    random.shuffle(tokens)
    return " ".join(tokens)

def apply_word_drop(tokens):
    drop_threshold = random.random() / 2
    arr = []
    for token in tokens:
        val = random.random()
        if val < drop_threshold :
            continue
        arr.append(token)
    # keep on running until there is at least one token
    if len(arr) == 0 or len(arr) == len(tokens):
        return apply_word_drop(tokens)
    return " ".join(arr)

def apply_word_repeat(tokens):
    arr = []
    for token in tokens:
        arr.append(token)

        # check whether or not to duplicate
        if random.random() < 0.9:
            continue

        # get how many words to duplicate
        max_word = min(len(arr), 3)
        word_probab = [1]
        if max_word == 2:
            word_probab = [0.75, 1]
        elif max_word == 3:
            word_probab = [0.7, 0.9, 1]

        threshold = random.random()
        for idx, val in enumerate(word_probab):
            if threshold < val:
                break
        max_word = idx+1

        # get how many time to duplicate
        threshold2 = random.random()
        time_probab = [0.8, 0.95, 1]
        for idx, val in enumerate(time_probab):
            if threshold2 < val:
                break
        time = idx+1

        arr += (arr[-max_word:]) * time

    if (len(arr) == len(tokens)):
        return apply_word_repeat(tokens)
    return " ".join(arr)

def apply_word_reverse(tokens):
    tokens = tokens[::-1]
    return " ".join(tokens)

def apply_retain_noun(tokens):
    arr = []
    pos_tags = nltk.pos_tag(tokens)
    for token, pos in pos_tags:
        if pos in ['NN', 'NNP']:
            arr.append(token)
    return " ".join(arr)

def apply_remove_puntuation(tokens):
    if tokens[-1] in ['.', '!', '?', '...', '!!!', '!!', '..', ',', ',,']:
        return " ".join(tokens[:-1])
    return " ".join(tokens)

def apply_remove_stopwords(tokens):
    #stopwords = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'])
    #stopwords = set(['a', 'an', 'the', 'have', 'has', 'had', 'be', 'is', 'am', 'are', 'was', 'were', 'been', 'will', 'can', 'may', 'would', 'could', 'might'])
    stopwords = set(['a', 'an', 'the', 've', '\'', 's', 'll', 'd', 're', 'm', 't'])
    arr = []
    for token in tokens:
        if token not in stopwords:
            arr.append(token)
    return " ".join(arr)

def apply_simplify_response(tokens):
    tags = nltk.pos_tag(tokens)
    arr = []
    for word, tag in tags:
        if tag in ['NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBZ', 'VBN', 'VBG', 'VBD', 'IN']:
            arr.append(word)
    return " ".join(arr)
