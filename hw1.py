import re
import sys

import nltk
import numpy
from sklearn.linear_model import LogisticRegression
import codecs


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    with open(corpus_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    corpus = []
    for line in lines:
        xy = line.split('\t')
        x = xy[0].split(' ')
        y = int(xy[1].rstrip())
        corpus.append((x, y))
    return corpus


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    return word in negation_words or word.endswith('n\'t')


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    tags = nltk.pos_tag(snippet)
    negated_snippet = []
    negating = False
    N = len(snippet)
    for i in range(N):
        word = snippet[i]
        if not is_negation(word):
            if not negating:
                negated_snippet.append(word)
            else:
                if word in negation_enders or word in sentence_enders or tags[i][1] == 'JJR' or tags[i][1] == 'RBR':
                    negating = False
                    negated_snippet.append(word)
                else:
                    negated_snippet.append('NOT_' + word)
        elif word == 'not' and i < N - 1 and snippet[i + 1] == 'only':
            negated_snippet.append('not')
        else:
            negating = True
            if word.endswith('n\'t'):
                negated_snippet.append(word[:len(word) - 3])
                negated_snippet.append('n\'t')
            else:
                negated_snippet.append(word)
    return negated_snippet



# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    pos = 0
    d = {}
    for snippet, label in corpus:
        for word in snippet:
            if not word in d.keys():
                d[word] = pos
                pos += 1
    return d
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    arr = numpy.zeros(len(feature_dict))
    for word in snippet:
        if word in feature_dict:
            arr[feature_dict[word]] += 1
    return arr


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    n = len(corpus)
    d = len(feature_dict)
    X = numpy.empty((n, d))
    Y = numpy.empty(n)
    for i in range(n):
        snippet, label = corpus[i]
        X[i] = vectorize_snippet(snippet, feature_dict)
        Y[i] = label
    return X, Y


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    n = len(X)
    d = len(X[n - 1])
    for j in range(d):
        col_min, col_max = numpy.min(X[:, j]), numpy.max(X[:, j])
        diff = col_max - col_min
        if diff != 0:
            X[:, j] = (X[:, j] - col_min) / diff


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    corpus = load_corpus(corpus_path)
    feature_dict = get_feature_dictionary(corpus)
    x, y = vectorize_corpus(corpus, feature_dict)
    normalize(x)
    model = LogisticRegression().fit(x, y)
    return model, feature_dict


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    tp = len(Y_pred[(Y_pred == 1) & (Y_test == 1)])
    fp = len(Y_pred[(Y_pred == 1) & (Y_test == 0)])
    fn = len(Y_pred[(Y_pred == 0) & (Y_test == 1)])
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2 * p * r / (p + r)
    return p, r, f


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    corpus = load_corpus(corpus_path)
    n = len(corpus)
    tagged_corpus = []
    for i in range(n):
        snippet, label = corpus[i]
        tagged_corpus.append((tag_negation(snippet), label))
    x, y = vectorize_corpus(tagged_corpus, feature_dict)

    normalize(x)
    y_pred = model.predict(x)
    return evaluate_predictions(y_pred, y)    


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    weights = [(index, value) for index, value in enumerate(logreg_model.coef_[0])]
    sorted_weights = sorted(weights, key=lambda x: abs(x[1]), reverse=True)
    inv_map = {v: k for k, v in feature_dict.items()}
    return [(inv_map[index], weight) for index, weight in sorted_weights[:k]]


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
