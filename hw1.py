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
    # correct line 1092
    corpus[1091][0][3] = 'can\'t'
    corpus[1091][0].pop(4)
    corpus[1091][0].pop(4)
    corpus[1091][0][10] = 'kilmer\'s'
    corpus[1091][0].pop(11)
    corpus[1091][0].pop(11)
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
    negating, append_only = False, False
    N = len(snippet)
    for i in range(N):
        word = snippet[i]
        if append_only:
            negated_snippet.append(word)
            append_only = False
            continue
        if not is_negation(word):
            if not negating:
                negated_snippet.append(word)
            else:
                if word in negation_enders or word in sentence_enders or tags[i] == 'JJR' or tags[i] == 'RBR':
                    negating = False
                    negated_snippet.append(word)
                else:
                    negated_snippet.append('NOT_' + word)
        elif word == 'not' and i < N - 1 and snippet[i + 1] == 'only':
            negated_snippet.append('not')
            append_only = True
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
    pass
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    pass


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    pass


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    pass


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    M = load_corpus(corpus_path)
    print(tag_negation(M[10][0]))


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    pass


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    pass


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    pass


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
