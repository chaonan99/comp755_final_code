#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize


__author__ = ['chaonan99']


def squad_sentence_dump(corpus_json, output_file):
    """ train: 94224 sentences, takes about 20s on nlp5
    all: 100617 sentences
    """
    corpus = json.load(open(corpus_json))
    sents = []
    for doc in corpus['data']:
        for para in doc['paragraphs']:
            context = sent_tokenize(para['context'])
            sents.extend(context)

    sents = list(map(lambda x: ' '.join(word_tokenize(x)), sents))
    with open(output_file, 'w') as f:
        f.write('\n'.join(sents))

    from IPython import embed; embed(); import os; os._exit(1)

def squad_vocabulary(sent_file):
    with open(sent_file) as f:
        words = [w for l in f for w in l.rstrip().split(' ')]
    from IPython import embed; embed(); import os; os._exit(1)
    counter = Counter(words)
    common_words = counter.most_common()
    common_words_5 = list(filter(lambda x: x[1] > 5, common_words))
    with open('../../../data/vocabulary/squad_5.txt', 'w') as f:
        f.write('\n'.join([w[0] for w in common_words_5]))



def main():
    train_json = '../../../data/SQuAD/train-v2.0.json'
    dev_json = '../../../data/SQuAD/dev-v2.0.json'
    train_file = '../../../data/SQuAD/train.txt'
    dev_file = '../../../data/SQuAD/dev.txt'
    output_file = '../../../data/SQuAD/all.txt'

    # squad_sentence_dump(dev_json, dev_file)
    squad_vocabulary(output_file)


if __name__ == '__main__':
    main()