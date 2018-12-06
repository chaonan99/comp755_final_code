#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import argparse
import operator
from glob import glob
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from flashtext import KeywordProcessor

from afflm.utils.tflm.prepo_wacky import Dumper
from afflm.utils.tflm.tfidf import TfIdf


__author__ = ['chaonan99']


def _get_ngrams(sent, n):
    words = sent if type(sent) == list else sent.split(' ')
    output = []
    for i in range(len(words) - n + 1):
        output.append(' '.join(words[i: i+n]))
    return set(output)


def _get_grams_1ton(sent, n):
    output = set()
    words = sent if type(sent) == list else sent.split(' ')
    for i in range(n):
        output |= _get_ngrams(words, i+1)
    return output

def _get_grams_dict(sent, end, start=1):
    words = sent.split(' ')
    output = dict()
    for i in range(start, end+1):
        output.update({o: i for o in _get_ngrams(words, i)})
    return output


def _test_get_grams_dict():
    sent = "I don't believe in astrology; I'm a " \
           "Sagittarius and we're skeptical."
    print(_get_grams_dict(sent, 3))


def _get_all_grams(query_file, n):
    all_grams = set()
    with open(query_file) as f:
        for l in f:
            all_grams |= _get_grams_1ton(l.rstrip(), n)
    return all_grams


def _get_all_gram_dict(query_file, n):
    all_gram_dict = {}
    with open(query_file) as f:
        for l in f:
            all_gram_dict.update(_get_grams_dict(l.rstrip(), n))
    return all_gram_dict


class SimilarityScorer(object):
    """docstring for SimilarityScorer
    LM paper method (Answered by Trieu Trinh):

    The F1 score is:

    2 * precision * recall / (precision + recall)

    Where

    precision = overlapping_count / document_count
    recall = overlapping_count / WSC_count

    For a specific n, you extract all n_gram in the document and the WSC corpus

    Then the size of the intersection between these two sets is
    overlapping_count
    The size of the n_gram set in the document is document_count
    The size of the n_gram set in the WSC corpus is WSC_count
    """
    def __init__(self, query_file, n=4):
        self._query_file = query_file
        self._all_grams = _get_all_grams(query_file, n=n)
        self._all_grams_length = len(self._all_grams)

        ## This does not take much time so just leave it in initialization
        self._keyword_processor = KeywordProcessor()
        self._keyword_processor.add_keywords_from_list(list(self._all_grams))

        ## Only used for tfidf similarity
        self._tfidf_table = self._populate_table(query_file)

        self._n = n

    @classmethod
    def _populate_table(cls, query_file):
        table = TfIdf()
        with open(query_file) as f:
            for i, l in enumerate(f):
                table.add_document(str(i), l.rstrip().split(' '))
        return table

    def _construct_grams_list(self):
        self._grams_list = {}
        with open(self._query_file) as f:
            for l in f:
                l.rstrip()

        for i in range(4):
            self._grams_list[i+1] = _get_ngrams(sent, i+1)

    def similarity_score_lmpaper_true(self, sent):
        if not hasattr(self, '_grams_list'):
            self._construct_grams_list()

        from IPython import embed; embed(); import os; os._exit(1)


    def similarity_score_lmpaper_1(self, sent):
        sent_grams = _get_grams_1ton(sent, self._n)
        overlap = self._all_grams & sent_grams
        return len(overlap) / (len(sent_grams) + self._all_grams_length)

    def similarity_score_lmpaper_2(self, sent):
        """A possibly faster version, but need unsplited sentence input
        """
        keywords = self._keyword_processor.extract_keywords(sent)
        space_count = sent.count(' ')
        gram_count = (2*space_count + 3 - self._n) * self._n / 2
        return len(keywords) / (gram_count + self._all_grams_length)

    def similarity_score_tfidf(self, sent):
        """Similarity based on tfidf, using query file as document set
        """
        scores = self._tfidf_table.similarities(sent \
                 if type(sent) == list else sent.split(' '))
        return np.mean(list(map(lambda x: x[1], scores)))

    def similarity_score_tfidf_divide_length(self, sent):
        sent = sent if type(sent) == list else sent.split(' ')
        scores = self._tfidf_table.similarities(sent)
        return np.mean(list(map(lambda x: x[1], scores))) / len(sent)

    def similarity_score_important_words(self, sent):
        """
        """
        raise NotImplementedError('Important word similarity not implemented')

    def similarity_score_nnsm(self, sent):
        """Ask yijin about it
        """
        raise NotImplementedError('NNSM similarity not implemented')


def _test_similarity_score_lmpaper():
    query_root = '../../../data/Selectional_Restrictions/'
    query_file = os.path.join(query_root, 'Pylkkanen2007_processed.txt')
    sent = "An opera begins long before the curtain goes up and ends long" \
    "after it has come down . It starts in my imagination, it becomes my " \
    "life , and it stays part of my life long after I've left the opera house ."
    scorer = SimilarityScorer(query_file)
    print(scorer.similarity_score_lmpaper_1(sent))
    print(scorer.similarity_score_lmpaper_2(sent))


def _test_tfidf_similarity_score():
    query_root = 'test/tfidf_test_query_file.txt'
    scorer = SimilarityScorer(query_root)
    sent = "The woman still remembers the reason why the civil service makes " \
           "her patience bleeding at the time of feudalism ."
    scores = scorer.similarity_score_tfidf(sent.split(' '))
    from IPython import embed; embed(); import os; os._exit(1)


def similarity_filter_wacky(root_path, query_file, output_dir=None):
    output_dir = output_dir or '../../../data/WaCKy/tmp/'
    scorer = SimilarityScorer(query_file)
    # score_func = scorer.similarity_score_tfidf_divide_length
        ## 0.00016817846937428402
    score_func = scorer.similarity_score_lmpaper_1  ## 0.0021551724137931034
    # score_func = scorer.similarity_score_tfidf  ## 0.0037980620940262205

    all_scores, all_lengths = [], []
    threshold = 0.0021551724137931034

    with Dumper(output_dir, max_snum_per_file=5000,
                byteout=False) as dumper:
        for full_path in tqdm(glob(os.path.join(root_path, '*.txt'))):
            with open(full_path, 'r') as f:
                for l in f:
                    sent = l.rstrip()
                    sent_split = sent.split(' ')
                    score = score_func(sent)
                    # score = score_func(sent_split)
                    all_scores.append(score)
                    if score >= threshold:
                        all_lengths.append(len(sent_split))
                        dumper.add_sentence(sent)

    print(f'Average filtered sentence length: {np.mean(all_lengths)}')
    print(f'Average score between WaCKy and P2007 data: {np.mean(all_scores)}')

    from IPython import embed; embed(); os._exit(1)

    ## Pick threshold
    import matplotlib.pyplot as plt
    plt.hist(all_scores)
    all_scores = np.array(all_scores)
    all_scores.sort()
    threshold = all_scores[len(all_scores) - 100000]
    print(threshold)
    np.sum(np.array(all_scores) >= threshold)


def similarity_filter_bnc(root_path, query_file, output_dir=None):



def main():
    data_root = '../../../data/WaCKy/test_sample/'
    query_root = '../../../data/Selectional_Restrictions/'
    # query_file = os.path.join(query_root, 'Warren2015_processed.txt')
    query_file = os.path.join(query_root, 'Pylkkanen2007_processed.txt')

    # output_dir = '../../../data/WaCKy/relevant/p2007_lmpaper_1_th006/'
    output_dir = '../../../data/WaCKy/relevant/sample_p2007_tfidf_th005/'
    output_dir = None

    # similarity_filter_wacky_1(data_root, query_file, output_dir)
    similarity_filter_wacky(data_root, query_file, output_dir)
    # _test_similarity_score_lmpaper()
    _test_tfidf_similarity_score()

    # _test_get_grams_dict()


if __name__ == '__main__':
    main()