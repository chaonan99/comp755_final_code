#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import argparse
from itertools import chain

import numpy as np

from afflm.utils.tflm.helper_dataloader import \
     DirectoryTxtDataLoader, \
     DirectoryTxtDumper
from afflm.utils.tflm.tfidf import TfIdf


__author__ = ['chaonan99']


"""Max n-gram 'n'
"""
MAX_N = 4

"""How many sentences to keep
"""
FILTER_NUM = 1000000

"""Max sentence length. If the value is None, does not filter long sentences.
"""
MAX_LENGTH = 50


class SimilarityScorer(object):
    """SimilarityScorer

    Similarity score indicates the relevance of a sentence to a corpus. We use
    this to filter the training data of the language model to only include those
    relevant to our test corpus (which now is selectional restriction prediction
    tasks P2007, W2015). This idea follows a paper titiled ``A Simple Method for
    Commonsense Reasoning''.
    """
    def __init__(self, ref_file):
        super(SimilarityScorer, self).__init__()
        self._ref_file = ref_file

    @classmethod
    def _get_ngram_for_line(cls, line, n):
        output = []
        for i in range(len(line) - n + 1):
            output.append(' '.join(line[i: i+n]))
        return set(output)

    def _prepo_f1(self):
        if not hasattr(self, '_grams_list'):
            print('Construct Ngram list')
            self._grams_list, self._grams_counts = {}, {}
            with open(self._ref_file) as f:
                lines = [l.rstrip() for l in f]

            for n in range(1, MAX_N + 1):
                ngram_func = lambda x: self._get_ngram_for_line(x.split(' '), n)
                ngrams_each_line = map(ngram_func, lines)
                self._grams_list[n] = set(chain.from_iterable(ngrams_each_line))
                self._grams_counts[n] = len(self._grams_list[n])

    def _compute_f1(self, sent):
        """Method used in original LM paper (Answered by Trieu Trinh):

        The F1 score is:

        2 * precision * recall / (precision + recall)

        Where

        precision = overlapping_count / document_count
        recall = overlapping_count / WSC_count

        For a specific n, you extract all n_gram in the document and the WSC
        corpus

        Then the size of the intersection between these two sets is
        overlapping_count
        The size of the n_gram set in the document is document_count
        The size of the n_gram set in the WSC corpus is WSC_count

        The similarity score defined in the paper is::

            sum(n * F1[n]) / sum(n)
        """
        sum_f1, sum_n = 0, 0
        for n in range(1, MAX_N + 1):
            sent_ngram = self._get_ngram_for_line(sent.split(' '), n)
            ref_ngram = self._grams_list[n]
            overlapping_count = len(sent_ngram & ref_ngram)
            sent_count = len(sent_ngram)
            ref_count = self._grams_counts[n]
            f1 = 2 * overlapping_count / (sent_count + ref_count)
            sum_f1 += n*f1
            sum_n += n

        return sum_f1 / sum_n

    def _prepo_tfidf(self):
        if not hasattr(self, '_tfidf_table'):
            table = TfIdf()
            with open(self._ref_file) as f:
                for i, l in enumerate(f):
                    table.add_document(str(i), l.rstrip().split(' '))

            self._tfidf_table = table

    def _compute_tfidf(self, sent):
        """This use the 3rd party tf-idf package.
        [TODO] Check the implementation of tf-idf to make sure it's correct!!!
        """
        scores = self._tfidf_table.similarities(sent.split(' '))
        return np.mean(list(map(lambda x: x[1], scores)))

    def all_scores(self, data_root, method):
        """Collect similarity scores on the whole corpus
        """
        getattr(self, '_prepo_' + method)()
        data_loader = DirectoryTxtDataLoader(data_root)
        sentences = data_loader.sentence_iter(filter_length=MAX_LENGTH)
        compute_func = getattr(self, '_compute_' + method)
        return [compute_func(sent) for sent in sentences]

    def filter_threshold(self, data_root, save_path, threshold, method='f1'):
        getattr(self, '_prepo_' + method)()
        data_loader = DirectoryTxtDataLoader(data_root)
        data_dumper = DirectoryTxtDumper(save_path)
        sentences = data_loader.sentence_iter(filter_length=MAX_LENGTH)
        compute_func = getattr(self, '_compute_' + method)

        for sent in sentences:
            score = compute_func(sent)
            if score >= threshold:
                data_dumper.dump(sent)

    @classmethod
    def filter_random(cls, data_root, save_path, filter_num):
        data_loader = DirectoryTxtDataLoader(data_root)
        data_dumper = DirectoryTxtDumper(save_path)
        sentences = list(data_loader.sentence_iter(filter_length=MAX_LENGTH))
        random.shuffle(sentences)
        data_dumper.dump_all(sentences[:filter_num])


def hist_plot(all_scores, method):
    """Plot histogram on all scores
    """
    import matplotlib.pyplot as plt

    plt.hist(all_scores, bins=20)
    plt.savefig(f'dump/hist_{method}.pdf')
    plt.show()


def document_selection(ref_file, data_root, save_path, method='f1'):
    """Call score compute and threshold filter functions
    """
    scorer = SimilarityScorer(ref_file)
    all_scores = scorer.all_scores(data_root, method)

    all_scores.sort(reverse=True)
    all_scores = np.array(all_scores)
    threshold = all_scores[FILTER_NUM]
    from IPython import embed; embed(); import os; os._exit(1)

    scorer.filter_threshold(data_root, save_path, threshold, method)
    from IPython import embed; embed(); import os; os._exit(1)


def document_selection_random(ref_file, data_root, save_path):
    scorer = SimilarityScorer(ref_file)
    scorer.filter_random(data_root, save_path, FILTER_NUM)


def _test_ngram_construct():
    scorer = SimilarityScorer('test/test_ngram.txt')
    scorer._construct_grams_list()
    print(scorer._grams_list)


def _test_document_selection(ref_file=None, data_root=None, save_path=None):
    ref_file = ref_file or '../../../data/Selectional_Restrictions/' \
                           'Pylkkanen2007_processed.txt'
    data_root = data_root or '../../../data/BNC/splits/traindev/'
    save_path = save_path or '../../../data/BNC/splits/tfidf_1m_sf50'

    document_selection(ref_file, data_root, save_path, 'tfidf')
    # document_selection_random(ref_file, data_root, save_path)


def main(args):
    # _test_ngram_construct()
    _test_document_selection()
    # document_selection(**args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_file',
                        help='Reference file')
    parser.add_argument('--data_root',
                        help='The root directory containing a collection of ' \
                             'txt files')
    parser.add_argument('--save_path',
                        help='The root directory to save results')
    parser.add_argument('--method', default='f1', choices=['f1', 'tfidf'],
                        help='Which method to use')
    args = parser.parse_args()
    main(args)