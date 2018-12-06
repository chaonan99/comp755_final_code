#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

from afflm.utils.tflm.exp_similarity_filter import SimilarityScorer
from afflm.utils.tflm.helper_dataloader import DirectoryTxtDataLoader


__author__ = ['chaonan99']


def dataset_query_similarity(query_file, dataset_root, sim_measure):
    """Calculate dataset and query file similarity

    ``query_file`` is a file name contains a set of tokenized sentences, each in
    a separate line. ``dataset_root`` is a directory and all ``.txt`` files
    under the directory is considered a data file. Each data file contains a set
    of tokenized sentences in separate lines. The similarity scorer is built
    with the query file, and all sentences in the dataset are feed to the
    similarity scorer to compute similarities. An average score over all
    sentences is computed as the final similarity measurement of ``query_file``
    and ``dataset_root``. The similarity measurement is specified by
    ``sim_measure``.
    """

    scorer = SimilarityScorer(query_file)
    score_func = getattr(scorer, sim_measure)
    dataloader = DirectoryTxtDataLoader(dataset_root)
    sentence_iter = dataloader.sentence_iter()
    all_scores = [score_func(sent.split(' ')) for sent in sentence_iter]
    return np.mean(all_scores)


def main():
    import os

    query_root = '../../../data/Selectional_Restrictions/'
    query_file = os.path.join(query_root, 'Pylkkanen2007_processed.txt')
    # query_file = os.path.join(query_root, 'Warren2015_processed.txt')
    # dataset_root = '../../../data/WaCKy/test_sample/'
    dataset_root = '../../../data/BNC/splits/traindev/'
    sim_measure = 'similarity_score_lmpaper_1'
    # sim_measure = 'similarity_score_tfidf'
    # sim_measure = 'similarity_score_tfidf_divide_length'
    similarity = dataset_query_similarity(query_file, dataset_root, sim_measure)
    print(sim_measure, query_file, dataset_root, similarity)


if __name__ == '__main__':
    main()