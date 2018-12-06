#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import shutil
from glob import glob
from collections import Counter

from tqdm import tqdm
from nltk.corpus.reader.bnc import BNCCorpusReader


__author__ = ['chaonan99']


def bnc_vocabulary(root_path):
    """This process prepares the vocabulary file for BNC. Words occur less than
    or equal to 5 times are eliminated.

    There are 4049 xml's in BNC. Total processing time ~ 1 h 30 min
    """
    word_counter = Counter()
    for full_path in tqdm(glob(os.path.join(root_path, r'*/*/*.xml'))):
        root, fileids = os.path.split(full_path)
        bnc_reader = BNCCorpusReader(root=root, fileids=fileids)
        words = bnc_reader.words()
        word_counter.update(words)

    common_words = word_counter.most_common()
    common_words_5 = list(filter(lambda x: x[1] > 5, common_words))
    with open('../../../data/vocabulary/vocab_bnc_5.txt', 'w') as f:
        f.write('\n'.join([w[0] for w in common_words_5]))


def bnc_sentence_dump(root_path):
    """This process randomly dumps sentences in xmls into txt files under train,
    dev, test split (roughly 7:1:2)
    """
    all_xmls = glob(os.path.join(root_path, r'*/*/*.xml'))
    random.shuffle(all_xmls)
    train_dir = '../../../data/BNC/train/'
    test_dir = '../../../data/BNC/test/'
    dev_dir = '../../../data/BNC/dev/'

    for directory in [train_dir, test_dir, dev_dir]:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        print(f'Creating directory {directory}')
        os.mkdir(directory)

    for i, full_path in tqdm(enumerate(all_xmls)):
        root, fileid = os.path.split(full_path)
        bnc_reader = BNCCorpusReader(root=root, fileids=fileid)
        filename, ext = os.path.splitext(fileid)

        if i % 10 == 9 or i % 10 == 3:
            save_dir = test_dir
        elif i % 10 == 6:
            save_dir = dev_dir
        else:
            save_dir = train_dir

        save_path = os.path.join(save_dir, filename + '.txt')
        with open(save_path, 'w') as f:
            f.write('\n'.join([' '.join(s) for s in bnc_reader.sents()]))


def bnc_check_train_dev_test():
    """This process validate the train, dev, test splits. Currently it check if
    there are repeated files.
    """
    train_dir = '../../../data/BNC/splits/train/'
    test_dir = '../../../data/BNC/splits/test/'
    dev_dir = '../../../data/BNC/splits/dev/'

    all_sent_files = []
    for directory in [train_dir, test_dir, dev_dir]:
        all_sent_files.append(glob(os.path.join(directory, '*.txt')))
    sep_len = sum([len(a) for a in all_sent_files])
    set_len = len(set([aa for a in all_sent_files for aa in a]))
    print(sep_len)
    assert set_len == sep_len


def bnc_sentence_length_stat():
    """This process gives statistics on sentence length of BNC
    Sentence reading takes about 30 s

    1.641% of all sentences has length > 50 (divided at .)
    3.319% of all sentences has length > 50 (not divided at .)
    """
    import numpy as np

    train_dir = '../../../data/BNC/splits/train'
    test_dir = '../../../data/BNC/splits/test/'
    dev_dir = '../../../data/BNC/splits/dev/'

    all_length = []
    for directory in [train_dir, test_dir, dev_dir]:
        for file in tqdm(glob(os.path.join(directory, '*.txt'))):
            with open(file, 'r') as f:
                # sents = [s for l in f.readlines() \
                #            for s in l.rstrip().split('.')]
                sents = [l for l in f]
                lengths = [len(s.split(' ')) for s in sents]
                # if 2171 in lengths:
                #     sents_record = sents
                all_length.extend(lengths)

    all_length = np.array(all_length)
    print(f'{(all_length > 50).sum() / all_length.shape[0] * 100:.3f}% '
          'of all sentences has length > 50')

    from IPython import embed; embed(); os._exit(1)


def main():
    # bnc_reader = BNCCorpusReader(root='../../data/BNC/2554/download/Texts/',
    #                              fileids=r'A/A0/\w*\.xml')
    # sentences = bnc_reader.sents()

    bnc_vocabulary('../../../data/BNC/2554/download/Texts/')
    # bnc_sentence_dump('../../data/BNC/2554/download/Texts/')
    # bnc_check_train_dev_test()
    # bnc_sentence_length_stat()


if __name__ == '__main__':
    main()