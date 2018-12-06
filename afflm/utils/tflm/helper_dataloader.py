#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import tee
from glob import glob

import numpy as np
from tqdm import tqdm


__author__ = ['chaonan99']


class DirectoryTxtDataLoader:
    """DirectoryTxtDataLoader

    A simple dataloader for textual dataset, in which data files are in .txt
    format under a directory ``data_root`` and each file contains tokenized
    sentences, each in a separate line.
    """
    def __init__(self, data_root):
        self._data_root = data_root

    def sentence_iter(self, show_progress=True,
                      filter_length=None, ext='*.txt'):
        filelist = glob(os.path.join(self._data_root, ext))
        it = tqdm(filelist) if show_progress else filelist
        for data_file in it:
            with open(data_file) as fp:
                for l in fp:
                    line = l.rstrip()
                    if filter_length is not None:
                        words = line.split(' ')
                        if len(words) <= filter_length:
                            yield line
                        else:
                            pass
                    else:
                        yield line


class DirectoryTxtDumper:
    """Wacky corpus processor extracts sentences Wack .xml files and dumps
    into .txt files.
    """
    def __init__(self, output_dir, max_snum_per_file=5000):
        self._output_dir = output_dir
        self._max_snum = max_snum_per_file

    def open(self):
        self._fileid = 0
        self._snum_current = 0
        self._fout = self._get_file_and_next()

    def dump_all(self, sents):

        from itertools import tee

        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        if not hasattr(self, '_fout'):
            self.open()

        step = self._max_snum
        for i, j in pairwise(range(0, len(sents) + step, step)):
            self._fout.write('\n'.join(sents[i:j]))
            self._fout = self._get_file_and_next()

    def _switch_file(self):
        self._fout.close()
        self._snum_current = 0
        self._fout = self._get_file_and_next()

    def dump(self, sent):
        if not hasattr(self, '_snum_current'):
            self.open()
        if self._snum_current >= self._max_snum:
            self._switch_file()
        self._fout.write(sent)
        self._fout.write('\n')
        self._snum_current += 1

    def close(self):
        self._fout.close()

    def _get_file_and_next(self):
        filename = os.path.join(self._output_dir, f'{self._fileid:06d}.txt')
        self._fileid += 1
        return open(filename, 'w')


def sentence_count(data_root, ext='*.txt'):
    loader = DirectoryTxtDataLoader(data_root)
    sentence_iter = loader.sentence_iter(show_progress=False, ext=ext)
    all_lengths = []
    for sent in tqdm(sentence_iter):
        all_lengths.append(len(sent.split(' ')))
    print(f'Number of sentences {len(all_lengths)}')
    print(f'Average token number per sentence {np.mean(all_lengths)}')


def main():
    bnc_data_root = '../../../data/BNC/splits/traindev'
    wacky_data_root = '../../../data/WaCKy/all_utf8'
    wacky_test_sample_data_root = '../../../data/WaCKy/test_sample'
    squad_data_root = '../../../data/SQuAD'
    oneb_data_root = '../../../data/1Billion/' \
                     'training-monolingual.tokenized.shuffled/'
    gutenberg_data_root = '../../../data/Gutenberg/sentall'

    # sentence_count(bnc_data_root)
    # sentence_count(wacky_data_root)
    # sentence_count(wacky_test_sample_data_root)
    # sentence_count(squad_data_root, ext='all.txt')
    # sentence_count(oneb_data_root, ext='*')
    sentence_count(gutenberg_data_root, ext='*')


if __name__ == '__main__':
    main()