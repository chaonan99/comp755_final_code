#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from glob import glob
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm
from flashtext import KeywordProcessor


__author__ = ['chaonan99']


class Dumper:
    """Wacky corpus processor extracts sentences Wack .xml files and dumps
    into .txt files.
    """
    def __init__(self, output_dir, max_snum_per_file=5000, byteout=True):
        self._output_dir = output_dir
        self._max_snum = max_snum_per_file
        self._byteout = byteout

    def __enter__(self):
        self._fileid = 0
        self._snum_current = 0
        self._snum_total = 0
        self._fout = self._get_file_and_next()
        return self

    def new_sentence(self):
        if self._snum_current >= self._max_snum:
            self._switch_file()
        self._started = True
        self._buffer = []

    def start_new_file(self):
        self._started = False

    def _switch_file(self):
        self._fout.close()
        self._snum_current = 0
        self._fout = self._get_file_and_next()

    def end_sentence(self):
        if len(self._buffer) > 0:
            if self._byteout:
                self._fout.write(b' '.join(self._buffer))
                self._fout.write(b'\n')
            else:
                self._fout.write(' '.join(self._buffer))
                self._fout.write('\n')
            self._snum_current += 1
            self._snum_total += 1

    def add_sentence(self, sentence):
        self.new_sentence()
        self._buffer = sentence.split(b' ' if self._byteout else ' ')
        self.end_sentence()

    def add_word(self, word):
        if self._started:
            self._buffer.append(word)

    def _get_file_and_next(self):
        filename = os.path.join(self._output_dir, f'{self._fileid:06d}.txt')
        self._fileid += 1
        return open(filename, 'wb' if self._byteout else 'w')

    def __exit__(self, type, value, traceback):
        self._fout.close()


def process_wacky(data_root, output_dir):
    """5000 sentence per file results in 17642 files.
    """
    with Dumper(output_dir, max_snum_per_file=5000) as dumper:
        for ind in [1, 2, 3, 4, 5]:
            with open(data_root.format(ind), 'rb') as fin:
                dumper.start_new_file()
                ## Only an approximation
                for l in tqdm(fin, total=500000000):
                    if l == b'<s>\n':
                        dumper.new_sentence()
                    elif l == b'</s>\n':
                        dumper.end_sentence()
                    elif len(l) == 0:
                        continue
                    else:
                        dumper.add_word(l.split(b'\t')[0])


def wacky_sentence_dump(root_path):
    output_dir = '/playpen2/home/.chaonan99/data/nlp/WaCKy/all_utf8/'
    for full_path in tqdm(glob(os.path.join(root_path, '*.txt'))):
        dirname, filename = os.path.split(full_path)
        out_path = os.path.join(output_dir, filename)
        with open(full_path, 'rb') as fin, open(out_path, 'w') as fout:
            sents = [l.decode('utf-8', 'ignore').rstrip() for l in fin]
            fout.write('\n'.join(sents))


def wacky_vocabulary(root_path):
    """Takes around 15 min to read all files
    1614654 words appear > 5 times
    229462 words appear > 100 times
    """
    word_counter = Counter()
    for full_path in tqdm(glob(os.path.join(root_path, '*.txt'))):
        # root, fileids = os.path.split(full_path)
        # bnc_reader = BNCCorpusReader(root=root, fileids=fileids)
        # words = bnc_reader.words()
        with open(full_path, 'rb') as f:
            sents = [l.decode('utf-8', 'ignore').rstrip() for l in f]
            words = [w for l in sents for w in l.split(' ')]
        word_counter.update(words)

    from IPython import embed; embed(); os._exit(1)

    common_words = word_counter.most_common()
    common_words_5 = list(filter(lambda x: x[1] > 5, common_words))
    common_words_100 = list(filter(lambda x: x[1] > 100, common_words))
    print(f'Appear > 5: {len(common_words_5)}')
    print(f'Appear > 100: {len(common_words_100)}')
    with open('../../../data/vocabulary/vocab_wacky_100.txt', 'w') as f:
        f.write('\n'.join([w[0] for w in common_words_100]))

    from IPython import embed; embed(); os._exit(1)


def wacky_sentence_length_stat(root_path):
    """Not divided on .
    7.360% of all sentences has length > 50
    1.087% of all sentences has length > 100
    Divided on .
    2.337% of all sentences has length > 50
    0.256% of all sentences has length > 100
    """
    all_length = []
    for file in tqdm(glob(os.path.join(root_path, '*.txt'))):
        with open(file, 'r') as f:
            # sents = [s for l in f.readlines() \
            #            for s in l.rstrip().split('.')]
            sents = [l for l in f]
            lengths = [len(s.split(' ')) for s in sents]
            all_length.extend(lengths)

    all_length = np.array(all_length)
    print(f'{(all_length > 50).sum() / all_length.shape[0] * 100:.3f}% '
          'of all sentences has length > 50')

    from IPython import embed; embed(); os._exit(1)


def dump_random_sample(root_path, dump_path, folds=100):
    all_files = list(glob(os.path.join(root_path, '*.txt')))
    random.shuffle(all_files)
    current_dir = os.getcwd()
    if not os.path.isdir(dump_path):
        os.mkdir(dump_path)
    os.chdir(dump_path)

    for i, data_file in enumerate(all_files):
        if i % folds == folds - 1:
            root_rel_to_dump = os.path.relpath(root_path, dump_path)
            _, filename = os.path.split(data_file)
            linksrc = os.path.join(root_rel_to_dump, filename)
            os.symlink(linksrc, filename)

    os.chdir(current_dir)


def train_dev_test_split(root_path):
    all_files = list(glob(os.path.join(root_path, '*.txt')))
    random.shuffle(all_files)
    for i in range(len(all_files)):
        if i % 10 == 9 or i % 10 == 3:
            save_dir = test_dir
        elif i % 10 == 6:
            save_dir = dev_dir
        else:
            save_dir = train_dir


def main():
    data_root = '../../../data/nlp/WaCKy/ukwac{}.xml'
    output_dir = '/playpen2/home/.chaonan99/data/nlp/WaCKy/all/'
    output_dir_utf8 = '/playpen2/home/.chaonan99/data/nlp/WaCKy/all_utf8/'

    # data_root = 'test/ukwac_test.xml'
    # output_dir = 'test'

    # process_wacky(data_root, output_dir)
    # wacky_vocabulary(output_dir)
    # wacky_sentence_dump(output_dir)
    # wacky_sentence_length_stat(output_dir_utf8)

    dump_dir = '/playpen2/home/.chaonan99/data/nlp/WaCKy/test_sample'
    dump_random_sample(output_dir_utf8, dump_dir)


if __name__ == '__main__':
    main()