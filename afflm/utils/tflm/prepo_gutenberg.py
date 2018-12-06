#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import random
from glob import glob
from collections import Counter

from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize

from afflm.utils.tflm.helper_dataloader import DirectoryTxtDumper


__author__ = ['chaonan99']


def gutenberg_sentence_dump(data_root, output_dir):
    """26330806 sentences
    """
    sents = []
    for filename in tqdm(glob(os.path.join(data_root, '*'))):
        with open(filename, 'rb') as f:
            for l in f:
                line = l.rstrip().decode('utf-8', 'ignore')
                if len(line) > 0:
                    sents.extend(sent_tokenize(line))

    ## This will take a long time
    sents = list(map(lambda x: ' '.join(word_tokenize(x)), sents))

    dumper = DirectoryTxtDumper(output_dir)
    dumper.open()
    dumper.dump_all(sents)
    dumper.close()


def gutenberg_vocabulary(root_path):
    word_counter = Counter()
    for full_path in tqdm(glob(os.path.join(root_path, '*.txt'))):
        with open(full_path, 'r') as f:
            sents = [l.rstrip() for l in f]
            words = [w for l in sents for w in l.split(' ')]
        word_counter.update(words)

    from IPython import embed; embed(); os._exit(1)

    common_words = word_counter.most_common()
    common_words_5 = list(filter(lambda x: x[1] > 5, common_words))
    common_words_100 = list(filter(lambda x: x[1] > 100, common_words))
    print(f'Appear > 5: {len(common_words_5)}')
    print(f'Appear > 100: {len(common_words_100)}')

    with open('../../../data/vocabulary/vocab_gutenberg_5.txt', 'w') as f:
        f.write('\n'.join([w[0] for w in common_words_5]))

    from IPython import embed; embed(); os._exit(1)


def gutenberg_train_test_split(root_path, train_path, test_path, test_rate=0.1):
    all_files = list(glob(os.path.join(root_path, '*.txt')))
    random.shuffle(all_files)
    split_point = int(len(all_files) * test_rate)
    test_files = all_files[:split_point]
    train_files = all_files[split_point:]

    def create_links(files, path):
        current_dir = os.getcwd()
        os.chdir(path)
        print(os.getcwd())
        root_rel_to_dump = os.path.relpath(root_path, path)
        for file_path in files:
            _, file_name = os.path.split(file_path)
            linksrc = os.path.join(root_rel_to_dump, file_name)
            try:
                os.symlink(linksrc, file_name)
            except OSError as e:
                pass
        os.chdir(current_dir)

    create_links(train_files, train_path)
    create_links(test_files, test_path)


def main():
    data_root = '../../../data/Gutenberg/txt/'
    output_dir = '../../../data/Gutenberg/sentall/'

    train_path = '../../../data/Gutenberg/sentsplit/train'
    test_path = '../../../data/Gutenberg/sentsplit/test'

    # gutenberg_sentence_dump(data_root, output_dir)
    # gutenberg_vocabulary(output_dir)
    gutenberg_train_test_split(output_dir, train_path, test_path)

    from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()