#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Results for char LM
|               | Full   | Partial |
|---------------|--------|---------|
| Warren2015    | 70%    | 46.7%   |
| Pylkkanen2007 | 85.71% | 41.43%  |
"""

import argparse
import itertools

import nltk
import numpy as np
from bilm.training import load_options_latest_checkpoint, load_vocab

from config import Config
from data import SentenceDataset, UnicodeCharsVocabularyPad
from train import test  # Leverage test data for training!!! :-p


__author__ = ['chaonan99']


def bilm_predict():
    options, ckpt_file = load_options_latest_checkpoint('dump/bilm_pretrain')
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    vocab = load_vocab('dump/bilm_pretrain/vocab-2016-09-10.txt',
                       max_word_length)
    test_prefix = '../../deps/bilm-tf/tests/fixtures/train/data.txt'

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    else:
        data = LMDataset(test_prefix, vocab, **kwargs)

    perplexity = test(options, ckpt_file, data, batch_size=1)
    from IPython import embed; embed(); import os; os._exit(1)


def _sequence_mask(lengths, max_length=None):
    """Same as _sequence_mask in tensorflow
    """
    max_length = max_length or max(lengths)
    range_arr = np.tile(np.arange(max_length)[np.newaxis], (len(lengths), 1))
    return (range_arr.T < lengths).T


def _get_changed_positions(sents, num_per_group):
    """Used for partial probability
    """
    positions = []

    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    for i, j in pairwise(range(0, len(sents) + num_per_group, num_per_group)):
        sents_group = sents[i:j]
        sents_group = list(map(lambda x: x.split(' '), sents_group))
        sents_lengths = list(map(len, sents_group))
        max_length = max(sents_lengths)
        for ind, sent in enumerate(sents_group):
            for _ in range(max_length - len(sent)):
                sents_group[ind].insert(0, '')

        sents_group = np.array(sents_group)
        pos_array = np.all(sents_group == sents_group[0, :], axis=0)
        pos = np.where(~pos_array)[0][-1]
        for l in sents_lengths:
            positions.append(pos - max_length + l)

    return positions


def thematic_fit_eval(args):
    config = Config(args)
    options, ckpt_file = load_options_latest_checkpoint(config.save_path)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
        vocab = UnicodeCharsVocabularyPad(args.vocab_file, max_word_length)
    else:
        ## Not tested yet
        vocab = VocabularyPad(args.vocab_file)

    test_path = 'data/Selectional_Restrictions/Pylkkanen2007_processed.txt'
    # test_path = 'data/Selectional_Restrictions/Warren2015_processed.txt'
    # test_path = 'data/CSR/WSC_sent.txt'

    with open(test_path) as f:
        sents = [l.rstrip() for l in f.readlines()]
    num_per_group = 2 if 'WSC' in test_path else 3
    positions = _get_changed_positions(sents, num_per_group)
    data = SentenceDataset(test_path, vocab, test=True, shuffle_on_load=False,
                           tokenizer=nltk.word_tokenize)

    # if options.get('bidirectional'):
    #     data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    # else:
    #     data = LMDataset(test_prefix, vocab, **kwargs)

    all_losses, all_lengths = test(options, ckpt_file, data,
                                   batch_size=args.batch_size)

    # Full score
    print('Full probability results')
    scores = all_losses.sum(axis=1) / all_lengths
    scores = np.array(scores).reshape(-1, num_per_group)
    res = scores.argmax(axis=1)
    for i in range(num_per_group):
        print(sum(res == i) / len(res))

    # Partial score
    print('Partial probability results')
    seq_mask = _sequence_mask(np.array(positions) + 1, options['unroll_steps'])
    partial_losses = seq_mask * all_losses
    loss_mask = partial_losses > 0
    scores = partial_losses.sum(axis=1) / loss_mask.sum(axis=1)
    scores = np.array(scores).reshape(-1, num_per_group)
    res = scores.argmax(axis=1)
    for i in range(num_per_group):
        print(sum(res == i) / len(res))

    from IPython import embed; embed(); import os; os._exit(1)


def special_words():
    config = Config(args)
    options, ckpt_file = load_options_latest_checkpoint(config.save_path)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
        vocab = UnicodeCharsVocabularyPad(args.vocab_file, max_word_length)
    else:
        ## Not tested yet
        vocab = VocabularyPad(args.vocab_file)

    #test_path = 'data/Selectional_Restrictions/Pylkkanen2007_processed.txt'
    test_path = 'data/Selectional_Restrictions/Warren2015_processed.txt'

    all_losses, all_lengths = test(options, ckpt_file, data,
                                   batch_size=args.batch_size)

    from IPython import embed; embed(); import os; os._exit(1)


def main(args):
    thematic_fit_eval(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--run_name', help='Name of test run', required=True)
    parser.add_argument('--vocab_file',
                        default='data/vocabulary/vocab_bnc_5.txt',
                        help='Vocabulary file')
    parser.add_argument('--batch_size', type=int, default=30,
                        help='Batch size')

    args = parser.parse_args()
    main(args)
