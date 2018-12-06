#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset: sentence language model
"""

import glob
import random

import numpy as np

from bilm.data import LMDataset, BidirectionalLMDataset, Vocabulary, \
     UnicodeCharsVocabulary


__author__ = ['chaonan99']
__copyright__ = "Copyright 2018, Haonan Chen"


class VocabularyPad(Vocabulary):
    '''
    A token vocabulary with a padding of sentence.  Holds a map from token to
    ids and provides a method for encoding text to a sequence of ids.
    '''
    _bos_tag = '<sos>'
    _eos_tag = '<eos>'
    _unk_tag = '<unk>'
    _pad_tag = '<pad>'
    def __init__(self, filename, validate_file=False):
        '''
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line.  In addition, the file should also
            contain the special tokens <S>, </S>, <UNK> (case sensitive).
        '''
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1
        self._pad = -1

        with open(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == self._bos_tag:
                    self._bos = idx
                elif word_name == self._eos_tag:
                    self._eos = idx
                elif word_name == self._unk_tag:
                    self._unk = idx
                elif word_name == self._pad_tag:
                    self._pad = idx

                # if word_name == '!!!MAXTERMID':
                #     continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

        # check to ensure file has special tokens
        if validate_file:
            if self._bos == -1 or self._eos == -1 or \
               self._unk == -1 or self._pad == -1:
                raise ValueError('Ensure the vocabulary file has '
                                 '<sos>, <eos>, <unk>, <pad> tokens')

    @property
    def pad(self):
        return self._pad

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids
                                                 if cur_id != self.pad])

    def pad_indexed_sentence(self, indexes, pad_to_length):
        if len(indexes) > pad_to_length:
            print('WARNING: sentence length exceed max length')
        return np.pad(indexes, (0, max(0, pad_to_length - len(indexes))),
                      'constant', constant_values=(0, self.pad))


class UnicodeCharsVocabularyPad(VocabularyPad, UnicodeCharsVocabulary):
    """Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.

    WARNING: for prediction, we add +1 to the output ids from this
    class to create a special padding id (=0).  As a result, we suggest
    you use the `Batcher`, `TokenBatcher`, and `LMDataset` classes instead
    of this lower level class.  If you are using this lower level class,
    then be sure to add the +1 appropriately, otherwise embeddings computed
    from the pre-trained model will be useless.
    """
    def __init__(self, filename, max_word_length, **kwargs):
        UnicodeCharsVocabularyPad.__mro__[1].__init__(self, filename, **kwargs)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260  # <padding>

        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length],
            dtype=np.int32)

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r

        def _make_pad():
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            return r

        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)
        self.pad_chars = _make_pad()

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars
        self._word_char_ids[self.pad] = self.pad_chars
        # TODO: properly handle <UNK>

    def pad_indexed_sentence_chars(self, index_arr, pad_to_length=50):
        pad_length = pad_to_length - index_arr.shape[0]
        if pad_length < 0:
            print('WARNING: sentence length exceed max length')
        return np.vstack([index_arr] + [self.pad_chars] * max(0, pad_length))

    def decode_chars(self, char_ids):
        words = []
        for row in char_ids:
            if row[0] == self.pad_char:
                continue
            chars = []
            for a in row:
                if a < self.bos_char:
                    chars.append(chr(a))
                elif a == self.bos_char:
                    chars.append(self._bos_tag)
                elif a == self.eos_char:
                    chars.append(self._eos_tag)
            words.append(''.join(chars))
        return ' '.join(words).strip()


class SentenceDataset(LMDataset):
    """
    Hold a language model dataset on sentence level.

    A dataset is a list of tokenized files.  Each file contains one sentence per
        line.  If no tokenizer is provided, each sentence should be
        pre-tokenized and white space joined.
    """
    def __init__(self, filepattern, vocab, reverse=False, test=False,
                 shuffle_on_load=False, tokenizer=None):
        '''
        filepattern = a glob string that specifies the list of files.
        vocab = an instance of Vocabulary or UnicodeCharsVocabulary
        reverse = if True, then iterate over tokens in each sentence in reverse
        test = if True, then iterate through all data once then stop.
            Otherwise, iterate forever.
        tokenizer = a custom tokenizer
        shuffle_on_load = if True, then shuffle the sentences after loading.
        '''
        self._vocab = vocab
        self._all_shards = glob.glob(filepattern)
        print('Found %d shards at %s' % (len(self._all_shards), filepattern))
        self._shards_to_choose = []

        self._reverse = reverse
        self._test = test
        self._tokenizer = tokenizer
        self._shuffle_on_load = shuffle_on_load
        self._use_char_inputs = hasattr(vocab, 'encode_chars')

        # self._ids = self._load_random_shard()

    def _reset(self):
        self._shards_to_choose = list(self._all_shards)
        self._i, self._nids = 0, 0
        random.shuffle(self._shards_to_choose)

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            raise StopIteration
        return self._shards_to_choose.pop()

    def _load_shard(self, shard_name):
        """Read one file and convert to ids.

        Args:
            shard_name: file path.

        Returns:
            list of (id, char_id) tuples.
        """
        print('Loading data from: %s' % shard_name)
        with open(shard_name) as f:
            sentences_raw = f.readlines()

        if self._tokenizer is not None:
            sentences_raw = list(map(lambda x: ' '.join(self._tokenizer(x)),
                                    sentences_raw))

        if self._reverse:
            sentences = []
            for sentence in sentences_raw:
                splitted = sentence.split()
                splitted.reverse()
                sentences.append(' '.join(splitted))
        else:
            sentences = sentences_raw

        if self._shuffle_on_load:
            random.shuffle(sentences)

        ids = [self.vocab.encode(sentence, self._reverse)
               for sentence in sentences]
        if self._use_char_inputs:
            chars_ids = [self.vocab.encode_chars(sentence, self._reverse)
                     for sentence in sentences]
        else:
            chars_ids = [None] * len(ids)

        print('Loaded %d sentences.' % len(ids))
        print('Finished loading')
        return list(zip(ids, chars_ids))

    def iter_batches(self, batch_size, seq_length):
        """Unlike the PyTorch counterpart, this generator ensures a full batch.
        The last batch, if less than batch size, will be discarded.
        """
        self._reset()
        generator = self.get_sentence()
        while True:
            inputs, targets, lengths = [], [], []
            char_inputs = [] if self._use_char_inputs else None
            i = 0
            while i < batch_size:
                ## This will raise StopIteration exception when sentences
                #  exhausted
                indexes, indexes_arr = list(next(generator))
                length = indexes.shape[0] - 1
                if length > seq_length:
                    continue

                input = self.vocab.pad_indexed_sentence(
                            indexes[:-1], seq_length)
                if self._use_char_inputs:
                    char_input = self.vocab.pad_indexed_sentence_chars(
                                indexes_arr[:-1, :], seq_length)
                target = self.vocab.pad_indexed_sentence(
                            indexes[1:], seq_length)

                inputs.append(input)
                if self._use_char_inputs:
                    char_inputs.append(char_input)
                targets.append(target)
                lengths.append(length)
                i += 1

            X = {'token_ids': inputs, 'tokens_characters': char_inputs,
                 'next_token_id': targets, 'lengths': lengths}
            yield X


def test_original_dataset_implementation():
    """Trying to show how the original `LMDataset`
    and `BidirectionalLMDataset` works.
    """
    from bilm.data import LMDataset, BidirectionalLMDataset, \
         UnicodeCharsVocabulary

    test_prefix = 'data/test/violin_test.txt'
    vocab_path = 'dump/bilm_pretrain/vocab-2016-09-10.txt'

    vocabulary = UnicodeCharsVocabulary(vocab_path, max_word_length=50)
    dataset = LMDataset(test_prefix, vocabulary)
    a = dataset.iter_batches(batch_size=10, num_steps=50)
    b = next(a)
    print(f'Keys: {b.keys()}')
    for k, v in b.items():
        print(f'Shape of {k}: {v.shape}')

    print(vocabulary.decode(b['token_ids'][0]))
    print(vocabulary.decode(b['next_token_id'][0]))
    print(vocabulary.decode_chars(b['tokens_characters'][0]))

    from IPython import embed; embed(); import os; os._exit(1)


def test_vocabulary_pad():
    import nltk

    vocab_path = 'data/vocabulary/vocab_bnc_5.txt'
    test_sents = ["People can have the Model T in any color "
                  "– so long as it's black.",
                  "I saw a bank that said '24-Hour Banking', "
                  "but I didn't have that much time."]
    vocabulary = VocabularyPad(vocab_path, validate_file=True)
    for sent in test_sents:
        tokenized_sent = nltk.word_tokenize(sent)
        indexes = vocabulary.encode(tokenized_sent, split=False)
        print(f'Index shape: {indexes.shape}')
        indexes = vocabulary.pad_indexed_sentence(indexes, pad_to_length=30)
        print(f'Padded index shape: {indexes.shape}')
        decoded_sent = vocabulary.decode(indexes)
        print(tokenized_sent)
        print(decoded_sent)

    from IPython import embed; embed(); import os; os._exit(1)


def test_unichar_vocabulary_pad():
    import nltk

    vocab_path = 'data/vocabulary/vocab_bnc_5.txt'
    test_sents = ["People can have the Model T in any color "
                  "– so long as it's black.",
                  "I saw a bank that said '24-Hour Banking', "
                  "but I didn't have that much time."]

    vocabulary = UnicodeCharsVocabularyPad(vocab_path,
                                           max_word_length=12,
                                           validate_file=True)
    for sent in test_sents:
        tokenized_sent = nltk.word_tokenize(sent)
        indexes = vocabulary.encode_chars(tokenized_sent, split=False)
        print(f'Index shape: {indexes.shape}')
        indexes = vocabulary.pad_indexed_sentence_chars(indexes,
                                                        pad_to_length=30)
        print(f'Padded index shape: {indexes.shape}')
        # from IPython import embed; embed(); import os; os._exit(1)
        decoded_sent = vocabulary.decode_chars(indexes)
        print(tokenized_sent)
        print(decoded_sent)

    from IPython import embed; embed(); import os; os._exit(1)


def test_sentence_dataset():
    test_prefix = 'data/test/violin_test.txt'
    vocab_path = 'data/vocabulary/vocab_bnc_5.txt'
    batch_size = 6
    seq_length = 20
    max_word_length = 15

    vocabulary = UnicodeCharsVocabularyPad(vocab_path,
                                           max_word_length=max_word_length)
    dataset = SentenceDataset(test_prefix, vocabulary)
    a = dataset.iter_batches(batch_size=batch_size, seq_length=seq_length)
    b = next(a)
    assert len(b['token_ids']) == batch_size
    assert b['token_ids'][0].shape[0] == seq_length
    assert b['tokens_characters'][0].shape[1] == max_word_length

    print(vocabulary.decode(b['token_ids'][0]))
    print(vocabulary.decode_chars(b['tokens_characters'][0]))
    print(vocabulary.decode(b['next_token_id'][0]))

    from IPython import embed; embed(); import os; os._exit(1)


def main():
    test_sentence_dataset()


if __name__ == '__main__':
    main()