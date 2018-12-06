#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nltk
import xmltodict


__author__ = ['chaonan99']


def prepo_wsc(data_root, save_path):
    with open(data_root) as f:
        res = xmltodict.parse(f.read(), process_namespaces=True)

    with open(save_path, 'w') as f:
        for schema in res['collection']['schema']:
            txt1 = schema['text']['txt1']
            txt2 = schema['text']['txt2']

            sents = []
            for ans in schema['answers']['answer']:
                sent = ' '.join([txt1, ans , txt2])
                sent = ' '.join(nltk.word_tokenize(sent))
                sents.append(sent)

            if 'B' in schema['correctAnswer']:
                sents[0], sents[1] = sents[1], sents[0]

            f.write('\n'.join(sents))
            f.write('\n')

    from IPython import embed; embed(); import os; os._exit(1)


def main():
    data_root = '../../../data/CSR/WSCollection.xml'
    save_path = '../../../data/CSR/WSC_sent.txt'
    prepo_wsc(data_root, save_path)


if __name__ == '__main__':
    main()