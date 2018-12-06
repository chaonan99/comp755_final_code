#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rnnagr.agreement_acceptor import PredictVerbNumber
from rnnagr import filenames


__author__ = ['chaonan99']


def main():
    pvn = PredictVerbNumber(filenames.deps, prop_train=0.1)
    pvn.pipeline()
    from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()