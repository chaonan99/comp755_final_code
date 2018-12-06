# coding: utf-8
"""Training: bilm sentence input
"""
import argparse
import os
import time
import json

import tensorflow as tf
import numpy as np
from bilm.training import print_variable_summary, average_gradients, \
     clip_grads, summary_gradient_updates

from data import SentenceDataset, UnicodeCharsVocabularyPad, VocabularyPad
from model import SentenceLanguageModel
from config import Config


__author__ = 'chaonan99'
__copyright__ = 'Copyright 2018, Haonan Chen'


def train(options, data, n_gpus, tf_save_dir, tf_log_dir, logger,
          restart_ckpt_file=None):
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # set up the optimizer
        lr = options.get('learning_rate', 0.2)
        opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                        initial_accumulator_value=1.0)

        # calculate the gradients on each GPU
        tower_grads = []
        models = []
        train_perplexity = tf.get_variable(
            'train_perplexity', [],
            initializer=tf.constant_initializer(0.0), trainable=False)
        norm_summaries = []
        for k in range(n_gpus):
            with tf.device('/gpu:%d' % k):
                with tf.variable_scope('lm', reuse=k > 0):
                    # calculate the loss for one model replica and get
                    #   lstm states
                    model = SentenceLanguageModel(options, True)
                    loss = model.total_loss
                    models.append(model)
                    # get gradients
                    grads = opt.compute_gradients(
                        loss * options['unroll_steps'],
                        aggregation_method=\
                            tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )
                    tower_grads.append(grads)
                    # keep track of loss across all GPUs
                    train_perplexity += loss

        print_variable_summary()

        # calculate the mean of each gradient across all GPUs
        grads = average_gradients(tower_grads, options['batch_size'], options)
        grads, norm_summary_ops = clip_grads(grads, options, True, global_step)
        norm_summaries.extend(norm_summary_ops)

        # log the training perplexity
        train_perplexity = tf.exp(train_perplexity / n_gpus)
        perplexity_summmary = tf.summary.scalar(
            'train_perplexity', train_perplexity)

        # some histogram summaries.  all models use the same parameters
        # so only need to summarize one
        histogram_summaries = [
            tf.summary.histogram('token_embedding', models[0].embedding)
        ]
        # tensors of the output from the LSTM layer
        lstm_out = tf.get_collection('lstm_output_embeddings')
        histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_0', lstm_out[0]))
        if options.get('bidirectional', False):
            # also have the backward embedding
            histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_1', lstm_out[1]))

        # apply the gradients to create the training operation
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # histograms of variables
        for v in tf.global_variables():
            histogram_summaries.append(\
                tf.summary.histogram(v.name.replace(":", "_"), v))

        # get the gradient updates -- these aren't histograms, but we'll
        # only update them when histograms are computed
        histogram_summaries.extend(
            summary_gradient_updates(grads, opt, lr))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        summary_op = tf.summary.merge(
            [perplexity_summmary] + norm_summaries
        )
        hist_summary_op = tf.summary.merge(histogram_summaries)

        init = tf.initializers.global_variables()

    # do the training loop
    bidirectional = options.get('bidirectional', False)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)

        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            loader = tf.train.Saver()
            loader.restore(sess, restart_ckpt_file)

        summary_writer = tf.summary.FileWriter(tf_log_dir, sess.graph)

        # For each batch:
        # Get a batch of data from the generator. The generator will
        # yield batches of size batch_size * n_gpus that are sliced
        # and fed for each required placeholer.

        batch_size = options['batch_size']
        unroll_steps = options['unroll_steps']
        epochs = options['n_epochs']
        log_interval = options['log_interval']
        checkpoint_interval = options['checkpoint_interval']

        char_inputs = 'char_cnn' in options
        logger.info('Start training loop')

        t1 = time.time()
        for epoch in range(epochs):
            data_gen = data.iter_batches(batch_size * n_gpus, unroll_steps)
            for batch_no, batch in enumerate(data_gen, start=1):
                # slice the input in the batch for the feed_dict
                X = batch
                feed_dict = {}
                for k in range(n_gpus):
                    model = models[k]
                    start = k * batch_size
                    end = (k + 1) * batch_size

                    feed_dict.update(
                        _get_feed_dict_from_X(X, start, end, model,
                                              char_inputs, bidirectional)
                    )

                if batch_no % checkpoint_interval != 0:
                    ret = sess.run(
                        [train_op, summary_op, train_perplexity],
                        feed_dict=feed_dict
                    )
                else:
                    # also run the histogram summaries
                    ret = sess.run(
                        [train_op, summary_op,
                         train_perplexity, hist_summary_op],
                        feed_dict=feed_dict
                    )

                if batch_no % checkpoint_interval == 0:
                    summary_writer.add_summary(ret[3], batch_no)
                if batch_no % log_interval == 0:
                    # write the summaries to tensorboard and display perplexity
                    summary_writer.add_summary(ret[1], batch_no)
                    logger.info(f'Epoch {epoch} | Batch {batch_no} | '
                                f'train_perplexity={ret[2]}')
                    logger.info(f'Total time: {time.time() - t1}')

                if batch_no % checkpoint_interval == 0:
                    # save the model
                    checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)

            checkpoint_path = os.path.join(tf_save_dir,
                                           f'model_epoch{epoch:02d}.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)


def _get_feed_dict_from_X(X, start, end, model, char_inputs, bidirectional):
    feed_dict = {}
    if not char_inputs:
        token_ids = X['token_ids'][start:end]
        feed_dict[model.token_ids] = token_ids
    else:
        # character inputs
        char_ids = X['tokens_characters'][start:end]
        feed_dict[model.tokens_characters] = char_ids

    if bidirectional:
        if not char_inputs:
            feed_dict[model.token_ids_reverse] = \
                X['token_ids_reverse'][start:end]
        else:
            feed_dict[model.tokens_characters_reverse] = \
                X['tokens_characters_reverse'][start:end]

    # now the targets with weights
    next_id_placeholders = [[model.next_token_id, '']]
    if bidirectional:
        next_id_placeholders.append([model.next_token_id_reverse, '_reverse'])

    for id_placeholder, suffix in next_id_placeholders:
        name = 'next_token_id' + suffix
        feed_dict[id_placeholder] = X[name][start:end]

    feed_dict[model.seq_length] = X['lengths'][start:end]

    return feed_dict


def test(options, ckpt_file, data, batch_size=256):
    '''
    Get the test set perplexity!
    '''

    bidirectional = options.get('bidirectional', False)
    char_inputs = 'char_cnn' in options
    if char_inputs:
        max_chars = options['char_cnn']['max_characters_per_token']

    unroll_steps = options['unroll_steps']
    all_losses, all_lengths = [], []

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/cpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            # NOTE: the number of tokens we skip in the last incomplete
            # batch is bounded above batch_size * unroll_steps
            test_options['batch_size'] = batch_size
            model = SentenceLanguageModel(test_options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        t1 = time.time()
        batch_losses = []
        total_loss = 0.0
        data_gen = data.iter_batches(batch_size, unroll_steps)
        for batch_no, batch in enumerate(data_gen, start=1):
            # slice the input in the batch for the feed_dict
            X = batch
            feed_dict = _get_feed_dict_from_X(X, 0, len(X['token_ids']), model,
                                              char_inputs, bidirectional)
            ret = sess.run(
                [model.losses, model.total_loss],
                feed_dict=feed_dict
            )
            losses, total_loss = ret
            batch_perplexity = np.exp(total_loss)

            if bidirectional:
                raise NotImplementedError('Bidirectional test not implemented.')
            else:
                losses = losses[0]

            all_losses.extend(losses.tolist())
            all_lengths.extend(X['lengths'])

            print("batch=%s, batch_perplexity=%s, time=%s" %
                (batch_no, batch_perplexity, time.time() - t1))

    all_losses = np.array(all_losses).reshape(-1, unroll_steps)
    return all_losses, all_lengths


def main(args):
    config = Config(args)
    options = config.get_options()

    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
        vocab = UnicodeCharsVocabularyPad(args.vocab_file, max_word_length)
    else:
        ## Haven't tested on model yet!!!
        vocab = VocabularyPad(args.vocab_file)
    data = SentenceDataset(args.prefix, vocab, test=False, shuffle_on_load=True)
    train(options, data, int(args.ngpus), config.save_path, config.save_path,
          config.get_logger(), restart_ckpt_file=args.start_from)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', help='Name of a single run')
    parser.add_argument('--ngpus', default=1, help='Number of GPUs')
    parser.add_argument('--vocab_file',
                        default='data/vocabulary/vocab_bnc_5.txt',
                        help='Vocabulary file')
    parser.add_argument('--start_from',
                        help='Checkpoint file')
    parser.add_argument('--prefix',
                        default='data/BNC/splits/traindev/*',
                        help='Number of GPUs')

    args = parser.parse_args()
    main(args)