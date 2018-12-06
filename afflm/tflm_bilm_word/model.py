"""Model: bilm on sentences
"""
import numpy as np
import tensorflow as tf

from bilm.training import LanguageModel


__author__ = ['chaonan99']
__copyright__ = 'Copyright 2018, Haonan Chen'

DTYPE = tf.float32
DTYPE_INT = 'int64'

tf.logging.set_verbosity(tf.logging.INFO)


class SentenceLanguageModel(LanguageModel):
    """docstring for SentenceLanguageModel"""
    def __init__(self, options, is_training):
        super(SentenceLanguageModel, self).__init__(options, is_training)

    def _build_loss(self, lstm_outputs):
        '''
        Create:
            self.total_loss: total loss op for training
            self.softmax_W, softmax_b: the softmax variables
            self.next_token_id / _reverse: placeholders for gold input

        '''
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        n_tokens_vocab = self.options['n_tokens_vocab']

        # DEFINE next_token_id and *_reverse placeholders for the gold input
        def _get_next_token_placeholders(suffix):
            name = 'next_token_id' + suffix
            id_placeholder = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, unroll_steps),
                                   name=name)
            return id_placeholder

        # get the window and weight placeholders
        self.next_token_id = _get_next_token_placeholders('')
        if self.bidirectional:
            self.next_token_id_reverse = _get_next_token_placeholders(
                '_reverse')

        # DEFINE seq_length placeholders
        self.seq_length = tf.placeholder(DTYPE_INT,
                                         shape=(batch_size,),
                                         name='seq_length')
        seq_mask = tf.sequence_mask(self.seq_length,
                                    unroll_steps,
                                    dtype=DTYPE)
        seq_mask = tf.squeeze(tf.reshape(seq_mask, [-1, 1]), squeeze_dims=[1])

        # DEFINE THE SOFTMAX VARIABLES
        # get the dimension of the softmax weights
        # softmax dimension is the size of the output projection_dim
        softmax_dim = self.options['lstm']['projection_dim']

        # the output softmax variables -- they are shared if bidirectional
        if self.share_embedding_softmax:
            # softmax_W is just the embedding layer
            self.softmax_W = self.embedding_weights

        with tf.variable_scope('softmax'), tf.device('/cpu:0'):
            # Glorit init (std=(1.0 / sqrt(fan_in))
            softmax_init = tf.random_normal_initializer(0.0,
                1.0 / np.sqrt(softmax_dim))
            if not self.share_embedding_softmax:
                self.softmax_W = tf.get_variable(
                    'W', [n_tokens_vocab, softmax_dim],
                    dtype=DTYPE,
                    initializer=softmax_init
                )
            self.softmax_b = tf.get_variable(
                'b', [n_tokens_vocab],
                dtype=DTYPE,
                initializer=tf.constant_initializer(0.0))

        # now calculate losses
        # loss for each direction of the LSTM
        self.individual_losses = []
        self.losses = []

        if self.bidirectional:
            next_ids = [self.next_token_id, self.next_token_id_reverse]
        else:
            next_ids = [self.next_token_id]

        for id_placeholder, lstm_output_flat in zip(next_ids, lstm_outputs):
            # flatten the LSTM output and next token id gold to shape:
            # (batch_size * unroll_steps, softmax_dim)
            # Flatten and reshape the token_id placeholders
            next_token_id_flat = tf.reshape(id_placeholder, [-1, 1])

            with tf.control_dependencies([lstm_output_flat]):
                if self.is_training and self.sample_softmax:
                    losses = tf.nn.sampled_softmax_loss(
                                   self.softmax_W, self.softmax_b,
                                   next_token_id_flat, lstm_output_flat,
                                   self.options['n_negative_samples_batch'],
                                   self.options['n_tokens_vocab'],
                                   num_true=1)

                else:
                    # get the full softmax loss
                    output_scores = tf.matmul(
                        lstm_output_flat,
                        tf.transpose(self.softmax_W)
                    ) + self.softmax_b
                    # NOTE: tf.nn.sparse_softmax_cross_entropy_with_logits
                    #   expects unnormalized output since it performs the
                    #   softmax internally
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=output_scores,
                        labels=tf.squeeze(next_token_id_flat, squeeze_dims=[1])
                    )

            losses *= seq_mask
            individual_loss = tf.reduce_sum(losses) / tf.reduce_sum(seq_mask)
            self.losses.append(losses)
            self.individual_losses.append(individual_loss)

        # now make the total loss -- it's the mean of the individual losses
        if self.bidirectional:
            self.total_loss = 0.5 * (self.individual_losses[0]
                                    + self.individual_losses[1])
        else:
            self.total_loss = self.individual_losses[0]


def main():
    import json
    from data import SentenceDataset, VocabularyPad

    option_file_path = 'dump/sentpad_test/options.json'
    test_prefix = 'data/test/violin_test.txt'
    vocab_path = 'data/vocabulary/vocab_bnc_5.txt'

    with open(option_file_path, 'r') as fin:
        options = json.load(fin)

    with tf.variable_scope('lm'):
        model = SentenceLanguageModel(options, is_training=False)

    init = tf.initializers.global_variables()
    batch_size = options['batch_size']
    max_seq_length = options['unroll_steps']

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    sess = tf.InteractiveSession()
    sess.run(init)
    vocabulary = VocabularyPad(vocab_path)
    dataset = SentenceDataset(test_prefix, vocabulary)
    a = dataset.iter_batches(batch_size=batch_size,
                             seq_length=max_seq_length)
    b = next(a)

    feed_dict = {
        model.token_ids: b['token_ids'],
        model.seq_length: b['lengths'],
        model.next_token_id: b['next_token_id']
    }
    total_loss = sess.run(model.total_loss, feed_dict=feed_dict)
    losses = sess.run(model.losses, feed_dict=feed_dict)
    print(f'Loss: {total_loss} (should be around 12)')

    from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()