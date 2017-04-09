import logging
from argparse import ArgumentParser
from itertools import zip_longest, chain
from pathlib import Path
from typing import Dict, List

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import (Dense, LSTM, GRU, TimeDistributed,
                          Input, RepeatVector, Embedding)
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split

from util import load_data


class MappingOutput(Callback):
    ids2elements = None
    X_val = None
    y_val = None

    def on_epoch_end(self, epoch, logs={}):
        preds = self.model.predict(self.X_val)
        preds_ipa = pred_to_ipa(preds, self.ids2elements)
        val_ipa = pred_to_ipa(self.y_val, self.ids2elements)
        correct = 0

        for pred, val in zip(preds_ipa, val_ipa):
            pred = trim_output(pred)
            val = trim_output(val)
            if pred == val:
                correct += 1

            pred = ' '.join(pred)
            val = ' '.join(val)

            print(pred, ' --- ', val)

        accuracy = correct / self.X_val.shape[0]
        print('correct: {:.2%}'.format(accuracy))


def bucket_data(pronunciations: Dict[str, List[str]]):
    buckets = [(10, 10), (15, 15), (20, 20), (35, 35)]
    data_buckets = [[] for _ in range(len(buckets))]

    end_char = '#'

    for word, pronunciation in pronunciations.items():
        for n, (bucket_word, bucket_pron) in enumerate(buckets):
            if len(word) <= bucket_word and len(pronunciation) <= bucket_pron:
                word = [c for c, _ in zip_longest(word, range(bucket_word),
                                                  fillvalue=end_char)]
                pronunciation = [s for s, _ in zip_longest(pronunciation,
                                                           range(bucket_pron),
                                                           fillvalue=end_char)]
                data_buckets[n].append((word, pronunciation))
                break

    return buckets, data_buckets


def prepare_data(pronunciations, max_length=0):
    logging.info('preparing the data...')
    end_char = '#'

    if not max_length:
        longest_input = max(len(input) for input, _ in pronunciations)
        longest_output = max(len(output) for _, output in pronunciations)
        max_length = max((longest_input, longest_output))

    padded_pronunciations = []
    for word, pronunciation in pronunciations:

        if len(word) <= max_length and len(pronunciation) <= max_length:
            word = [c for c, _ in zip_longest(word, range(max_length),
                                              fillvalue=end_char)]
            pronunciation = [s for s, _ in zip_longest(pronunciation, range(max_length),
                                                       fillvalue=end_char)]

            padded_pronunciations.append((word, pronunciation))

    pronunciations = padded_pronunciations

    # assign an id to each letter
    pronunciations_keys = (element[0] for element in pronunciations)
    all_letters = sorted(set(chain.from_iterable(pronunciations_keys)))
    letter_ids = {letter: n for n, letter in enumerate(all_letters, 1)}
    letter_ids[end_char] = 0

    # assign an id to each ipa symbol
    pronunciations_values = (element[1] for element in pronunciations)
    all_ipa = sorted(set(chain.from_iterable(pronunciations_values))) # experiment
    ipa_ids = {symbol: n for n, symbol in enumerate(all_ipa, 1)}
    ipa_ids[end_char] = 0

    # one-hot
    X = np.zeros((len(pronunciations), max_length, len(letter_ids)+1),
                 dtype='int32')
    y = np.zeros((len(pronunciations), max_length, len(ipa_ids)+1),
                 dtype='int32')

    # X = np.zeros((len(pronunciations), max_length),
    #              dtype='int32')
    # y = np.zeros((len(pronunciations), max_length),
    #              dtype='int32')

    for n, (word, pronunciation) in enumerate(pronunciations):
        for i, c in enumerate(word):
            letter_id = letter_ids[c]
            # X[n, i] = letter_id
            X[n, i, letter_id] = 1

        for i, ipa_symbol in enumerate(pronunciation):
            symbol_id = ipa_ids[ipa_symbol]
            # y[n, i] = symbol_id
            y[n, i, symbol_id] = 1

    return X, y, letter_ids, ipa_ids, max_length


def build_model(n_letters, n_symbols, max_len,
                input_layers=1, output_layers=1,
                hidden_units=512):
    inputs = Input(shape=(max_len, n_letters+1))
    # inputs = Input(shape=(max_len,))
    # encoded = Embedding(output_dim=512, input_dim=n_letters+1,
    #                     input_length=max_len, mask_zero=True)(inputs)

    # add the encoder layers

    # assign this variable
    # so that multiple encoder layers
    # can be stacked in a loop
    encoded = inputs
    for _ in range(input_layers - 1):
        encoded = LSTM(hidden_units, return_sequences=True)(encoded)

    encoded = LSTM(hidden_units)(encoded)

    decoded = RepeatVector(max_len)(encoded)

    # add the decoder layers
    for _ in range(output_layers):
        decoded = LSTM(hidden_units, return_sequences=True)(decoded)

    predictions = TimeDistributed(
                    Dense(n_symbols+1,
                          activation='softmax'))(decoded)

    model = Model(inputs=inputs, outputs=predictions)
    print(model.inputs, model.output_shape)

    optimizer = Adam()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def encode_words(words, letter_ids: dict, max_length: int):

    end_char = '#'
    padded_words = []

    for word in words:
        word = [c for c, _ in zip_longest(word, range(max_length),
                                          fillvalue=end_char)]
        padded_words.append(word)

    word_ids = np.zeros((len(padded_words), max_length, len(letter_ids)+1),
                        dtype=bool)

    for n, word in enumerate(padded_words):
        for i, letter in enumerate(word):
            letter_id = letter_ids[letter]
            word_ids[n, i, letter_id] = True

    return word_ids


def pred_to_ipa(pred: np.ndarray, ids2symbols: dict, verbose=False):
    for prediction in pred:
        ipa = []
        highest_probs = []
        for timestep in prediction:
            max_prob = max(timestep)
            symbol_id = np.argmax(timestep)
            symbol = ids2symbols[symbol_id]
            ipa.append(symbol)

            if verbose:
                print(max_prob, symbol)
        yield ipa


def train_model(model, encoded, val_output, fp='model.h5',
                batch_size=5, n_epochs=5, verbose=True):
    # stop after 1 epoch of no improvement
    early_stopping = EarlyStopping(patience=1)
    model_checkpoint = ModelCheckpoint(fp, save_best_only=True)

    callbacks = [early_stopping, model_checkpoint]
    if verbose:
        callbacks.append(val_output)

    for X, y in encoded:
        bucket_size = X.shape[1], y.shape[1]
        logging.info('training on bucket {}'.format(bucket_size))

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33)

        X_val = X_test[:100]
        y_val = y_test[:100]

        val_output.X_val = X_val
        val_output.y_val = y_val

        model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs,
                  validation_data=(X_test, y_test),
                  callbacks=callbacks)

    return model


def trim_output(ipa, end_char='#'):
    ipa_trimmed = []
    for symbol in ipa:
        if symbol == end_char:
            break
        ipa_trimmed.append(symbol)

    return ipa_trimmed


def main(prons_fp, model_fp, max_length,
         n_input_layers, n_output_layers, n_hidden_units,
         n_epochs, batch_size, show_intermediate_output=False):
    logging.basicConfig(level=logging.INFO)

    data_path = prons_fp
    pronunciations = load_data(data_path)
    # encoded, letter_ids, ipa_ids, buckets = prepare_data(pronunciations)
    X, y, letter_ids, ipa_ids, max_length = prepare_data(pronunciations, max_length=max_length)

    assert '#' in letter_ids and '#' in ipa_ids
    # print('buckets', buckets)
    print('X', X.shape)
    print('y', y.shape)

    ids2ipa = {symbol_id: symbol for symbol, symbol_id
               in ipa_ids.items()}

    model_path = Path(model_fp)
    if model_path.exists():
        logging.info('loading the model "{}"...'.format(model_path))
        model = load_model(str(model_path))
    else:
        val_output = MappingOutput()
        val_output.ids2elements = ids2ipa

        logging.info('building a new model...')
        model = build_model(len(letter_ids), len(ipa_ids), max_length,
                            input_layers=n_input_layers,
                            output_layers=n_output_layers,
                            hidden_units=n_hidden_units)
        train_model(model, [(X, y)], val_output,
                    n_epochs=n_epochs, batch_size=batch_size,
                    verbose=show_intermediate_output)
        model.save(model_fp)

    while True:
        try:
            words = input('> ').split('  ')

            # in case we are doing sentences, not words
            words = [w.split() for w in words]

            x = encode_words(words, letter_ids, max_length)
            pred = model.predict(x, batch_size=1)
            for pron in pred_to_ipa(pred, ids2ipa):
                trimmed_pron = trim_output(pron)
                print(' '.join(trimmed_pron), end='  ')
            print()

        except Exception as e:
            print(e, e.args)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--file', help='Path to the pronunciation file.',
                            required=True)
    arg_parser.add_argument('--model-file', help='Path to the model file.',
                            default='model.h5')
    arg_parser.add_argument('--max-length', type=int, default=0,
                            help=('Only input sequences up to this length'
                                  + ' will be included.\n'
                                  + 'Defaults to the longest input sequence.'))
    arg_parser.add_argument('--n-input-layers', type=int, default=1,
                            help='The number of layers in the encoder.')
    arg_parser.add_argument('--n-output-layers', type=int, default=1,
                            help='The number of layers in the decoder.')
    arg_parser.add_argument('--n-hidden-units', type=int, default=512,
                            help='The number of hidden units in each layer.')
    arg_parser.add_argument('--n-epochs', type=int, default=5,
                            help='The number of epochs used to train the model.')
    arg_parser.add_argument('--show-intermediate-output', action='store_true',
                            help='Show intermediate output between training epochs.')
    arg_parser.add_argument('--batch-size', type=int, default=5,
                            help='Size of the mini-batch used in training.')
    args = arg_parser.parse_args()
    main(args.file, args.model_file, args.max_length,
         args.n_input_layers, args.n_output_layers,
         args.n_hidden_units, args.n_epochs, args.batch_size,
         show_intermediate_output=args.show_intermediate_output)
