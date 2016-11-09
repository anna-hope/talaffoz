import logging
from itertools import chain
from pathlib import Path
from typing import Dict, List

import nltk
import numpy as np
from keras.engine import Input, Model
from keras.layers import GRU, Dense
from keras.models import load_model
from sklearn.model_selection import train_test_split

from predict_length import get_wrapped_model
import util


def make_ngram_mappings(pronunciations: Dict[str, List[str]]):
    # each word will have a beginning character (*)
    # and a word-end character (#)
    pronunciations = {'**' + word + '##': pronunciation
                      for word, pronunciation in pronunciations.items()}

    # we iterate over words and pronunciations in both directions
    # give each ngram an id
    # and make it point to a phoneme

    # assign an id to each letter
    all_letters = sorted(set(chain.from_iterable(pronunciations)))
    letter_ids = {letter: n for n, letter in enumerate(all_letters)}

    # assign an id to each ipa symbol
    all_ipa = sorted(set(chain.from_iterable(pronunciations.values())))
    ipa_ids = {symbol: n for n, symbol in enumerate(all_ipa)}

    ngrams_phonemes = []

    # the n-gram length
    n = 5

    for word, pronunciation in pronunciations.items():
        word_ngrams = list(nltk.ngrams(word, n))

        for ngram, phoneme in zip(word_ngrams, pronunciation):
            ngram_ids = [letter_ids[letter] for letter in ngram]

            ngrams_phonemes.append((ngram_ids, ipa_ids[phoneme]))

        # now do the same from right to left
        # because letters and phonemes often don't correspond 1:1
        for ngram, phoneme in zip(reversed(word_ngrams),
                                  reversed(pronunciation)):
            ngram_ids = [letter_ids[letter] for letter in ngram]
            ngrams_phonemes.append((ngram_ids, ipa_ids[phoneme]))

    return letter_ids, ipa_ids, ngrams_phonemes


def prepare_data_ngrams(pronunciations: Dict[str, List[str]], n_gram=5):
    logging.info('preparing the data...')

    letter_ids, ipa_ids, ngrams_phonemes = make_ngram_mappings(pronunciations)

    # encode ngrams as one-hot of each letter id
    X = np.zeros((len(ngrams_phonemes), n_gram, len(letter_ids)),
                 dtype=bool)
    y = np.zeros((len(ngrams_phonemes), len(ipa_ids)), dtype=bool)

    for n, (ngram_ids, phoneme_id) in enumerate(ngrams_phonemes):
        for i, letter_id in enumerate(ngram_ids):
            X[n, i, letter_id] = True

        y[n, phoneme_id] = True

    return X, y, letter_ids, ipa_ids


def build_model_ngrams(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)
    ngram_inputs = Input(shape=X.shape[1:])

    x = GRU(64, input_shape=X.shape[1:], return_sequences=True)(ngram_inputs)
    x = GRU(64)(x)
    predictions = Dense(y.shape[1], activation='softmax')(x)
    model = Model(input=ngram_inputs, output=predictions)

    model.compile(optimizer='rmsprop', loss='mse',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=256, nb_epoch=3,
              validation_data=(X_test, y_test))

    return model


def word_to_ngrams(word: str, letter_ids: dict) -> np.ndarray:
    if not word.startswith('**'):
        word = '**' + word

    if not word.endswith('##'):
        word += '##'

    word_ngrams = list(nltk.ngrams(word, 5))
    x = np.zeros((len(word_ngrams), 5, len(letter_ids)),
                 dtype=bool)

    for n, ngram in enumerate(word_ngrams):
        for i, letter in enumerate(ngram):
            letter_id = letter_ids[letter]
            x[n, i, letter_id] = True

    return x


def ngrams_to_word(ngrams_as_ids, ids_letters):
    word = []

    for i in range(ngrams_as_ids.shape[0]):
        context = ngrams_as_ids[i]

        for element in context:
            letter_id = np.argmax(element)
            letter = ids_letters[letter_id]
            word.append(letter)

    return word


def get_model(X, y, path='model.h5', rebuild=False):
    if Path(path).exists() and not rebuild:
        model = load_model(path)
    else:
        model = build_model_ngrams(X, y)
        model.save(path)

    return model


def pred_to_ipa(pred: np.ndarray, ids2symbols: dict, verbose=False):
    ipa = []
    highest_probs = []
    for prediction in pred:
        max_prob = max(prediction)
        symbol_id = np.argmax(prediction)
        symbol = ids2symbols[symbol_id]
        ipa.append(symbol)
        highest_probs.append(max_prob)

        if verbose:
            print(max_prob, symbol)

    return ipa, highest_probs


def trim_to_length(ipa, highest_probs, word, length_model):
    trimmed_prediction = ipa[:]
    predicted_length = length_model.predict([word])[0][0]
    predicted_length = round(predicted_length)

    while len(highest_probs) > predicted_length:
        lowest_prob_segment = np.argmin(highest_probs)
        del highest_probs[lowest_prob_segment]
        del trimmed_prediction[lowest_prob_segment]

    return trimmed_prediction


def main():
    logging.basicConfig(level=logging.INFO)

    data_path = 'pronunciations_en.json'
    pronunciations = util.load_data(data_path)
    X, y, letter_ids, ipa_ids = prepare_data_ngrams(pronunciations)

    if '#' not in ipa_ids:
        ipa_ids['#'] = len(ipa_ids)

    print('X', X.shape)
    print('y', y.shape)

    ids2ipa = {symbol_id: symbol for symbol, symbol_id
               in ipa_ids.items()}

    model = get_model(X, y, path='model_ngrams.h5')

    length_model = get_wrapped_model(data_path)

    while True:
        try:
            word = input('> ')
            x1 = word_to_ngrams(word, letter_ids)
            pred = model.predict(x1)
            ngram_pron, probs = pred_to_ipa(pred, ids2ipa,
                                                   verbose=True)
            trimmed_pron = trim_to_length(ngram_pron, probs,
                                          word, length_model)
            print(' '.join(ngram_pron))
            print(' '.join(trimmed_pron))
        except Exception as e:
            print(e, e.args)
            continue


if __name__ == '__main__':
    main()