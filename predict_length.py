import logging
from pathlib import Path
from pprint import pformat
from typing import Tuple, Optional, List

from keras.models import Model, load_model
from keras.layers import GRU, Dense, Input

from nltk.corpus import swadesh
import numpy as np
from sklearn.model_selection import train_test_split

import util


class LengthModelWrapper(util.ModelWrapper):

    def __init__(self, model: Model, pronunciations, letter_ids):
        self.model = model
        self.pronunciations = pronunciations
        self.letter_ids = letter_ids
        max_len = util.get_max_length(pronunciations.values())
        self.max_len = max_len


    def predict(self, words, round_preds=False):
        """
        Accepts an iterable of words,
        converts them to one-hot letter ID representations,
        and returns an array of predicted lengths.
        :param words: sequence, iterable
            An iterable of strings whose phonemic lengths should be predicted.
        :param round_preds: (optional) bool
            If True, predictions will be rounded to the nearest integer.
        :return: array[float]
            An array of predicted phonemic length as floating points
        """
        xs = words_to_ids(words, self.letter_ids, self.max_len)
        preds = self.model.predict(xs)
        return preds


def words_to_ids(words, letter_ids, maxlen):
    X = np.zeros((len(words), maxlen, len(letter_ids)),
                        dtype=bool)
    for n, word in enumerate(words):
        for i, letter in enumerate(word):
            letter_id = letter_ids[letter]
            X[n,i,letter_id] = True

    return X


def prepare_data(pronunciations, letter_ids):
    logging.info('preparing the data...')

    max_length = util.get_max_length(pronunciations)

    X = np.zeros((len(pronunciations), max_length, len(letter_ids)),
                 dtype=bool)
    y = np.zeros(len(pronunciations), dtype=int)

    for n, word in enumerate(pronunciations):
        for i, letter in enumerate(word):
            letter_id = letter_ids[letter]
            X[n, i, letter_id] = True

        y[n] = len(pronunciations[word])

    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=21)

    word_input = Input(shape=(X.shape[1:]))
    x = GRU(256, input_shape=(X.shape[1:]), return_sequences=True)(word_input)
    x = GRU(256)(x)
    predictions = Dense(1)(x)
    model = Model(input=word_input, output=predictions)
    model.compile(optimizer='rmsprop', loss='mse',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=4, nb_epoch=3,
              validation_data=(X_test, y_test))
    return model


def get_model(fp, X=None, y=None) -> Model:
    logging.info('getting the model...')

    if Path(fp).exists():
        model = load_model(fp)
    else:
        if X is None or y is None:
            raise ValueError("You must provide training data if model doesn't exist")
        model = train_model(X, y)
        model.save(fp)
    return model


def test_accuracy(words, model_wrapper: LengthModelWrapper) -> Tuple[float, List]:
    correct = 0
    errors = []

    preds = model_wrapper.predict(words)
    pronunciations = model_wrapper.pronunciations

    for word, pred in zip(words, preds):
        prediction = pred[0]
        predicted_length = round(prediction)
        real_length = len(pronunciations[word])
        if predicted_length == real_length:
            correct += 1
        else:
            errors.append((word, prediction, real_length))

    accuracy = correct / len(words)
    return accuracy, errors


def test_swadesh(model, lang) -> Tuple[Optional[float], Optional[List]]:
    swadesh_langs = set(swadesh.fileids())
    if lang in swadesh_langs:
        logging.info('Testing model on Swadesh list for {}...'.format(lang))
        # some entries in the swadesh list have multiple words
        # because they include contextual definitions
        # so we need to only take the first word
        words = swadesh.words(fileids=lang)
        words = [word.split()[0].casefold() for word in words]
        accuracy, errors = test_accuracy(words, model)
    else:
        logging.error('No Swadesh corpus for "{}"'.format(lang))
        accuracy = None
        errors = None
    return accuracy, errors


def get_wrapped_model(data_path=None):
    """
    Loads or builds a model, then wraps it in a ``LengthModelWrapper``.
    :param data_path: str, Path
        The path to the JSON file with words mapped to pronunciations.
    :return: LengthModelWrapper
        Model wrapped in a LengthModelWrapper.
    """

    if not data_path:
        data_path = Path('pronunciations_en.json')

    prons = util.load_data(data_path)
    letter_ids = util.make_sequence_ids(prons)
    # phoneme_ids = util.make_sequence_ids(prons.values())

    X, y = prepare_data(prons, letter_ids)
    model = get_model('model_lengths.h5', X, y)

    model_wrapper = LengthModelWrapper(model, prons, letter_ids)

    return model_wrapper


def main():
    data_path = Path('pronunciations_en.json')
    logging.basicConfig(level=logging.INFO)

    model_wrapper = get_wrapped_model(data_path)

    lang = data_path.stem.rsplit('_')[-1]
    accuracy, errors = test_swadesh(model_wrapper, lang)
    logging.info('Swadesh accuracy: {:.2%}'.format(accuracy))
    logging.info('errors:\n{}'.format(pformat(errors)))

    pronunciations = model_wrapper.pronunciations

    while True:
        word = input('> ')
        pred = model_wrapper.predict([word])[0][0]
        rounded_pred = round(pred)
        print('predicted length: {} ({})'.format(rounded_pred, pred))

        try:
            real_length = len(pronunciations[word])
            print('real length: {}'.format(real_length))
        except KeyError:
            print('"{}" is not the dictionary'.format(word))

if __name__ == '__main__':
    main()
