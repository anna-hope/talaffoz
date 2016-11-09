import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

from collections import Counter, defaultdict
import json
from pathlib import Path


def load_data(file_path):
    print('loading the data file {}'.format(file_path))
    file_path = Path(file_path)
    with file_path.open() as f:
        pronunciations = json.load(f)

    words_ipa = {word: v['ipa']
                 for word, v in pronunciations.items()}
    return words_ipa


def find_element_distributions(seqs):
    element_counts = Counter()
    element_positions = defaultdict(Counter)

    for seq in seqs:
        for n, element in enumerate(seq):
            element_counts[element] += 1
            element_position = round(n / len(seq), 3)
            element_positions[element][element_position] += 1

    lb = LabelBinarizer()
    lb.fit(sorted(element_counts.keys()))

    total_elements = sum(element_counts.values())

    distributions = []
    labels = []

    for element, count in element_counts.items():
        element_freq = count / total_elements
        positions = element_positions[element].most_common(10)
        if len(positions) == 10:
            positions = [e[0] for e in positions][:10]
            labels.append(element)
            distributions.append([element_freq] + positions)

    labels = lb.transform(labels)
    return distributions, labels, lb


def get_dists(words, pronunciations):
    letter_dists, letter_labels, letters_lb = find_element_distributions(words)
    phoneme_dists, phoneme_labels, phonemes_lb = find_element_distributions(pronunciations)

    letters = letters_lb.inverse_transform(letter_labels)
    phonemes = letters_lb.inverse_transform(phoneme_labels)

    X = np.array(phoneme_dists, dtype=float)
    y = phoneme_labels
    print('X', X.shape)
    print('y', y.shape)

    X_letters = np.array(letter_dists, dtype=float)
    y_letters = letter_labels

    return X, y, X_letters, y_letters


def find_distributions(prons):
    words = list(prons)
    pronunciations = list(prons.values())

    words_train, words_test, prons_train, prons_test = train_test_split(words, pronunciations,
                                                                        test_size=0.3)

    X_train, y_train = get_dists(words_train, prons_train)[:2]
    X_test, y_test = get_dists(words_test, prons_test)[:2]

    print('X_letters', X_train.shape)

    classifier = LogisticRegression()
    ovr = OneVsRestClassifier(classifier, n_jobs=-1)
    ovr.fit(X_train, y_train)
    print('score', ovr.score(X_test, y_test))

    # for letter_dist, letter in zip(letter_dists, letters):
    #     pred = neigh.predict([letter_dist])
    #     phoneme_pred = phonemes_lb.inverse_transform(pred)
    #     print(letter, phoneme_pred)


if __name__ == '__main__':
    prons = load_data('pronunciations_en.json')
    find_distributions(prons)

