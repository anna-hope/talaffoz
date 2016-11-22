from abc import ABCMeta, abstractmethod
import json
import logging
from itertools import chain
from pathlib import Path
from typing import Iterable, Hashable, Dict, Union, List

import numpy as np


class ModelWrapper(metaclass=ABCMeta):

    @abstractmethod
    def predict(self, input):
        pass



def load_data(file_path: Union[Path, str],
              only_alpha=True) -> Dict[str, List[str]]:
    logging.info('loading the data file {}'.format(file_path))
    file_path = Path(file_path)
    with file_path.open() as f:
        seq2seq = json.load(f)

    # words_ipa = {word: v['ipa']
    #              for word, v in pronunciations.items()}
    return seq2seq



def make_sequence_ids(seqs: Iterable[Iterable[Hashable]]) -> Dict[Hashable, int]:
    """
    Maps sequence elements to ids based on sort order.
    Sequence elements must be hashable.
    :param seqs: Iterable
        An iterable of hashable elements.
    :return: dict
        A dictionary of sequence elements map to their ids.
    """
    elements = sorted(set(chain.from_iterable(seqs)))
    element_ids = {element: n for n, element in enumerate(elements)}
    return element_ids


def ids_to_sequence(seq_as_ids, element_ids):
    """
    Maps sequence represented by its one-hot ids back into a sequence of elements.
    :param seq_as_ids: array like
        2d array of one-hot ids.
    :param element_ids: dict
        A dictionary mapping each element onto an id.
    :return: list
        A list of lists of original elements according to element_ids.
    """
    ids_to_elements = {n: element for element, n in element_ids.items()}
    sequence = []
    for one_hot in seq_as_ids:
        element_id = np.argmax(one_hot)
        element = ids_to_elements[element_id]
        sequence.append(element)
    return sequence


def get_max_length(seqs) -> int:
    max_length = len(max(seqs, key=len))
    return max_length


