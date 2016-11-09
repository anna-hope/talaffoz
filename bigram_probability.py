from collections import Counter, defaultdict
import json
from pprint import pprint

import nltk

def get_words():
    # data_path = 'pronunciations.json'
    # with open(data_path) as f:
    #     data = json.load(f)
    words = nltk.corpus.brown.words()
    normalised_words = (w.casefold() for w in words
                        if all(c.isalpha() for c in w))
    brown_words = Counter(normalised_words)
    #pprint(brown_words.most_common(200))
    return [word for word, _ in brown_words.most_common(1000)]



def get_probs(words):
    co_counts = defaultdict(Counter)
    for word in words:
        bigrams = nltk.ngrams(word, 2)
        for first, second in bigrams:
            co_counts[first][second] += 1

    co_probs = {}
    for letter, co_letter_counts in co_counts.items():
        total_letter_counts = sum(co_letter_counts.values())
        co_letter_probs = {letter: count / total_letter_counts
                           for letter, count in co_letter_counts.items()}

        co_probs[letter] = sorted(co_letter_probs.items(), key=lambda x: x[1],
                                  reverse=True)[:20]


    return co_probs



if __name__ == '__main__':
    print('working...')
    words = get_words()
    co_probs = get_probs(words)
    with open('co_probs.json', 'w') as f:
        json.dump(co_probs, f, sort_keys=True, indent='    ')



