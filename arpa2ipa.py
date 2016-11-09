arpa_table = {
    'AO': 'ɔ',
    'AO0': 'ɔ',
    'AO1': 'ɔ',
    'AO2': 'ɔ',
    'AA': 'ɑ',
    'AA0': 'ɑ',
    'AA1': 'ɑ',
    'AA2': 'ɑ',
    'IY': 'i',
    'IY0': 'i',
    'IY1': 'i',
    'IY2': 'i',
    'UW': 'u',
    'UW0': 'u',
    'UW1': 'u',
    'UW2': 'u',
    'EH': 'e',  # modern versions use 'e' instead of 'ɛ'
    'EH0': 'e',  # ɛ
    'EH1': 'e',  # ɛ
    'EH2': 'e',  # ɛ
    'IH': 'ɪ',
    'IH0': 'ɪ',
    'IH1': 'ɪ',
    'IH2': 'ɪ',
    'UH': 'ʊ',
    'UH0': 'ʊ',
    'UH1': 'ʊ',
    'UH2': 'ʊ',
    'AH': 'ʌ',
    'AH0': 'ə',
    'AH1': 'ʌ',
    'AH2': 'ʌ',
    'AE': 'æ',
    'AE0': 'æ',
    'AE1': 'æ',
    'AE2': 'æ',
    'AX': 'ə',
    'AX0': 'ə',
    'AX1': 'ə',
    'AX2': 'ə',
    'EY': 'eɪ',
    'EY0': 'eɪ',
    'EY1': 'eɪ',
    'EY2': 'eɪ',
    'AY': 'aɪ',
    'AY0': 'aɪ',
    'AY1': 'aɪ',
    'AY2': 'aɪ',
    'OW': 'oʊ',
    'OW0': 'oʊ',
    'OW1': 'oʊ',
    'OW2': 'oʊ',
    'AW': 'aʊ',
    'AW0': 'aʊ',
    'AW1': 'aʊ',
    'AW2': 'aʊ',
    'OY': 'ɔɪ',
    'OY0': 'ɔɪ',
    'OY1': 'ɔɪ',
    'OY2': 'ɔɪ',

    # consonants

    'P': 'p',
    'B': 'b',
    'T': 't',
    'D': 'd',
    'K': 'k',
    'G': 'g',

    # diphthongs

    'CH': 'tʃ',
    'JH': 'dʒ',

    # fricatives

    'F': 'f',
    'V': 'v',
    'TH': 'θ',
    'DH': 'ð',
    'S': 's',
    'Z': 'z',
    'SH': 'ʃ',
    'ZH': 'ʒ',
    'HH': 'h',

    # nasals

    'M': 'm',
    'N': 'n',
    'NG': 'ŋ',

    # liquids

    'L': 'l',
    'R': 'r',

    # r-coloured vowels

    'ER': 'ɝ',
    'ER0': 'ɝ',
    'ER1': 'ɝ',
    'ER2': 'ɝ',
    'AXR': 'ɚ',
    'AXR0': 'ɚ',
    'AXR1': 'ɚ',
    'AXR2': 'ɚ',

    # semivowels

    'W': 'w',
    'Y': 'j'
}


def arpa2ipa(arpa):
    ipa = [arpa_table[c] for c in arpa]
    return ipa
