# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Main lib for london-emotions Project
"""

from os.path import split
import pandas as pd
import datetime

import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

pd.set_option('display.width', 200)


def clean_data(data):
    """
    clean and preprocess data
    """

    # Lowercase text
    data['clean_text'] = data['text'].apply(
        lambda x: x.lower()
        )

    # Remove numbers
    data['clean_text'] = data['clean_text'].apply(
        lambda x: ''.join(let for let in x if not let.isdigit())
        )

    # Remove punctuation
    data['clean_text'] = data['clean_text'].apply(
        lambda x: ''.join(let for let in x if not let in string.punctuation)
        )

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    data['clean_text'] = data['clean_text'].apply(
        lambda x: word_tokenize(x)
        )

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    data['clean_text'] = data['clean_text'].apply(
        lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x)
        )


    # Return data
    return data


if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    import london-emotions
    folder_source, _ = split(london-emotions.__file__)
    df = pd.read_csv('{}/data/data.csv.gz'.format(folder_source))
    clean_data = clean_data(df)
    print(' dataframe cleaned')
