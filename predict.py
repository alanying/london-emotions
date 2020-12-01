from LondonEmotions.params import BUCKET_NAME, REVIEW_PATH
from LondonEmotions.data import clean_data
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.lib.io import file_io
from tensorflow.keras.models import load_model
import pickle

def get_review_data(local=False):
    if local:
        path = 'raw_data/prediction.csv'
    else:
        path = "gs://{}/{}".format(BUCKET_NAME, REVIEW_PATH)

    review_df = pd.read_csv(path)
    review_df.rename(columns = {'review': 'Text'}, inplace=True)

    return review_df

def process_reviews(df, local=True):

    review_df = clean_data(df)

    num_classes = 5
    embed_num_dims = 300
    max_seq_len = 300

    reviews = review_df['tokenized_text']

    sentences_pred = [[_ for _ in sentence] for sentence in reviews]
    texts_pred = [' '.join([x for x in sentence]) for sentence in sentences_pred]

    # Load trained tokenizer
    if local:
        filepath = 'raw_data/tokenizer.pickle'
    else:
        filepath = "gs://{}/{}/{}".format(
                'models',
                'tokenizer',
                'model_tokenizer'
                )
    with file_io.FileIO(filepath, mode='rb') as handle:
        tokenizer = pickle.load(handle)

    sequence_pred = tokenizer.texts_to_sequences(texts_pred)

    index_of_words = tokenizer.word_index

    # vacab size is number of unique words + reserved 0 index for padding
    vocab_size = len(index_of_words) + 1

    # Padding text sentences
    X_pred_pad = pad_sequences(sequence_pred, maxlen = max_seq_len )

    return X_pred_pad

def predict_reviews(local=True):
    # Load reviews
    df = get_review_data(local=local)
    # Preprocess and Tokenize reviews
    padded_reviews = process_reviews(df, local=local)
    # Load model and predict
    if local:
        path = 'raw_data/saved_model_2.pb'
    else:
        path = "gs://{}/{}/{}/{}".format(
            'models',
            MODEL_NAME,
            MODEL_VERSION,
            'saved_model.pb'
            )
    model = load_model(path)
    predictions = model.predict(padded_reviews)
    preds_categorical = []
    for prediction in predictions:
        preds_categorical.append(np.argmax(prediction))

    encoding = {
        0: 'anger',
        1: 'joy',
        2: 'worry',
        3: 'neutral',
        4: 'sad'
    }
    pred_series = pd.Series(preds_categorical)
    review_predictions = pred_series.map(encoding)
    df['emotion'] = review_predictions
    df.to_csv('raw_data/review_predictions.csv')

if __name__ == '__main__':
    predict_reviews(local=False)
