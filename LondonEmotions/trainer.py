#!/usr/bin/env python

from LondonEmotions.utils import simple_time_tracker, instantiate_model, create_embedding_matrix

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle

from tensorflow.keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.utils import simple_preprocess
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import models
from tensorflow.python.lib.io import file_io

from google.cloud import storage
from LondonEmotions.params import MODEL_NAME, MODEL_VERSION, BUCKET_NAME, \
    BUCKET_TRAIN_DATA_PATH, WORD2VEC_PATH

MLFLOW_URI = "https://mlflow.lewagon.co/"

class Trainer():

    # Mlflow parameters identifying the experiment
    ESTIMATOR = "CNN"
    EXPERIMENT_NAME = "LondonEmotions"

    def __init__(self, X, y, **kwargs):
        self.pipeline = None
        self.kwargs = kwargs
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)  # cf doc above
        self.mlflow = kwargs.get('mlflow', False)
        self.local = kwargs.get('local', True)
        self.X_df = X
        self.y_df = y
        del X, y
        self.split = self.kwargs.get("split", False)  # cf doc above
        self.X_test_pad = None
        self.y_test_cat = None
        if self.split:
            self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X_df, self.y_df, test_size=0.15)

        self.log_kwargs_params()

    @simple_time_tracker
    def train(self):

        num_classes = 5
        embed_num_dims = 300
        max_seq_len = 300
        class_names = ['joy', 'worry', 'anger', 'sad', 'neutral']

        sentences_train = [[_ for _ in sentence] for sentence in self.X_train]
        sentences_test = [[_ for _ in sentence] for sentence in self.X_test]

        texts_train = [' '.join([x for x in sentence]) for sentence in sentences_train]
        texts_test = [' '.join([x for x in sentence]) for sentence in sentences_test]

        # Train tokenizer on training data (convert to integers)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts_train)

        # Save tokenizer
        if self.local:
            filepath = 'raw_data/tokenizer.pickle'
        else:
            filepath = 'tokenizer/tokenizer.pickle'
        with file_io.FileIO(filepath, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with file_io.FileIO(filepath, mode='r') as f:
            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)
            storage_location = '{}/{}/{}'.format(
                'models',
                'tokenizer',
                'model_tokenizer'
                )
            blob = bucket.blob(storage_location)
            blob.upload_from_filename(filename=filepath)
            print("tokenizer saved on GCP")

        # Convert texts to itegers
        sequence_train = tokenizer.texts_to_sequences(texts_train)
        sequence_test = tokenizer.texts_to_sequences(texts_test)

        index_of_words = tokenizer.word_index

        # vocab size is number of unique words + reserved 0 index for padding
        vocab_size = len(index_of_words) + 1

        # Padding text sentences
        X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len )
        self.X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len )

        # Encode target
        encoding = {
            'anger': 0,
            'joy': 1,
            'worry': 2,
            'neutral': 3,
            'sad': 4
        }

        y_train_enc = [encoding[x] for x in self.y_train]
        y_test_enc = [encoding[x] for x in self.y_test]

        y_train_cat = to_categorical(y_train_enc)
        self.y_test_cat = to_categorical(y_test_enc)

        # Create embedding matrix
        if self.local:
            file_path = 'embeddings/wiki-news-300d-1M.vec'
        else:
            file_path = "gs://{}/{}".format(BUCKET_NAME, WORD2VEC_PATH)

        embedd_matrix = create_embedding_matrix(file_path, index_of_words, embed_num_dims)
        embedd_matrix.shape

        # Train model
        batch_size = 256
        epochs = 4
        es = EarlyStopping(patience=1, restore_best_weights=True)

        model = instantiate_model(embedd_matrix, max_seq_len, vocab_size, embed_num_dims)

        hist = model.fit(X_train_pad, y_train_cat,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_split=0.3,
                         callbacks=[es])

        self.pipeline = model


    def evaluate(self):
        if self.split:
            f1_val = self.compute_score(self.X_test_pad, self.y_test_cat)
            self.mlflow_log_metric("f1_val", f1_val)
            print("f1 val: {}".format(f1_val))

    def compute_score(self, X_test, y_test):
        predictions = self.pipeline.predict(X_test, batch_size=32, verbose=1)
        preds_categorical = []
        for prediction in predictions:
            preds_categorical.append(np.argmax(prediction))
        preds_categorical = to_categorical(preds_categorical)

        test_score = f1_score(y_test, preds_categorical, average='weighted')
        return test_score

    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib """
        # Save to folder nlp_model on cloud machine
        self.pipeline.save('nlp_model')
        print("model saved locally")

        if upload:
            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)
            storage_location = '{}/{}/{}/{}'.format(
                'models',
                MODEL_NAME,
                MODEL_VERSION,
                'saved_model.pb')
            blob = bucket.blob(storage_location)
            blob.upload_from_filename(filename='nlp_model/saved_model.pb')
            print("model saved on GCP")

    ### MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

