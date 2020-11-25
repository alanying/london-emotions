#!/usr/bin/env python

from LondonEmotions.utils import simple_time_tracker, instantiate_model

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from tensorflow.keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.utils import simple_preprocess
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import models

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
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(self.X_df, self.y_df, test_size=0.15)

        self.log_kwargs_params()

    @simple_time_tracker
    def train(self):

        print(list(self.X_train.columns))

        # Calculate size of train data
        all_training_words = [word for tokens in self.X_train['tokenized_text'] for word in tokens]
        training_sentence_lengths = [len(tokens) for tokens in self.X_train['tokenized_text']]
        training_vocab = sorted(list(set(all_training_words)))
        # Calculate size of test data
        all_test_words = [word for tokens in self.X_val['tokenized_text'] for word in tokens]
        test_sentence_lengths = [len(tokens) for tokens in self.X_val['tokenized_text']]
        test_vocab = sorted(list(set(all_test_words)))
        # Load pretrained word2vec
        if self.local:
            word2vec_path = 'raw_data/google-vectors.bin.gz'
        else:
            word2vec_path = "gs://{}/{}".format(BUCKET_NAME, WORD2VEC_PATH)
        word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

        MAX_SEQUENCE_LENGTH = 201
        EMBEDDING_DIM = 300

        # Train Tokenization
        tokenizer = Tokenizer(num_words=len(training_vocab), lower=True, char_level=False)
        tokenizer.fit_on_texts(self.X_train['tokenized_text'].tolist())
        training_sequences = tokenizer.texts_to_sequences(self.X_train['tokenized_text'].tolist())
        train_word_index = tokenizer.word_index
        train_cnn_data = pad_sequences(training_sequences,
                                       maxlen=MAX_SEQUENCE_LENGTH)

        # Test Tokenization
        train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
        for word,index in train_word_index.items():
            train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
        test_sequences = tokenizer.texts_to_sequences(self.X_val['tokenized_text'].tolist())
        test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

        # instantiate model
        model = instantiate_model(train_embedding_weights,
                                MAX_SEQUENCE_LENGTH,
                                len(train_word_index)+1,
                                EMBEDDING_DIM,
                                5)

        # Prepare mapping for the sentiment
        sentiment_coding = {'anger': 0 , 'joy': 4, 'worry': 2, 'sad': 1 , 'neutral': 3}

        # apply mapping
        y_train_coded = self.y_train['Emotion'].map(sentiment_coding)
        y_test_coded= self.y_val['Emotion'].map(sentiment_coding)

        # Transform the numbers to categories
        y_train_cat = to_categorical(y_train_coded)
        y_test_cat = to_categorical(y_test_coded)

        num_epochs = 10
        batch_size = 32
        X_train_model = train_cnn_data
        y_train_model = y_train_cat

        print('####### Training model NOW #######')

        es = EarlyStopping(patience=10, restore_best_weights=True)
        history = model.fit(X_train_model, y_train_model,
                    validation_split=0.3,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1
                   )

    def evaluate(self):
        self.X_val = self.vectorizer.transform(self.X_val)
        f1_train = self.compute_score(self.X_train, self.y_train)
        self.mlflow_log_metric("f1_train", f1_train)

        if self.split:
            f1_val = self.compute_score(self.X_val, self.y_val, show=True)
            self.mlflow_log_metric("f1_val", f1_val)
            print("f1 train: {} || f1 val: {}".format(f1_train, f1_val))
        else:
            print("f1 train: {}".format(f1_train))

    def compute_score(self, X_test, y_test):
        y_pred = self.pipeline.predict(self.X_val)
        f1_score = f1_score(self.y_val, y_pred, average='micro') * 100
        return f1_score

    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib """
        joblib.dump(self.pipeline, '../raw_data/model.joblib')
        print("model.joblib saved locally")

        client = storage.Client().bucket(BUCKET_NAME)
        storage_location = '{}/{}/{}/{}'.format(
            'models',
            MODEL_NAME,
            MODEL_VERSION,
            'model.joblib')
        blob = client.blob(storage_location)
        blob.upload_from_filename(filename='../raw_data/model.joblib')
        print("model.joblib saved on GCP")

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

