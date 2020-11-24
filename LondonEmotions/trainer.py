from LondonEmotions.utils import simple_time_tracker

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

MLFLOW_URI = "https://mlflow.lewagon.co/"

class Trainer():

    # Mlflow parameters identifying the experiment
    ESTIMATOR = "CNN"
    EXPERIMENT_NAME = "LondonEmotions"

    def __init__(self, X, y, **kwargs):
        self.pipeline = None
        self.vectorizer = None
        self.kwargs = kwargs
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)  # cf doc above
        self.mlflow = kwargs.get('mlflow', False)
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", False)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(self.X_train, self.y_train, test_size=0.15)

        self.log_kwargs_params()

    def set_pipeline(self):
        self.pipeline = MultinomialNB()
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, norm='l2', ngram_range=(1, 2))

    @simple_time_tracker
    def train(self):
        self.set_pipeline()

        X_train_vect = self.vectorizer.fit(self.X_train)
        self.pipeline.fit(X_train_vect, self.y_train)
        self.mlflow_log_metric("train_time", int(time.time() - tic))

    def evaluate(self):
        f1_train = self.compute_score(self.X_train, self.y_train)
        self.mlflow_log_metric("f1_train", f1_train)
        if self.split:
            f1_val = self.compute_score(self.X_val, self.y_val, show=True)
            self.mlflow_log_metric("f1_val", f1_val)
            print("f1 train: {} || f1 val: {}".format(f1_train, f1_val))
        else:
            print("f1 train: {}".format(f1_train))

    def compute_score(self, X_test, y_test):
        X_test_vect = self.vectorizer.transform(X_test)
        y_pred = self.pipeline.predict(X_test_vect)

        f1_score = f1_score(self.y_val, y_pred, average='micro') * 100
        return f1_score

    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib """
        joblib.dump(self.pipeline, '../raw_data/model.joblib')
        print("model.joblib saved locally")

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
