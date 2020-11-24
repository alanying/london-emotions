from LondonEmotions.data import clean_data, retrieve_data
from LondonEmotions.trainer import Trainer

default_params = dict(nrows=40000,
                      upload=False,
                      local=True,  # set to False to get data from GCP (Storage or BigQuery)
                      gridsearch=False,
                      optimize=True,
                      estimator="NB",
                      mlflow=False,  # set to True to log params to mlflow
                      experiment_name="EmotionModel",
                      split=False)

if __name__ == "__main__":
    print("############  Fetching data   ############")
    df = retrieve_data()
    print("############  Cleaning data   ############")
    df = clean_data(df)
    X_train = df['Emotion']
    y_train = df['Text']
    t = Trainer(X=X_train, y=y_train)
    del X_train, y_train
    print("############  Training model   ############")
    t.train_predict()