from LondonEmotions.data import clean_data, retrieve_data
from LondonEmotions.trainer import Trainer

default_params = dict(nrows=40000,
                      upload=False,
                      local=True,  # set to False to get data from GCP (Storage or BigQuery)
                      gridsearch=False,
                      optimize=True,
                      estimator="NB",
                      mlflow=True,  # set to True to log params to mlflow
                      experiment_name="LondonEmotions",
                      split=True)

if __name__ == "__main__":
    print("############  Fetching data   ############")
    df = retrieve_data(local=True)
    print("############  Cleaning data   ############")
    df = clean_data(df)
    y_train = df[['Emotion']]
    X_train = df[['tokenized_text']]
    # Train and save model
    t = Trainer(X=X_train, y=y_train, **default_params)
    del X_train, y_train
    print("############  Training model   ############")
    t.train()
    # print("############  Evaluating model   ############")
    # t.evaluate()
    # print("############  Training model   ############")
    # t.save_model()

