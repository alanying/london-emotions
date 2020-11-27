from LondonEmotions.data import clean_data, retrieve_data
from LondonEmotions.trainer import Trainer

local = True

default_params = dict(nrows=40000,
                      upload=False,
                      local=local,  # set to False to get data from GCP (Storage or BigQuery)
                      gridsearch=False,
                      optimize=True,
                      estimator="CNN",
                      mlflow=True,  # set to True to log params to mlflow
                      experiment_name="LondonEmotions",
                      split=True)

if __name__ == "__main__":
    print("############  Fetching data   ############")
    df = retrieve_data(local=local)
    print("############  Cleaning data   ############")
    df = clean_data(df)
    X = df['tokenized_text']
    y = df['Emotion']
    # Train and save model
    t = Trainer(X=X, y=y, **default_params)
    del X, y
    print("############  Training model   ############")
    t.train()
    print("############  Evaluating model   ############")
    t.evaluate()
    print("############  Saving model   ############")
    t.save_model()

