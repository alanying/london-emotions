

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = "emotions"
MODEL_VERSION = "v2"

# gcp project
PROJECT_ID = "london-emotions"

# gcp Storage
BUCKET_NAME = "wagon-ml-london-emotions"

# gcp location of training dataset
BUCKET_TRAIN_DATA_PATH = "data/emotion_data.csv"
WORD2VEC_PATH = "data/wiki-news-300d-1M.vec"
REVIEW_PATH = 'data/prediction.csv'

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = "trainings"

# Machine configuration
REGION="europe-west1"
PYTHON_VERSION="3.7"
FRAMEWORK="scikit-learn"
RUNTIME_VERSION="1.15"

# package params
PACKAGE_NAME="LondonEmotions"
FILENAME="trainer"

# job
JOB_NAME = "emotion_training_pipeline_$(shell date + '%Y%m%d_%H%M%S')"

