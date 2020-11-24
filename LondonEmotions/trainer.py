from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class Trainer(object):
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(self.X_train, self.y_train, test_size=0.15)

    def train_predict(self):
        vect = TfidfVectorizer(tokenizer=preprocess_and_tokenize, sublinear_tf=True, norm='l2', ngram_range=(1, 2))
        X_train_vect = vect.fit_transform(self.X_train)
        X_test_vect = vect.transform(self.X_val)

        nb = MultinomialNB()
        nb.fit(X_train_vect, self.y_train)

        ynb_pred = nb.predict(X_test_vect)

        print("Accuracy: {:.2f}%".format(accuracy_score(y_test, ynb_pred) * 100))
        print("\nF1 Score: {:.2f}".format(f1_score(y_test, ynb_pred, average='micro') * 100))
        print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, ynb_pred))

