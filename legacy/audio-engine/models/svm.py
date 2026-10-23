from sklearn.svm import SVC
import numpy as np

class AudioSVM:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return {"kernel": self.model.kernel, "C": self.model.C}
