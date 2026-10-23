from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AudioDecisionTree:
    def __init__(self, max_depth=10):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return {"max_depth": self.model.max_depth}
