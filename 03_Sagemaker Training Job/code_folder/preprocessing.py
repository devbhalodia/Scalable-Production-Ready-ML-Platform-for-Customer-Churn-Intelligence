# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin

class TextStandardizer(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()
    # Apply string operations to each column
    for col in X.columns:
        X[col] = X[col].astype(str).str.strip().str.lower()
    return X


class CategoricalCaster(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.astype("category")
        return X