#vlastne transformatory pre pipeline z 2.3 B - samostatny modul, aby sa ulozeny model dal nacitat aj mimo notebooku
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PhysicalRangeToNaN(BaseEstimator, TransformerMixin):
    #krok cistenia z 1.2 B ako sucast pipeline: hodnoty mimo fyzikalneho rozsahu (chyba senzora) -> NaN,
    #ktore potom doplni imputer; pri filterDp len zaporne hodnoty (vysoke su realny signal poruchy, 1.2 B)
    #pravidla su fixne z domeny, nic sa neuci z dat - fit len zapamata nazvy stlpcov
    def __init__(self, ranges = None, lower_bound_only = ()):
        self.ranges = ranges
        self.lower_bound_only = lower_bound_only

    def fit(self, X, y = None):
        self.feature_names_in_ = np.asarray(X.columns, dtype = object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns = self.feature_names_in_).astype(float)
        for col in X.columns:
            if col not in (self.ranges or {}):
                continue
            lo, hi = self.ranges[col]
            invalid = (X[col] < lo) if col in self.lower_bound_only else ~X[col].between(lo, hi)
            X.loc[invalid & X[col].notna(), col] = np.nan
        return X

    def get_feature_names_out(self, input_features = None):
        return self.feature_names_in_
