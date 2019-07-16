"""Model factory. Returns an unfitted estimator given a model name and params.

LogReg is wrapped in a Pipeline with StandardScaler since it is sensitive
to feature scale. Tree-based models are returned bare.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(name, params):
    name = name.lower()
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(**params)
    if name == "xgboost":
        # imported lazily so the module loads even without xgboost installed
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    raise ValueError("unknown model: %s" % name)
