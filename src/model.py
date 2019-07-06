"""Model factory. Returns an unfitted estimator given a model name and params."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_model(name, params):
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(**params)
    if name == "random_forest":
        return RandomForestClassifier(**params)
    if name == "xgboost":
        # imported lazily so the module loads even without xgboost installed
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    raise ValueError("unknown model: %s" % name)
