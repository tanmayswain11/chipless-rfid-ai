
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_model_dictionary():
    """
    Create the common regression model collection.

    The same model definitions are reused by normal training
    and experimental RH training.
    """

    models = {

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=-1
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            loss="huber",
            random_state=42
        ),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=2,
            random_state=42
        ),

        "SVR RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", SVR(
                kernel="rbf",
                C=10.0,
                epsilon=0.1,
                gamma="scale"
            ))
        ]),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0))
        ]),
    }

    return models


def get_models(n_samples):
    """
    Models for normal supervised regression.

    Temperature and water-volume datasets must have
    at least four independent labelled curves.
    """

    if n_samples < 4:
        raise ValueError(
            "Normal model evaluation requires at least "
            "4 independent labelled curves."
        )

    return _build_model_dictionary()


def get_rh_models(n_samples):
    """
    Separate model factory for experimental RH training.

    Allows fitting with three independent labelled curves.

    IMPORTANT:
    Three samples are not sufficient to establish reliable
    generalization performance. This function is intended
    for experimental model fitting only.
    """

    if n_samples < 3:
        raise ValueError(
            "Experimental RH training requires at least "
            "3 independent labelled curves."
        )

    return _build_model_dictionary()


def get_model_names():
    """
    Return all available model names.
    """

    return list(_build_model_dictionary().keys())
