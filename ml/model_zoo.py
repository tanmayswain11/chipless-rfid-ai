
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR



def get_models():
    """Return seven regression models for chipless RFID sensing."""

    # PCA must use fewer components than the smallest CV training fold.
    # With 5 water-volume samples, LOOCV trains on only 4 samples.
    pca_components = 2

    models = {
        "Polynomial Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components)),
            ("poly", PolynomialFeatures(
                degree=2,
                include_bias=False,
            )),
            ("poly_scaler", StandardScaler()),
            ("ridge", Ridge(alpha=100.0)),
        ]),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            loss="huber",
            random_state=42,
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "SVR RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components)),
            ("svr", SVR(
                kernel="rbf",
                C=10.0,
                epsilon=0.1,
                gamma="scale",
            )),
        ]),

        "SVR Polynomial": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components)),
            ("svr", SVR(
                kernel="poly",
                degree=2,
                C=1.0,
                epsilon=0.1,
                gamma="scale",
                coef0=1.0,
            )),
        ]),
    }

    return models

