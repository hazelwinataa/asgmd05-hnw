from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor


def build_model(random_state: int = 42) -> LogisticRegression:
    """
    Build the Logistic Regression model.

    Parameters
    ----------
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    LogisticRegression
        Configured Logistic Regression model.
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=2000,
        C=1.0,
        solver="lbfgs"
    )
    return model


def build_training_pipeline(random_state: int = 42) -> Pipeline:
    """
    Build the full sklearn pipeline:
    preprocessing + Logistic Regression.

    Parameters
    ----------
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    Pipeline
        Full training pipeline.
    """
    preprocessor = build_preprocessor()
    model = build_model(random_state=random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline