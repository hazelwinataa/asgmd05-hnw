from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler


def to_string(X):
    """
    Convert all values in the input array to string.
    Needed to avoid mixed types (bool + str) for OrdinalEncoder.
    """
    return X.astype(str)


def get_feature_columns():
    """
    Return the lists of categorical and numerical feature columns.

    Returns
    -------
    tuple[list[str], list[str]]
        Categorical and numerical feature names.
    """
    categorical_features = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
        "AgeGroup",
    ]

    numerical_features = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "CabinNum",
        "PassengerNumber",
        "GroupSize",
        "IsSolo",
        "FamilySize",
        "TotalSpending",
        "HasSpending",
        "AgeMissing",
        "CryoSleepMissing",
        "VIPMissing",
    ]

    return categorical_features, numerical_features


def build_preprocessor():
    """
    Build a ColumnTransformer for preprocessing categorical and numerical features.

    Returns
    -------
    ColumnTransformer
        Configured preprocessor.
    """
    categorical_features, numerical_features = get_feature_columns()

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("to_string", FunctionTransformer(to_string, validate=False)),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
            ),
        ]
    )

    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numerical_pipeline, numerical_features),
        ],
        remainder="drop"
    )

    return preprocessor