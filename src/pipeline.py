import os
import pickle

from sklearn.model_selection import cross_val_score, train_test_split

from src.evaluate import evaluate_model, print_evaluation
from src.features import add_engineered_features
from src.ingest import load_data
from src.train import build_training_pipeline


RANDOM_STATE = 42
TRAIN_PATH = "data/train.csv"
MODEL_OUTPUT_PATH = "models/pipeline.pkl"


def run_pipeline():
    """
    Run the complete machine learning workflow:
    load data -> feature engineering -> split -> build pipeline ->
    cross validation -> train -> evaluate -> save pipeline.
    """
    # =========================
    # 1. Load Data
    # =========================
    print("=== STEP 1: LOADING DATA ===")
    df = load_data(TRAIN_PATH)
    print(f"Dataset shape: {df.shape}")

    # =========================
    # 2. Feature Engineering
    # =========================
    print("\n=== STEP 2: FEATURE ENGINEERING ===")
    df = add_engineered_features(df)
    print("Feature engineering completed.")

    # =========================
    # 3. Split Features and Target
    # =========================
    print("\n=== STEP 3: PREPARING X AND y ===")
    X = df.drop(columns=["Transported"])
    y = df["Transported"].astype(int)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # =========================
    # 4. Train/Validation Split
    # =========================
    print("\n=== STEP 4: TRAIN/VALIDATION SPLIT ===")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")

    # =========================
    # 5. Build Pipeline
    # =========================
    print("\n=== STEP 5: BUILDING PIPELINE ===")
    pipeline = build_training_pipeline(random_state=RANDOM_STATE)
    print("Pipeline successfully built.")

    # =========================
    # 6. Cross Validation
    # =========================
    print("\n=== STEP 6: CROSS VALIDATION ===")
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"Std CV Accuracy : {cv_scores.std():.4f}")

    # =========================
    # 7. Train Model
    # =========================
    print("\n=== STEP 7: TRAINING PIPELINE ===")
    pipeline.fit(X_train, y_train)
    print("Training completed.")

    # =========================
    # 8. Evaluate Model
    # =========================
    print("\n=== STEP 8: EVALUATING MODEL ===")
    results = evaluate_model(pipeline, X_val, y_val)
    print_evaluation(results)

    # =========================
    # 9. Save Pipeline
    # =========================
    print("\n=== STEP 9: SAVING PIPELINE ===")
    os.makedirs("models", exist_ok=True)

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Pipeline saved successfully at: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()