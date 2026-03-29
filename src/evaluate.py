from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_val, y_val) -> dict:
    """
    Evaluate the trained model on validation data.

    Parameters
    ----------
    model : sklearn estimator
        Trained pipeline/model.
    X_val : pd.DataFrame or array-like
        Validation features.
    y_val : pd.Series or array-like
        Validation target.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_val)

    results = {
        "accuracy": accuracy_score(y_val, y_pred),
        "classification_report": classification_report(y_val, y_pred),
        "confusion_matrix": confusion_matrix(y_val, y_pred),
    }

    return results


def print_evaluation(results: dict) -> None:
    """
    Print evaluation results in a readable format.

    Parameters
    ----------
    results : dict
        Evaluation results dictionary.
    """
    print("\n=== EVALUATION RESULTS ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results["classification_report"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])