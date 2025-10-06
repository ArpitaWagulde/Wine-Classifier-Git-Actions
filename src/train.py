from sklearn.linear_model import LogisticRegression
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "../model/wine_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, _, y_train, _ = split_data(X, y)
    fit_model(X_train, y_train)