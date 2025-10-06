import joblib
from src.data import load_data, split_data
from sklearn.metrics import accuracy_score

def get_model_accuracy():
    """
    Calculate the accuracy of the trained model on the test data.
    Returns:
        accuracy (float): Accuracy of the model on the test set.
    """
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    model = joblib.load("model/wine_model.pkl")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy