import joblib
from src.data import load_data, split_data
from sklearn.metrics import classification_report

class Metrics:
    def __init__(self):
        X, y = load_data()
        _, X_test, _, y_test = split_data(X, y)
        model = joblib.load("model/wine_model.joblib")
        y_pred = model.predict(X_test)
        self.metrics = classification_report(y_test, y_pred, output_dict=True)

    def get_model_accuracy(self):
        """
        Calculate the accuracy of the trained model on the test data.
        Returns:
            accuracy (float): Accuracy of the model on the test set.
        """
        return self.metrics.get('accuracy')
    
    def get_f1_score(self):
        """
        Calculate the 'weighted avg' F1 score of the trained model on the test data.
        Returns:
            f1_score (float): F1 score of the model on the test set.
        """
        return self.metrics.get('weighted avg').get('f1-score')
    
    def get_precision(self):
        """
        Calculate the 'weighted avg' precision of the trained model on the test data.
        Returns:
            precision (float): Precision of the model on the test set.
        """
        return self.metrics.get('weighted avg').get('precision')

    def get_recall(self):
        """
        Calculate the 'weighted avg' recall of the trained model on the test data.
        Returns:
            recall (float): Recall of the model on the test set.
        """
        return self.metrics.get('weighted avg').get('recall')