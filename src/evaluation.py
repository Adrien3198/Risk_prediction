from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

"""
evaluation.py
============================
Module to get evaluation of model predictions
"""

def get_metrics(y, y_pred):
      """
      Return the a tuple of accuracy, precision, recall and f1 scores with the number of suport 
      for each class predicted from a model

      Parameters
      ----------
      y
            true example labels
      y_pred
            labels predicted by the model
      """
      accuracy = accuracy_score(y, y_pred)
      precision, recall, fscore, support = precision_recall_fscore_support(
            y, y_pred)
      print("Evaluation :")
      print(classification_report(y, y_pred))
      return accuracy, precision, recall, fscore, support