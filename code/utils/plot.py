

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(labels_list, predictions_list): 
    cm = confusion_matrix(labels_list, predictions_list)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')