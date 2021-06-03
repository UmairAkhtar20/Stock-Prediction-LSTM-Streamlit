import tkinter
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

plt.style.use('ggplot')
matplotlib.use('TkAgg')


def show(data, training_data_len, predictions):
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = 0
    valid['Predictions'] = predictions
    # showing the data
    plt.figure(figsize=(8, 4))
    plt.title('Model')
    plt.xlabel('data', fontsize=8)
    plt.ylabel('closed price $', fontsize=8)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Test', 'Predictions'], loc='upper left')
    plt.show()


def conf(y_test, predictions):
    data = {'y_Actual': [y_test],
            'y_Predicted': [predictions]
            }

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    cutoff = 0.7  # decide on a cutoff limit
    y_pred_classes = np.zeros_like(predictions)  # initialise a matrix full with zeros
    y_pred_classes[predictions > cutoff] = 1
    y_test_classes = np.zeros_like(predictions)
    y_test_classes[y_test > cutoff] = 1
    cm = confusion_matrix(y_test_classes, y_pred_classes)

    # cm= pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(cm, annot=True)
    plt.show()
