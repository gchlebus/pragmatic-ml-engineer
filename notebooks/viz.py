import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
from mnist import label_to_name
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def display_label_distribution(labels):
    df = pd.DataFrame({'label': [label_to_name(label) for label in labels]})
    return ggplot(df, aes(x='factor(label)')) + geom_bar() + theme(figure_size=(10, 5))


def display_confusion_matrix(y_true, y_pred, figsize=(10, 10)):
    cm = confusion_matrix([label_to_name(i) for i in y_true], [label_to_name(i) for i in y_pred])
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    # set x and y tick labels
    plt.xticks(np.arange(10) + 0.5, [label_to_name(i) for i in range(10)])
    plt.yticks(np.arange(10) + 0.5, [label_to_name(i) for i in range(10)])

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def display_images(images, y_true, y_pred=None, nrows=5, ncols=6, figsize=(10, 10), 
                   random=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if random:
            idx = np.random.choice(len(images))
        else:
            idx = i
        ax.imshow(images[idx].reshape(28, 28), cmap='gray')
        title = f"True: {label_to_name(y_true[idx])}"
        if y_pred is not None:
            title += f"\nPred: {label_to_name(y_pred[idx])}"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
