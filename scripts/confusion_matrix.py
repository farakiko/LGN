import numpy as np
import matplotlib.pyplot as plt
import torch

# use this while training
# generate one confusion matrix per epoch
# append this result to a list
def generate_confusion_matrix(preds, labels, num_classes=2):
    predictions_list = torch.clone(preds).tolist()
    labels_list = torch.clone(labels).tolist()
    confusion_matrix = np.zeros((num_classes, num_classes)) # 2 class
    for i in range(len(preds)):
        confusion_matrix[predictions_list[i]][labels_list[i]] += 1
        # 0: false 1: true
    return confusion_matrix

# normalization
def normalize_confusion_matrices(confusion_matrices):
    confusion_matrices_normalized = []
    for epoch in range(len(confusion_matrices)):
        confusion_matrix = np.array(confusion_matrices[epoch], dtype=float)
        for label in range(len(confusion_matrix)):
            row = confusion_matrix[label]
            row = row / row.sum() if row.sum() !=0 else 0
            confusion_matrix[label] = row
        confusion_matrices_normalized.append(confusion_matrix.tolist())
    return confusion_matrices_normalized

# e.g. labels = ["background", "signal"]
def plot_confusion_matrix(confusion_matrices, labels=["background", "signal"], epoch=-1, normalize=True, save=True, savepath=None, format=None):

    if normalize:
        confusion_matrices = normalize_confusion_matrices(confusion_matrices)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if normalize:
        cax = ax.matshow(confusion_matrices[epoch], interpolation='nearest',cmap='viridis', vmin=0, vmax=1)
    else:
        cax = ax.matshow(confusion_matrices[epoch], interpolation='nearest',cmap='viridis')
    fig.colorbar(cax)

    ax.set_xticklabels(['']+labels)
    ax.set_yticklabels(['']+labels)

    for (i, j), z in np.ndenumerate(confusion_matrices[epoch]):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    ax.set_title("Confusion Matrix at Epoch 1")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("Actual Labels")

    fig.tight_layout()

    if save:
        plt.savefig(savepath+"/confusion_matrix_at_Epoch_{}.".format(epoch) +format, bbox_inches="tight")
    plt.show()
