import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

def plot_training_results(history):
    epochs = list(range(12))
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    fig.text(s='Epochs vs. Training and Validation Accuracy/Loss', size=18, fontweight='bold', fontname='monospace', y=1, x=0.28, alpha=0.8)

    # Training and validation accuracy graph
    sns.despine()
    ax[0].plot(epochs, train_acc, marker='o',label='Training Accuracy')
    ax[0].plot(epochs, val_acc, marker='o', label='Validation Accuracy')
    ax[0].legend(frameon=False)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')

    # Training and validation loss graph
    sns.despine()
    ax[1].plot(epochs, train_loss, marker='o', label='Training Loss')
    ax[1].plot(epochs, val_loss, marker='o', label='Validation Loss')
    ax[1].legend(frameon=False)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Training and Validation Loss')

    plt.show(block=True)

def evaluation(test_images, test_labels, model):
    pred_labels = model.predict(test_images)
    pred_labels = np.argmax(pred_labels, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    print(classification_report(true_labels, pred_labels))

    # Uncomment to produce confusion matrix
    # confusion_matric(true_labels, pred_labels)

def confusion_matric(true_labels, pred_labels):
    classes = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    sns.heatmap(confusion_matrix(true_labels, pred_labels), ax=ax, xticklabels=classes, yticklabels=classes, annot=True, alpha=0.7, linewidths=2, linecolor='black')
    fig.text(s='Heatmap of the Confusion Matrix', size=18, fontweight='bold',fontname='monospace', color='black', y=0.92, x=0.28, alpha=0.8)

    plt.show()