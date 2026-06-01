import matplotlib.pyplot as plt
from xgboost import plot_tree

def print_lstm_model(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Plot für den Loss
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 2. Plot für die Accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy', color='green')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout() 
    plt.show()

def print_xgboost_model(xgb_classifier):
    plt.figure(figsize=(30, 20))  
    plot_tree(xgb_classifier, num_trees=1) 
    plt.show()