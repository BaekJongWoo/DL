from data.emo.emo_utils import EmoDataloader, label_to_emoji

from mine.model import Sequantial, ModelBase
from mine.layer import RNN, Linear, LSTM
from mine.dataloader import dataLoader
from mine.loss import CrossEntropyLoss, softmax
from mine.optimizer import SGDOptimizer, AdamOptimizer
from mine.util_layer import Dropout

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def MyRNN(optimizer):
    model = Sequantial([
        RNN(50, 128, to_many=True),
        RNN(128, 128, to_many=False),
        Linear(128, 5),
    ],
    optimizer)
    return model

def MyLSTM(optimizer):
    model = Sequantial([
        LSTM(50, 128, to_many=True),
        # Dropout(0.2),
        LSTM(128, 128, to_many=False),
        # Dropout(0.2),
        Linear(128, 5)
    ], optimizer)
    return model

def PrintCM(model: ModelBase, test_dataloader: dataLoader):
    y_true = []
    y_pred = []
    
    print("\nGenerating Confusion Matrix...")
    for test_data in tqdm(test_dataloader):
        tx, ty = test_data
       
        ty_pred = model.forward(tx)

        y_true.extend(np.argmax(ty, axis=1))
        y_pred.extend(np.argmax(ty_pred, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"confusion_matrix.png")
    plt.close()

    print("Complete.")

def PrintLosses(train_loss_values):
    term = 50

    loss_values = np.array(train_loss_values)
    n = len(loss_values) // term
    split_values = np.split(loss_values, n)
    mean_values = np.mean(split_values, axis=1)
    x_values = np.arange(0, len(loss_values), term)

    plt.figure()
    plt.plot(x_values, mean_values, label="Train Loss", alpha=0.6, color='b')
    plt.title("Loss over Iteration")
    plt.savefig(f"loss_graph.png")
    plt.close()

def train(model: ModelBase, train_dataloader: dataLoader, epochs: int):
    loss_fn = CrossEntropyLoss()
    losses = []
    for epoch in tqdm(range(epochs)):
        for data in train_dataloader:
            x, y = data

            y_pred = model.forward(x, is_train=True)
            loss, grad = loss_fn(y, y_pred)
            model.backward(grad)

            losses.append(loss)
    PrintLosses(losses)

def test(model: ModelBase, test_dataloader: dataLoader):
    total_num = 0
    correct_num = 0
    for data in tqdm(test_dataloader):
        x, y = data

        y_pred = model.forward(x, is_train=False)
        y_pred_probs = softmax(y_pred)
    
        for y_one_hot, y_pred_prob in zip(y, y_pred_probs):
            pred_label = np.argmax(y_pred_prob)
            print(label_to_emoji(pred_label), end='')
            if np.argmax(y_one_hot) == np.argmax(y_pred_prob):
                correct_num += 1
            total_num += 1
    print()
    print(f"Accuracy: {correct_num / total_num * 100 :.2f}%")

if __name__ == "__main__":
    optimizer = SGDOptimizer(learning_rate=0.05)
    model = MyLSTM(optimizer)
    train_dataloader = EmoDataloader(is_train=True, glove_size='50d')
    test_dataloader = EmoDataloader(is_train=False, glove_size='500d')

    train(model, train_dataloader, epochs=300)
    test(model, test_dataloader)
    PrintCM(model, test_dataloader)