from data.emo.emo_utils import EmoDataloader

from mine.model import Sequantial, ModelBase
from mine.layer import RNN, Linear, LSTM
from mine.dataloader import dataLoader
from mine.loss import CrossEntropyLoss, softmax
from mine.optimizer import SGDOptimizer, AdamOptimizer
from mine.activation import ReLU

from tqdm import tqdm
import numpy as np

def MyRNN(optimizer):
    model = Sequantial([
        RNN(50, 100),
        Linear(100, 50),
        ReLU(),
        Linear(50, 5)
    ],
    optimizer)
    return model

def MyLSTM(optimizer):
    model = Sequantial([
        LSTM(50, 100),
        Linear(100, 50),
        ReLU(),
        Linear(50,5)
    ], optimizer)
    return model

def train(model: ModelBase, train_dataloader: dataLoader, epochs: int):
    loss_fn = CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        for data in train_dataloader:
            x, y = data

            y_pred = model.forward(x)
            loss, grad = loss_fn(y, y_pred)
            model.backward(grad)

def test(model: ModelBase, test_dataloader: dataLoader):
    total_num = 0
    correct_num = 0
    for data in tqdm(test_dataloader):
        x, y = data

        y_pred = model.forward(x)
        y_pred_probs = softmax(y_pred)
    
        for y_one_hot, y_pred_prob in zip(y, y_pred_probs):
            if np.argmax(y_one_hot) == np.argmax(y_pred_prob):
                correct_num += 1
            total_num += 1

    print(f"Accuracy: {correct_num / total_num * 100 :.2f}%")

if __name__ == "__main__":
    optimizer = SGDOptimizer(learning_rate=0.1)
    model = MyLSTM(optimizer)
    train_dataloader = EmoDataloader(is_train=True)
    test_dataloader = EmoDataloader(is_train=False)
    epochs = 1

    train(model, train_dataloader, epochs=20)
    test(model, test_dataloader)