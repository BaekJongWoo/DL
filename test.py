from data.emo.emo_utils import EmoDataloader

from mine.model import Sequantial, ModelBase
from mine.layer import RNN, Linear, LSTM
from mine.dataloader import dataLoader
from mine.loss import CrossEntropyLoss, softmax
from mine.optimizer import SGDOptimizer, AdamOptimizer
from mine.activation import ReLU

from tqdm import tqdm
import numpy as np

def testModel(optimizer):
    model = Sequantial([
        LSTM(50, 100, to_many=True),
        LSTM(100, 64, to_many=False)
    ], optimizer)
    return model
 

def test(model: ModelBase, test_dataloader: dataLoader):
    loss_fn = CrossEntropyLoss()
    
    data = test_dataloader[0]
    x, y = data

    print(x.shape)

    y_pred = model.forward(x)
    loss, grad = loss_fn(y, y_pred)
    model.backward(grad)

    print(y_pred.shape)

if __name__ == "__main__":
    optimizer = SGDOptimizer(learning_rate=0.1)
    model = testModel(optimizer)
    test_dataloader = EmoDataloader(is_train=False)

    test(model, test_dataloader)