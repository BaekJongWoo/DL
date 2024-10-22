from dataset.dataloader import Dataloader
from model import MyNN
from Base import Model
from tqdm import tqdm
import numpy as np

learning_rate = 1e-3

def train(model: Model, dataloader: Dataloader):

    for data in tqdm(dataloader):
        x, y = data
        x = np.reshape(x, (-1, 28*28))
        y_pred = model.forward(x)
        loss = 
        print(y_pred)
        return

if __name__ == "__main__":
    train_dataloader = Dataloader('dataset', is_train=True, batch_size=8) # total 60000
    valid_dataloader = Dataloader('dataset', is_train=False, batch_size=8) # total 10000

    model = MyNN()

    train(model, train_dataloader)