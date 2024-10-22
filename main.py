from dataset.dataloader import Dataloader
from mine.model import Sequantial, Model
from mine.module import Conv2D, Flatten, Linear, MaxPooling, ReLU
from mine.loss import CrossEntropyLoss

from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt


learning_rate = 1e-3
epoch_num = 1

def MyNN():
    h1_size = 256
    h2_size = 128
    model = Sequantial([
        Flatten(),
        Linear(28*28, h1_size),
        ReLU(),
        Linear(h1_size, h2_size),
        ReLU(),
        Linear(h2_size, 10)
    ])
    return model

def MyCNN():
    model = Sequantial([                # n 1 28 28
        Conv2D(1, 3, kernel_size=3),    # n 3 26 26
        ReLU(),
        MaxPooling(),                   # n 3 13 13
        Conv2D(3, 5, kernel_size=6),    # n 5 8 8
        ReLU(),
        MaxPooling(),                   # n 5 4 4
        Flatten(),                      # n 80
        Linear(80, 10),
    ])
    return model

def train(model: Model, dataloader: Dataloader):
    
    loss_fn = CrossEntropyLoss()
    
    loss_values = []

    for epoch in range(epoch_num):
        total_loss = 0
        last_update_time = time.time()
        
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for data in dataloader:
                x, y = data
                y_pred = model.forward(x)
                loss, grad = loss_fn(y, y_pred)
                model.backward(grad, learning_rate)

                loss_values.append(loss)
                total_loss += loss
                if time.time() - last_update_time >= 1:
                    pbar.set_postfix(loss=f"{loss:7.4f}")
                    last_update_time = time.time()
                pbar.update()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    plt.plot(loss_values, label="Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Value")
    plt.title("Loss Over Time")
    plt.legend()
    plt.savefig("loss_graph.png")

if __name__ == "__main__":
    train_dataloader = Dataloader('dataset', is_train=True, batch_size=8) # total 60000
    valid_dataloader = Dataloader('dataset', is_train=False, batch_size=8) # total 10000

    model = MyCNN()

    train(model, valid_dataloader)