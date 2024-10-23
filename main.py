from dataset.dataloader import Dataloader
from mine.model import Sequantial, Model
from mine.module import Conv2D, Flatten, Linear, MaxPooling, ReLU
from mine.loss import CrossEntropyLoss

from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

def drawLosses(train_loss_values, test_loss_values):
    loss_values = np.array(train_loss_values)
    n = 100
    term = len(loss_values) // n
    split_values = np.split(loss_values, n)
    mean_values = np.mean(split_values, axis=1)
    x_values = np.arange(0, len(loss_values), term)

    # Plot the Train Loss with Mean ± Standard Deviation
    plt.figure()
    plt.plot(x_values, mean_values, label="Train Loss", alpha=0.6, color='b')

    loss_values = np.array(test_loss_values)
    n = 100
    term = len(loss_values) // n
    split_values = np.split(loss_values, n)
    mean_values = np.mean(split_values, axis=1)
    x_values = np.arange(0, len(loss_values), term)

    # Plot the Train Loss with Mean ± Standard Deviation
    plt.plot(x_values, mean_values, label="Test Loss", alpha=0.6, color='r')

    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss over Iteration")
    plt.legend()
    plt.savefig(f"loss_graph.png")
    plt.close()

def train(model: Model, train_dataloader: Dataloader, test_dataloader: Dataloader, epoch_num: int, learning_rate: float):
    
    loss_fn = CrossEntropyLoss()
    
    train_loss_values = []
    test_loss_values = []
    
    for epoch in range(epoch_num):
        total_train_loss = 0
        total_test_loss = 0
        last_update_time = time.time()
        
        # Training loop
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for data_idx in range(len(train_dataloader)):
                data = train_dataloader[data_idx]
                x, y = data
                y_pred = model.forward(x)
                loss, grad = loss_fn(y, y_pred)
                model.backward(grad, learning_rate)

                train_loss_values.append(loss)
                total_train_loss += loss


                test_idx = data_idx % len(test_dataloader)
                test_data = test_dataloader[test_idx]
                tx, ty = test_data
                ty_pred = model.forward(tx)
                tloss, _ = loss_fn(ty, ty_pred, no_grad=True)

                test_loss_values.append(tloss)
                total_test_loss += tloss

                if time.time() - last_update_time >= 1:
                    pbar.set_postfix(loss=f"{loss:7.4f}")
                    last_update_time = time.time()
                pbar.update()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_test_loss = total_test_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average Train Loss: {avg_train_loss:.4f}, Averge Test Loss: {avg_test_loss:.4f}")

        drawLosses(train_loss_values, test_loss_values)


if __name__ == "__main__":
    train_dataloader = Dataloader('dataset', is_train=True, batch_size=8) # total 60000
    test_dataloader = Dataloader('dataset', is_train=False, batch_size=8) # total 10000

    model = MyCNN()

    train(model, train_dataloader, test_dataloader, epoch_num=10, learning_rate=1e-3)