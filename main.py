from dataset.dataloader import Dataloader

from mine.model import Sequantial, Model
from mine.module import Conv2D, Flatten, Linear, MaxPooling, ReLU, BatchNorm1D
from mine.loss import CrossEntropyLoss, softmax

from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def MyNN():
    h1_size = 256
    h2_size = 128
    model = Sequantial([
        Flatten(),
        Linear(28*28, h1_size),
        ReLU(),
        Linear(h1_size, h2_size),
        ReLU(),
        Linear(h2_size, 10),
        BatchNorm1D()
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
        BatchNorm1D()
    ])
    return model

def MyCNN2():
    model = Sequantial([                # n 1 28 28
        Conv2D(1, 10, kernel_size=5),   # n 10 24 24
        ReLU(),
        MaxPooling(),                   # n 10 12 12
        Conv2D(10, 10, kernel_size=5),  # n 10 8 8
        ReLU(),
        MaxPooling(),                   # n 10 4 4
        Flatten(),                      # n 160
        Linear(160, 10),
        BatchNorm1D()
    ])
    return model


def PrintLosses(train_loss_values, test_loss_values):
    term = 50

    loss_values = np.array(train_loss_values)
    n = len(loss_values) // term
    split_values = np.split(loss_values, n)
    mean_values = np.mean(split_values, axis=1)
    x_values = np.arange(0, len(loss_values), term)

    plt.figure()
    plt.plot(x_values, mean_values, label="Train Loss", alpha=0.6, color='b')

    loss_values = np.array(test_loss_values)
    n = len(loss_values) // term
    split_values = np.split(loss_values, n)
    mean_values = np.mean(split_values, axis=1)
    x_values = np.arange(0, len(loss_values), term)

    plt.plot(x_values, mean_values, label="Test Loss", alpha=0.6, color='r')

    # plt.yscale('log')
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

        PrintLosses(train_loss_values, test_loss_values)
        PrintCM(model, test_dataloader)
        PrintTop3(model, test_dataloader)
        
def PrintCM(model: Model, test_dataloader: Dataloader):
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


def PrintTop3(model: Model, test_dataloader: Dataloader):
    top3_prob = np.zeros((10, 3))
    top3_image = np.zeros((10, 3), dtype=object)

    print("\nGenerating Top 3 Images")
    for test_data in tqdm(test_dataloader, desc="Finding Top 3 Images"):
        tx, ty = test_data

        ty_pred = model.forward(tx)

        ty_pred_probs = np.max(softmax(ty_pred), axis=1)
        ty_pred_index = np.argmax(ty_pred, axis=1)

        for idx in range(len(ty_pred_index)):
            pred_prob = ty_pred_probs[idx]
            pred_index = ty_pred_index[idx]
            image = tx[idx]

            for i in range(3):
                if pred_prob > top3_prob[pred_index, i]:
                    top3_prob[pred_index, i+1:] = top3_prob[pred_index, i:-1]
                    top3_image[pred_index, i+1:] = top3_image[pred_index, i:-1]

                    top3_prob[pred_index, i] = pred_prob
                    top3_image[pred_index, i] = image
                    break

    fig, axes = plt.subplots(10, 3, figsize=(12, 30))
    fig.suptitle("Top 3 Probability", fontsize=50)

    for num in range(10):
        for i in range(3):
            if top3_image[num, i] is not None:
                axes[num, i].imshow(top3_image[num, i].squeeze(), cmap='gray')
                axes[num, i].set_title(f"{top3_prob[num, i]*100:3.2f}%", fontsize=30)
                axes[num, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"top_3_images.png")
    plt.close()

    print("Complete.")

if __name__ == "__main__":
    batch_size = 50
    epoch_num = 10
    learning_rate = 0.5

    train_dataloader = Dataloader('dataset', is_train=True, batch_size=batch_size) # total 60000
    test_dataloader = Dataloader('dataset', is_train=False, batch_size=batch_size) # total 10000

    model = MyCNN2()

    train(model, train_dataloader, test_dataloader, epoch_num, learning_rate)