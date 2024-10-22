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

def train(model: Model, train_dataloader: Dataloader, valid_dataloader: Dataloader, epoch_num: int, learning_rate: float):
    
    loss_fn = CrossEntropyLoss()
    
    train_loss_values = []
    valid_loss_values = []
    
    for epoch in range(epoch_num):
        total_train_loss = 0
        total_valid_loss = 0
        last_update_time = time.time()
        
        # Training loop
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for data in train_dataloader:
                x, y = data
                y_pred = model.forward(x)
                loss, grad = loss_fn(y, y_pred)
                model.backward(grad, learning_rate)

                train_loss_values.append(loss)
                total_train_loss += loss

                if time.time() - last_update_time >= 1:
                    pbar.set_postfix(loss=f"{loss:7.4f}")
                    last_update_time = time.time()
                pbar.update()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation loop
        for data in valid_dataloader:
            x, y = data
            y_pred = model.forward(x)
            loss, _ = loss_fn(y, y_pred)
            valid_loss_values.append(loss)
            total_valid_loss += loss

        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        print(f"Epoch {epoch+1} completed. Average Valid Loss: {avg_valid_loss:.4f}")

        # Plot the Loss over epochs
        plt.figure()
        plt.plot(train_loss_values, label="Train Loss")
        plt.plot(valid_loss_values, label="Validation Loss", linestyle="--")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss Over Time")
        plt.legend()
        plt.savefig(f"loss_graph_epoch_{epoch+1}.png")
        plt.close()

    # After training, plot the confusion matrix
    all_preds = []
    all_labels = []

    for data in valid_dataloader:
        x, y = data
        y_pred = model.forward(x)
        y_pred_class = np.argmax(y_pred, axis=1)
        all_preds.extend(y_pred_class)
        all_labels.extend(y)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Show top 3 scored images with probability (for each class)
    classwise_top3_images = {i: [] for i in range(10)}  # Assumes 10 classes

    for data in valid_dataloader:
        x, y = data
        y_pred = model.forward(x)
        for i in range(len(y)):
            predicted_class = np.argmax(y_pred[i])
            probability = np.max(y_pred[i])
            classwise_top3_images[predicted_class].append((x[i], probability))
    
    # Sort and display top 3 images per class based on the probability
    for class_idx, images_probs in classwise_top3_images.items():
        sorted_images_probs = sorted(images_probs, key=lambda item: item[1], reverse=True)[:3]  # Sort by probability and take top 3
        plt.figure(figsize=(10, 10))
        for idx, (image, prob) in enumerate(sorted_images_probs):
            plt.subplot(1, 3, idx + 1)
            plt.imshow(image.transpose(1, 2, 0))  # Assumes images are in (C, H, W) format
            plt.title(f"Class: {class_idx}, Prob: {prob:.2f}")
            plt.axis("off")
        plt.savefig(f"top3_class_{class_idx}.png")
        plt.close()

if __name__ == "__main__":
    train_dataloader = Dataloader('dataset', is_train=True, batch_size=8) # total 60000
    valid_dataloader = Dataloader('dataset', is_train=False, batch_size=8) # total 10000

    model = MyCNN()

    train(model, train_dataloader, valid_dataloader, epoch_num=2, learning_rate=1e-3)