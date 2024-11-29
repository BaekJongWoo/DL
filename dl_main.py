from data.MNIST.dataloader import MNISTDataloader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

nn.RNN

class torchNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h1_size = 256
        h2_size = 128
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, h1_size)  # Linear(160, 10)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, 10)
        self.bn1 = nn.BatchNorm1d(10)  # BatchNorm1D

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.bn1(self.fc3(x))
        return x

# Neural Network using PyTorch
class torchCNN(nn.Module):
    def __init__(self):
        super(torchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # Conv2D(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)  # Conv2D(10, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling
        self.flatten = nn.Flatten()  # Flatten
        self.fc1 = nn.Linear(10 * 4 * 4, 10)  # Linear(160, 10)
        self.bn1 = nn.BatchNorm1d(10)  # BatchNorm1D

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv -> ReLU -> MaxPool
        x = self.flatten(x)  # Flatten
        x = self.bn1(self.fc1(x))  # Linear -> BatchNorm1D
        return x

def train(model, train_dataloader, test_dataloader, epoch_num, learning_rate, device):
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Optimizer

    model.to(device)

    train_loss_values = []
    test_loss_values = []

    for epoch in range(epoch_num):
        model.train()  # Training mode
        total_train_loss = 0
        
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for data_idx, (x, y) in enumerate(train_dataloader):
                x = torch.from_numpy(x).to(device)
                y = torch.from_numpy(y).to(device)

                optimizer.zero_grad()  # Zero the parameter gradients
                y_pred = model(x)  # Forward pass
                loss = criterion(y_pred, y)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Optimize

                total_train_loss += loss.item()
                train_loss_values.append(loss.item())

                x, y = test_dataloader[data_idx % len(test_dataloader)]
                x = torch.from_numpy(x).to(device)
                y = torch.from_numpy(y).to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                test_loss_values.append(loss.item())

                pbar.set_postfix(loss=f"{loss.item():7.4f}")
                pbar.update()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average Train Loss: {avg_train_loss:.4f}")
        
        PrintLosses(train_loss_values, test_loss_values)
        PrintCM(model, test_dataloader, device)
        PrintTop3(model, test_dataloader, device)

def PrintLosses(train_loss_values, test_loss_values):
    term = 50

    if len(train_loss_values) >= term:
        loss_values = np.array(train_loss_values)
        n = len(loss_values) // term
        if n > 0:  # n이 0이 아닌 경우에만 split 실행
            split_values = np.array_split(loss_values, n)
            mean_values = np.mean(split_values, axis=1)
            x_values = np.arange(0, len(loss_values), term)

            plt.figure()
            plt.plot(x_values, mean_values, label="Train Loss", alpha=0.6, color='b')

    if len(test_loss_values) >= term:
        loss_values = np.array(test_loss_values)
        n = len(loss_values) // term
        if n > 0:  # n이 0이 아닌 경우에만 split 실행
            split_values = np.array_split(loss_values, n)
            mean_values = np.mean(split_values, axis=1)
            x_values = np.arange(0, len(loss_values), term)

            plt.plot(x_values, mean_values, label="Test Loss", alpha=0.6, color='r')

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss over Iteration")
    plt.legend()
    plt.savefig(f"loss_graph.png")
    plt.close()


def PrintCM(model, test_dataloader, device):
    y_true = []
    y_pred = []

    print("\nGenerating Confusion Matrix...")
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            y_pred_probs = model(x)
            y_pred.extend(torch.argmax(y_pred_probs, axis=1).cpu().numpy())
            y_true.extend(torch.argmax(y, axis=1).cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"confusion_matrix.png")
    plt.close()

    print("Complete.")

def PrintTop3(model, test_dataloader, device):
    top3_prob = np.zeros((10, 3))
    top3_image = np.zeros((10, 3), dtype=object)

    print("\nGenerating Top 3 Images")
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(test_dataloader, desc="Finding Top 3 Images"):
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            y_pred = model(x)
            y_pred_probs = F.softmax(y_pred, dim=1)

            for idx in range(len(x)):
                pred_prob, pred_index = torch.topk(y_pred_probs[idx], 1)
                pred_prob = pred_prob.item()
                pred_index = pred_index.item()
                image = x[idx].cpu().numpy().transpose(1, 2, 0)  # Convert to numpy

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
    epoch_num = 30
    learning_rate = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataloader = MNISTDataloader('dataset', is_train=True, batch_size=batch_size) # total 60000
    test_dataloader = MNISTDataloader('dataset', is_train=False, batch_size=batch_size) # total 10000

    model = torchNN()

    train(model, train_dataloader, test_dataloader, epoch_num, learning_rate, device)
