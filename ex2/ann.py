import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

labels_list = {'A-J': 0, 'BBC': 1, 'J-P': 2, 'NY-T': 3}

class NeuralNetwork(nn.Module):
    """
    A neural network model with three hidden layers and a configurable activation function.

    Attributes:
        input_dim (int): Dimension of the input features.
        activation_fn (str): Activation function to use ('relu' or 'gelu').
        layers (nn.Sequential): The sequential layers of the neural network.
    """
    def __init__(self, input_dim, activation_fn='relu'):
        super(NeuralNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Linear(64, 64),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Linear(64, 32),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): Input features.

        Returns:
            Tensor: Output predictions.
        """
        return self.layers(x)


class CustomDataset(Dataset):
    """
    Custom dataset for handling feature and label tensors.

    Attributes:
        features (Tensor): Tensor of input features.
        labels (Tensor): Tensor of labels.
    """
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieves the feature and label for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Tensor, Tensor]: Features and labels for the given index.
        """
        return self.features[idx], self.labels[idx]


def process_directory(directory, classifier):
    """
    Processes a directory containing different folders and files for classification.

    Args:
        directory (str): Path to the root directory.
        classifier (function): Function to train the classifiers.
    """
    for folder in os.listdir(directory):
        if folder == 'TF-IDF':
            process_tfidf(folder, classifier, directory)
        else:
            for file in os.listdir(os.path.join(directory, folder)):
                process_csv(file, folder, classifier, directory)


def process_tfidf(folder, classifier, directory):
    """
    Processes TF-IDF data from a specific folder.

    Args:
        folder (str): Name of the folder containing TF-IDF files.
        classifier (function): Function to train the classifiers.
        directory (str): Path to the root directory.
    """
    clean_data, lemma_data = [], []
    tfidf_path = os.path.join(directory, folder)

    for file in sorted(os.listdir(tfidf_path)):
        file_path = os.path.join(tfidf_path, file)
        if 'clean' in file:
            clean_data.append(pd.read_excel(file_path).drop(columns=['DocumentIndex']))
        elif 'lemma' in file:
            lemma_data.append(pd.read_excel(file_path).drop(columns=['DocumentIndex']))

    y_labels = create_labels([df.shape[0] for df in clean_data])
    clean_df = pd.concat(clean_data, ignore_index=True)
    lemma_df = pd.concat(lemma_data, ignore_index=True)

    classifier(clean_df.to_numpy(dtype=np.float32), y_labels, 'TF-IDF clean')
    classifier(lemma_df.to_numpy(dtype=np.float32), y_labels, 'TF-IDF lemma')


def process_csv(file, folder, classifier, directory):
    """
    Processes a CSV file for classification.

    Args:
        file (str): Name of the CSV file.
        folder (str): Name of the folder containing the CSV file.
        classifier (function): Function to train the classifiers.
        directory (str): Path to the root directory.
    """
    file_path = os.path.join(directory, folder, file)
    df = pd.read_csv(file_path)
    y_labels = df['Sheet'].map({'A-J': 0, 'BBC': 1, 'J-P': 2, 'NY-T': 3})
    df.drop(columns=['RowIndex', 'Sheet'], inplace=True)
    classifier(df.to_numpy(dtype=np.float32), y_labels.to_numpy(), file)


def create_labels(shape_list):
    """
    Creates labels for data based on shape sizes.

    Args:
        shape_list (List[int]): List of sizes for each data group.

    Returns:
        np.array: Labels array.
    """
    labels = []
    for idx, size in enumerate(shape_list):
        labels.extend([idx] * size)
    return np.array(labels)


def train_neural_net(model, data_loaders, criterion, optimizer, max_epochs=15, patience=3, save_path='best_model.pth'):
    """
    Trains a neural network with early stopping and saves the best model.

    Args:
        model (nn.Module): Neural network model.
        data_loaders (Tuple[DataLoader, DataLoader]): Tuple containing train and validation data loaders.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        max_epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model.

    Returns:
        nn.Module: Best trained model.
    """
    train_loader, val_loader = data_loaders
    best_accuracy, no_improvement = 0, 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)
        val_accuracy = evaluate_model(model, val_loader)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improvement = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improvement += 1

        if no_improvement >= patience:
            break

    model.load_state_dict(torch.load(save_path))
    return model


def evaluate_model(model, data_loader):
    """
    Evaluates the model on a given data loader.

    Args:
        model (nn.Module): Neural network model.
        data_loader (DataLoader): Data loader for evaluation.

    Returns:
        float: Accuracy of the model.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in data_loader:
            predictions = model(features)
            correct += (torch.argmax(predictions, axis=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_classifiers(X, y, model_name):
    """
    Trains classifiers with both ReLU and GELU activation functions and evaluates them.

    Args:
        X (np.array): Feature data.
        y (np.array): Labels data.
        model_name (str): Name for saving the models.
    """
    dataset = CustomDataset(X, y)
    train_size = int(0.72 * len(dataset))
    val_size = int(0.08 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    model_relu = NeuralNetwork(X.shape[1], activation_fn='relu')
    model_gelu = NeuralNetwork(X.shape[1], activation_fn='gelu')

    optimizer_relu = torch.optim.Adam(model_relu.parameters(), lr=0.001)
    optimizer_gelu = torch.optim.Adam(model_gelu.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_neural_net(model_relu, (train_loader, val_loader), criterion, optimizer_relu, save_path=f'{model_name}_relu.pth')
    train_neural_net(model_gelu, (train_loader, val_loader), criterion, optimizer_gelu, save_path=f'{model_name}_gelu.pth')

    print(f"ReLU Accuracy: {evaluate_model(model_relu, test_loader):.4f}")
    print(f"GELU Accuracy: {evaluate_model(model_gelu, test_loader):.4f}")


PATH = 'IR-Newspapers-files-16.12/IR-Newspapers-files/IR-files'
process_directory(PATH, train_classifiers)
