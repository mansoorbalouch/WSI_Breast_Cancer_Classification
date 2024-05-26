import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from breast_cancer_detection.model import model 
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import copy
import pandas as pd
import shutil
from breast_cancer_detection.parse_config import ConfigParser  
import json


main_dir = "/media/ist/Drive2/MANSOOR/Neuroimaging-Project/Breast_Cancer_Classification_Project"
pretrained_model_dir = f"{main_dir}/WSI_Breast_Cancer_Classification/breast_cancer_detection"

target_dir = f"{main_dir}/Tiles_Data_Cat_Classification" # directory where to save the train/text folders
source_dir = f"{main_dir}/test_tiles/n_14_test" # original directory the data is located 
labels_file = "Labels/test_Predicted_labels_benign_malignant.csv"

model_path_dir = f'{pretrained_model_dir}/saved/'
model_config = f"{model_path_dir}/config.json"
pretrained_model_path = f"{model_path_dir}/model_best.pth"

fine_tuned_model_dir = "Model_Weights"
model_name = "DenseNet"

with open(model_config) as config_file:
    config_dict = json.load(config_file)

config = ConfigParser(config=config_dict)  

def setup_directories():
    if not os.path.dirname(f"{target_dir}/train"):
        os.makedirs(target_dir, exist_ok=True)
        train_dir = os.path.join(target_dir, 'train')
        val_dir = os.path.join(target_dir, 'val')
        for subdir in [train_dir, val_dir]:
            os.makedirs(os.path.join(subdir, 'ben'), exist_ok=True)
            os.makedirs(os.path.join(subdir, 'mal'), exist_ok=True)
        
        labels = pd.read_csv(labels_file)
        ben_files = [f"{labels.loc[i,"Filename"]}.png" for i in range(len(labels)) if labels.loc[i,"Label"] == 0] # 0 represents benign
        mal_files = [f"{labels.loc[i,"Filename"]}.png" for i in range(len(labels)) if labels.loc[i,"Label"] == 1] # 1 represents malignant

        split_and_move_files(ben_files, train_dir = os.path.join(train_dir, 'ben'), val_dir= os.path.join(val_dir, 'ben'))
        split_and_move_files(mal_files, train_dir = os.path.join(train_dir, 'mal'), val_dir= os.path.join(val_dir, 'mal'))

def split_and_move_files(files, train_dir, val_dir):
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)
    for f in train_files:
        shutil.move(os.path.join(source_dir, f), train_dir)
    for f in val_files:
        shutil.move(os.path.join(source_dir, f), val_dir)

def load_pretrained_model(model_path, num_classes=2):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Load your custom model
    myModel = model.densenet121()  # Replace this with the actual function if different
    
    # Assuming the classifier might need to be adjusted
    num_ftrs = myModel.classifier.in_features
    myModel.classifier = torch.nn.Linear(num_ftrs, num_classes)

    # Load state dictionary
    myModel.load_state_dict(checkpoint['state_dict'])

    myModel.eval()  # Set the model to evaluation mode
    return myModel


def train_model(myModel, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cpu'):
    best_acc = 0.0  # Initialize the best accuracy
    best_model_path = f'{fine_tuned_model_dir}/{model_name}_best_model_ben_vs_mal.pth'  # Define path to save the best model

    for epoch in range(num_epochs):
        myModel.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Training phase
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = myModel(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # Validation phase
        myModel.eval()
        val_loss = 0.0
        val_corrects = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = myModel(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')

        # Check if this is the best model and save
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(myModel.state_dict(), best_model_path)
            print(f'New best model saved with accuracy: {best_acc:.4f}')

    print('Training complete')
    return myModel


# Define transformations for the input data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


setup_directories()
# Setup data directories
train_dir = os.path.join(target_dir, 'train')
val_dir = os.path.join(target_dir, 'val')

# Create datasets using ImageFolder
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# Create DataLoaders
batch_size = 4  # Adjust based on your system capability
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

myModel = load_pretrained_model(model_path=pretrained_model_path, num_classes=2)
myModel = model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(myModel.fc.parameters(), lr=0.001)  # Only optimize the final layer

# Setup the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)

# Call the training function
myModel = train_model(myModel, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
