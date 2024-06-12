import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.models import resnet50, vit_b_16  # Import Vision Transformer
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

class CellClassifier:
    def __init__(self, data_dir, train_dir, val_dir, model_weights_dir, model_type='resnet', batch_size=8, num_epochs=5):
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.model_weights_dir = model_weights_dir
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data transformations, assuming both models use the same transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Ensure directories exist
        for directory in [train_dir, val_dir, model_weights_dir]:
            os.makedirs(directory, exist_ok=True)

    def extract_label(self, filename):
        return filename.split('_uniform')[0]

    def create_class_directories(self, base_dir, classes):
        for cls in classes:
            os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

    def copy_files_to_directories(self, files, source_dir, target_dir):
        for file in files:
            label = self.extract_label(file)
            shutil.copy2(os.path.join(source_dir, file), os.path.join(target_dir, label, file))

    def prepare_datasets(self):
        images = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
        labels = [self.extract_label(f) for f in images]
        unique_classes = set(labels)

        # Create directories and copy files
        self.create_class_directories(self.train_dir, unique_classes)
        self.create_class_directories(self.val_dir, unique_classes)
        train_files, val_files = train_test_split(images, test_size=0.2, stratify=labels)
        self.copy_files_to_directories(train_files, self.data_dir, self.train_dir)
        self.copy_files_to_directories(val_files, self.data_dir, self.val_dir)

    def load_data(self):
        train_dataset = datasets.ImageFolder(self.train_dir, transform=self.transform)
        val_dataset = datasets.ImageFolder(self.val_dir, transform=self.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, train_dataset, val_dataset
    
    def initialize_model(self, train_dataset):
        if self.model_type == 'resnet':
            model = resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(train_dataset.classes))
        elif self.model_type == 'vit':
            model = vit_b_16(pretrained=True)
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, len(train_dataset.classes))
        else:
            raise ValueError("Unsupported model type. Choose 'resnet' or 'vit'")

        model.to(self.device)
        return model

    def train_model(self, model, train_loader, val_loader, train_dataset, val_dataset):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_val_acc = 0.0  # Initialize best validation accuracy

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

            val_loss /= len(val_dataset)
            val_acc = val_corrects.double() / len(val_dataset)

            print(f'Epoch {epoch}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(self.model_weights_dir, f'{self.model_type}_cell_multi_category_classifier.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved best model with accuracy: {best_val_acc:.4f} at epoch {epoch}')

        return model


    def run(self):
        self.prepare_datasets()
        train_loader, val_loader, train_dataset, val_dataset = self.load_data()
        model = self.initialize_model(train_dataset)
        model = self.train_model(model, train_loader, val_loader, train_dataset, val_dataset)
        print("Model training complete and model saved.")


main_dir = "/media/ist/Drive2/MANSOOR/Neuroimaging-Project/Breast_Cancer_Classification_Project"
data_dir = f"{main_dir}/Tiles_Data_Cell_Category_Classification"
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
model_weights_dir = os.path.join(main_dir, "WSI_Breast_Cancer_Classification/Model_Weights")
classifier = CellClassifier(data_dir, train_dir, val_dir, model_weights_dir, model_type="vit", num_epochs=30)
classifier.run()
