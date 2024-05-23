import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import copy

class BinClassifierTrainer:
    def __init__(self, main_dir, num_epochs=25, batch_size=4, test_size=0.2):
        self.main_dir = main_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.base_dir = f"{main_dir}/Tiles_Data"
        self.source_dir = f"{main_dir}/Tiles_Data"
        self.model_path = "Model_Weights"
        self.model_name = "ResNet18"
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def setup_directories(self):
        if not os.path.dirname(f"{self.source_dir}/train"):
            os.makedirs(self.base_dir, exist_ok=True)
            train_dir = os.path.join(self.base_dir, 'train')
            val_dir = os.path.join(self.base_dir, 'val')
            for subdir in [train_dir, val_dir]:
                os.makedirs(os.path.join(subdir, 'p'), exist_ok=True)
                os.makedirs(os.path.join(subdir, 'n'), exist_ok=True)
            
            files = [f for f in os.listdir(self.source_dir) if os.path.isfile(os.path.join(self.source_dir, f))]
            pos_files = [f for f in files if f.startswith('p')]
            neg_files = [f for f in files if f.startswith('n')]
            self.split_and_move_files(pos_files, os.path.join(train_dir, 'p'), os.path.join(val_dir, 'p'))
            self.split_and_move_files(neg_files, os.path.join(train_dir, 'n'), os.path.join(val_dir, 'n'))

    def split_and_move_files(self, files, train_dir, val_dir):
        train_files, val_files = train_test_split(files, test_size=self.test_size, random_state=42)
        for f in train_files:
            shutil.move(os.path.join(self.source_dir, f), train_dir)
        for f in val_files:
            shutil.move(os.path.join(self.source_dir, f), val_dir)

    def load_data(self):
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.base_dir, x), self.data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        return dataloaders, dataset_sizes, class_names

    def initialize_model(self):
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return model, criterion, optimizer, scheduler

    def train_model(self, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_path = os.path.join(self.model_path, f'{self.model_name}_best_model_epoch_{epoch}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    print(f'Saved best model to {best_model_path}')

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
        return model
        
    def find_latest_checkpoint(self):
        model_files = [f for f in os.listdir(self.model_path) if f.endswith('.pth') and f.startswith(self.model_name)]
        if not model_files:
            return None
        latest_file = max(model_files, key=lambda x: os.path.getctime(os.path.join(self.model_path, x)))
        return os.path.join(self.model_path, latest_file)

    def run(self):
        dataloaders, dataset_sizes, class_names = self.load_data()
        model, criterion, optimizer, scheduler = self.initialize_model()

        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint:
            print(f'Resuming from {latest_checkpoint}')
            model.load_state_dict(torch.load(latest_checkpoint))

        trained_model = self.train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes)
        return trained_model

if __name__ == "__main__":
    main_dir = "/media/ist/Drive2/MANSOOR/Neuroimaging-Project/Breast_Cancer_Classification_Project"
    trainer = BinClassifierTrainer(main_dir=main_dir, num_epochs=20, batch_size=4)
    trainer.setup_directories()  # Only run this once to setup directories and distribute files
    model = trainer.run()
