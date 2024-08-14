import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
import cv2
from sklearn.metrics import jaccard_score

# Device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.images_dir = os.path.join(root, 'leftImg8bit', split)
        self.labels_dir = os.path.join(root, 'gtFine', split)
        self.images = []
        self.labels = []

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            lbl_dir = os.path.join(self.labels_dir, city)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                self.labels.append(os.path.join(lbl_dir, file_name.replace('leftImg8bit', 'gtFine_labelIds')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        label = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# Transformations
input_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

target_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 512), transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

# DataLoader
train_dataset = CityscapesDataset(root='Datasets', split='train', transform=input_transform, target_transform=target_transform)
val_dataset = CityscapesDataset(root='Datasets', split='val', transform=input_transform, target_transform=target_transform)
test_dataset = CityscapesDataset(root='Datasets', split='test', transform=input_transform, target_transform=target_transform)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 34, kernel_size=(1, 1))  # 34 classes for Cityscapes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)

            outputs = model(images)['out']
            loss = criterion(outputs, labels)

            running_loss += loss.item()

    return running_loss / len(loader)


def compute_metrics(model, loader, device):
    model.eval()
    iou_scores = []
    pixel_accuracies = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)

            outputs = model(images)['out']
            _, preds = torch.max(outputs, 1)

            preds_np = preds.cpu().numpy().reshape(-1) #Reshapes the NumPy array into a 1-dimensional array. This step flattens the 2D predictions into a 1D array, making it easier to compute metrics.
            labels_np = labels.cpu().numpy().reshape(-1)
            """
             'macro' computes the IoU for each class independently and then averages them, giving equal weight to each class regardless of class imbalance.
             """
            iou_scores.append(jaccard_score(labels_np, preds_np, average='macro'))
            pixel_accuracies.append(np.mean(preds_np == labels_np))

    mean_iou = np.mean(iou_scores)
    mean_pixel_accuracy = np.mean(pixel_accuracies)

    return mean_iou, mean_pixel_accuracy



if __name__ == "__main__":
    print(device)
    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

    mean_iou, mean_pixel_accuracy = compute_metrics(model, val_loader, device)
    print(f'Mean IoU: {mean_iou}, Mean Pixel Accuracy: {mean_pixel_accuracy}')

    test_mean_iou, test_mean_pixel_accuracy = compute_metrics(model, test_loader, device)
    print(f'Test Mean IoU: {test_mean_iou}, Test Mean Pixel Accuracy: {test_mean_pixel_accuracy}')