import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop2d = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.drop2d(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop2d(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    epochs = 5
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
        
        train_losses.append(epoch_loss / len(train_loader))
        
        model.eval()
        test_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                if epoch == epochs - 1:
                    all_preds.extend(pred.squeeze().cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
        
        test_loss /= len(test_loader)
        acc = 100. * correct / len(test_loader.dataset)
        test_accuracies.append(acc)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)\n')
        
        scheduler.step(test_loss)
        
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy (%)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Curves')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    report = classification_report(all_targets, all_preds)
    print(report)
    
    torch.save(model.state_dict(), "mnist_cnn.pth")
    
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(model, dummy_input, "mnist_cnn.onnx", 
                      input_names=["image"], output_names=["logits"], 
                      dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}})
                      
    return model

def predict(image_path, model, device):
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)
    
    tensor_check = transforms.ToTensor()(image)
    if tensor_check.mean() > 0.5:
        image = ImageOps.invert(image)
        
    image = image.point(lambda p: p if p > 80 else 0)
    
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
        
    transform = transforms.Compose([
        transforms.Resize((20, 20), antialias=True),
        transforms.Pad(4, fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1).squeeze().tolist()
        prediction = output.argmax(dim=1, keepdim=True).item()
        confidence = probs[prediction]
        
    return prediction, probs, confidence

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    
    if not os.path.exists("mnist_cnn.pth"):
        print("Training model...")
        model = train()
    else:
        print("Loading saved model...")
        model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device, weights_only=True))
    
    while True:
        img_path = input("\nEnter the path to an image of a digit (or 'q' to quit): ")
        if img_path.lower() == 'q':
            break
        
        if os.path.exists(img_path):
            try:
                pred, probs, conf = predict(img_path, model, device)
                print(f"The model predicts this digit is: {pred} (Confidence: {conf*100:.2f}%)")
                print(f"All Probabilities: {['{:.4f}'.format(p) for p in probs]}")
            except Exception as e:
                print(f"Error processing image: {e}")
        else:
            print("File not found. Please try again.")
