# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled.


# Imports here
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

print("\nImporting is now complete!")

# define data loaders
def init_data(data_dir):
    # Settring data dirs
    # data_dir = 'flowers'
    train_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    print("\nSetting data dirs is now complete!")

    # Define datasets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    validation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    image_datasets = {
        'train_data': datasets.ImageFolder(train_dir, transform = train_transforms),
        'validation_data': datasets.ImageFolder(validation_dir, transform = validation_transforms),
        'test_data': datasets.ImageFolder(test_dir ,transform = test_transforms)
    }

    data_loaders = {
        'train_data_loader': DataLoader(image_datasets['train_data'], batch_size = 64, shuffle = True),
        'validation_data_loader': DataLoader(image_datasets['validation_data'], batch_size = 32),
        'test_data_loader': DataLoader(image_datasets['test_data'], batch_size = 32)
    }

    print("\nDataLoader seting is now complete!")

    return image_datasets, data_loaders

# define set_device by prefered device
def set_device(prefered_device = 'gpu'):
    device = torch.device("cuda" if (prefered_device == 'gpu' and torch.cuda.is_available()) else "cpu")
    print(device)
    print("\nSetting device is now complete!")

    return device

# define model_setup
def model_setup(device, structure = 'vgg16', hidden_layer = 4096, lr = 0.001, dropout = 0.2):
    structures = {
        'vgg16': 25088,
        'alexnet' : 9216
    }
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("The {} is not a valid model. Please input either vgg16 or alexnet!".format(structure))
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
        ('dropout1', nn.Dropout(dropout)),
        ('fc1', nn.Linear(structures[structure], hidden_layer)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layer, 1024)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(1024, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    model.to(device)
    
    return model, optimizer, criterion

# Define validation func to validate loss and accuracy during training
def validate(model, data_loader, criterion, device):
    val_loss = 0
    val_accuracy = 0

    with torch.no_grad():
        model.eval()
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            val_loss += criterion(outputs, labels)
            
            ps = torch.exp(outputs)
            probs, indices = ps.topk(1, dim = 1)
            equalilty = indices == labels.view(*indices.shape)
            val_accuracy += torch.mean(equalilty.type(torch.FloatTensor)).item()

    return val_loss, val_accuracy

# Define train_model func
def train_model(train_data_loader, validation_data_loader, train_data, model, optimizer, criterion, device, epochs = 5, print_every = 50):
    print("\nTraining process is now start with!", device)
    
    model.train()
    steps = 0
    running_loss = 0
    val_len = len(validation_data_loader)

    for epoch in range(epochs):
        for images, labels in train_data_loader:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (steps % print_every == 0):
                val_loss, val_accuracy = validate(model, validation_data_loader, criterion, device)

                print(f"Epoch: {epoch + 1}/{epochs}.."
                     f"Training loss: {running_loss/print_every:.3f}.."
                     f"Validation loss: {val_loss/val_len:.3f}.."
                     f"Validation accuracy: {val_accuracy/val_len:.3f}")
                running_loss = 0
                model.train()
    model.class_to_idx = train_data.class_to_idx
    print("\nTraining process is now complete!")

# Save the checkpoint
def save_checkpoint(model, save_dir):
    print("\nStart saving checkpoint")

    checkpoint = {
        'structure': 'vgg16',
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print("\nComplete saving checkpoint")

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(device, load_dir):
    checkpoint = torch.load(load_dir + '/checkpoint.pth')
    
    structure = checkpoint['structure']
    model, optimizer, criterion = model_setup(device, structure)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, optimizer, criterion

def process_image(image_file):
    image = Image.open(image_file)
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_tensor = image_transforms(image)
    
    return img_tensor

def predict(image_file, model, device, topk = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    img_torch = process_image(image_file)
    
    with torch.no_grad():
        img_torch = img_torch.unsqueeze_(0)
        img_torch = img_torch.float()
        img_torch = img_torch.to(device)
        output = model.forward(img_torch)
        
    ps = torch.exp(output)
    
    probs, indices = ps.topk(topk, dim = 1)
    
    probs = [float(prob) for prob in probs[0]]
    class_map = {val:key for key, val in model.class_to_idx.items()}
    classes = [class_map[int(idx)] for idx in indices[0]]
    
    return probs, classes
