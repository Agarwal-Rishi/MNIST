# README: READ THE COMMENTS, THEY WILL HELP YOU UNDERSTAND WHAT IS HAPPENING WITH EACH LINE OF CODE. LINK FOR CROSS-ENTROPY-LOSS READING: https://huggingface.co/blog/royswastik/cross-entropy-simplified
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# Define the number of subprocesses to use for data loading, the batch size, and validation size
num_workers = 0
batch_size = 128
valid_size = 0.2
#convert to float tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# choose the training and test datasets
train_data = datasets.FashionMNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False,
                                  download=True, transform=transform)
print (train_data)

# split the training data into training and validation sets
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# create data loaders for the training, validation, andtest sets so we aren't handing 60000 images to the model at once
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

# grab a batch of data from the sets
dataiter = iter(train_loader)
images, labels = next(dataiter)
# keep the tensor form for feeding the model, and a numpy copy for plotting
images_tensor = images
images = images.numpy()

# defining the class and layers; conv layers extract features, linear layers classify, dropout prevents overfitting
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
        )
        
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    # forward probagation
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

    # initialize the NN
model = Net()
print(model)

# set up loss and optimizer functions, as well as initializing the number of epochs and another var
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
n_epochs = 20
valid_loss_min = np.inf

for epoch in range(n_epochs):
    # keeps track of all the chain - ruling --> we r training, so we need to keep track
    train_loss = 0.0
    valid_loss = 0.0
    model.train()
    # for our x's and y's in our data, do this stuff
    for data, target in train_loader:
        optimizer.zero_grad()
        # forward probagation, inputing some data x
        output = model(data)
        # loss function
        loss = criterion(output, target)
        # back probagation, built into nn.Module, covers all the calculus for us
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    # now we are evaluating, we are only looking for how much the loss function is, and we are not back probagating, so we don't need model.train()
    model.eval()
    # for our x's and y's in our data, do this stuff
    for data, target in valid_loader:
        output = model(data)
        # loss function
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch + 1,
        train_loss,
        valid_loss
    ))

# MODEL TESTING
model.eval()
test_loss = 0
class_correct = {i: 0 for i in range(10)}
class_total = {i: 0 for i in range(10)}
# for our x's and y's in the test set, do this stuff
for data, target in test_loader:
    # forward probagation
    output = model(data)
    # loss function
    loss = criterion(output, target)
    # test data
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))

    for i in range(len(target)):
        label = target.data[i].item()
        class_correct[label] += correct[i].item()
        class_total[label] += 1
# calculate and print avg test loss


print('Test Loss: {:.6f}\n'.format(test_loss))

accuracy = sum(class_correct.values()) / len(test_loader.sampler)
print('Accuracy: {:.6f}\n'.format(accuracy))

# get model predictions for the batch of images we're about to plot
model.eval()
output = model(images_tensor)
_, preds = torch.max(output, 1)

# rendering the pixels
fig = plt.figure(figsize=(25, 4))
for i in np.arange(20):
    ax = fig.add_subplot(2, 20 // 2, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[i]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[i].item()), str(labels[i].item())),
                color=("green" if preds[i]==labels[i] else "red"))

plt.show()