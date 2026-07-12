# README: READ THE COMMENTS, THEY WILL HELP YOU UNDERSTAND WHAT IS HAPPENING WITH EACH LINE OF CODE. LINK FOR CROSS-ENTROPY-LOSS READING: https://huggingface.co/blog/royswastik/cross-entropy-simplified
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# Define the number of subprocesses to use for data loading, the batch size, and validation size
num_workers = 0
batch_size = 20
valid_size = 0.2
#convert to float tensor
transform = transforms.ToTensor()
# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
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
images = images.numpy()

# defining the class and layers; one input layer, 2 hidden layers, an output layer, and a dropout layer do prevent dependency on the previous layer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_1 = 512
        hidden_2 = 512
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
        self.dropout = nn.Dropout(0.5)

    # forward probagation
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x
        
# initialize the NN
model = Net()
print(model)

# set up loss and optimizer functions, as well as initializing the number of epochs and another var
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
n_epochs = 10
valid_loss_min = np.inf

for epoch in range(n_epochs):
    # keeps track of all the chain - ruling --> we r training, so we need to keep track
    train_loss = 0.0
    valid_loss = 0.0
    model.train()
    # for our x's and y's in our data, do this stuff after
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
    # for our x's and y's in our data, do this stuff after
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
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
# calculate and print avg test loss
test_loss = test_loss/len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

# rendering the pixels
fig = plt.figure(figsize=(25, 4))
for i in np.arange(20):
    ax = fig.add_subplot(2, 20/2, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[i]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[i].item()), str(labels[i].item())),
                color=("green" if preds[i]==labels[i] else "red"))

plt.show()
