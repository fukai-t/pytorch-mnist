import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))]
)

trainset = torchvision.datasets.MNIST(root = '/data',
                                      train = True,
                                      download=True,
                                      transform=_transforms)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.MNIST(root='/data',
                                     train=False,
                                     download=True,
                                     transform=_transforms)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=2)

###############
import torch.nn as nn
import torch.nn.functional as F

class Net (nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        # 1 channel --> 32 channel, 3x3 kernel, 1 stride
        # (3x3 kernel x 32 = 288 parameters)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        # 32 channel --> 64 channel, 3x3 kernel, 1 stride
        # (3x3 kernel x 2 (=64/2) = 18 parameters)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Dropout with p = 0.25
        self.dropout1 = nn.Dropout2d(0.25)

        # Dropout with p = 0.5
        self.dropout2 = nn.Dropout2d(0.5)

        # 9216 nodes --> 128 nodes
        # (9216 x 128 = 1179648 parameters)
        self.fc1 = nn.Linear(9216, 128)

        # 128 nodes --> 10 nodes
        # (128 x 10 = 1280 parameters)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input: 28x28 x 1 channel

        # 28x28 --[3x3 kernel, 1 stride convolution]--> 26x26
        # 1 channel --> 32 channel
        x = self.conv1(x)
        x = F.relu(x)

        # 26x26--[3x3 kernel, 1 stride convolution]-->24x24
        # 32 channel --> 64 channel
        x = self.conv2(x)
        x = F.relu(x)


        # 24x24--[2x2 maxpooling]-->12x12
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # 24x24 x 64 channel --> 9216 nodes
        x = torch.flatten(x, 1)

        # 9216 nodes --> 128 nodes
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # 128 nodes --> 10 nodes
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


########
import torch.optim as optim

net = Net()
# From https://qiita.com/fukuit/items/215ef75113d97560e599
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, moment=0.9)

# From Official example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
criterion = F.nll_loss
optimizer = optim.Adadelta(net.parameters(), lr=1.0)
## value of lr is default value in the example code

epochs = 2

for epoch in range (epochs):
    running_loss = 0.0
    for itr, (inputs, labels) in enumerate (trainloader, 0):
        optimizer.zero_grad()

        output = net(inputs)             # Forward
        loss = criterion(output, labels) # Calculate loss
        loss.backward()                  # Backward
        optimizer.step()                 # Update parameters

        # Print avarage of loss per 100 iterations
        running_loss += loss.item()
        if (itr + 1) % 100 == 0:
            print ('[epoch #{:d}, iter #{:d}] loss: {:3f}'.format(
                (epoch + 1), (itr + 1),
                running_loss / 100))
            running_loss = 0.0


correct = 0
total = 0

with torch.no_grad():
    for (images, labels) in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))
