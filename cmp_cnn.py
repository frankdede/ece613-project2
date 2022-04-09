import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sampler import sampler

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 96, 11,stride=4) # (227-11)/4+1 =55
        self.pool1 = nn.MaxPool2d(3,stride=2) # (55-3)/2 +1 =27 
        self.batchnorm1 = nn.BatchNorm2d(96,momentum=None) 
        self.conv2 = nn.Conv2d(96, 128, 5,stride=1) # 27-5+1 = 23
        self.pool2 = nn.MaxPool2d(3,stride=2) # (23-3)/2 +1= 11
        self.batchnorm2 = nn.BatchNorm2d(128,momentum=None)
        self.conv31 = nn.Conv2d(128, 256, 3,stride=1,padding=1) # 11-3+2=11
        self.conv32= nn.Conv2d(256, 256, 3,stride=1,padding=1) # 9-3+2=11
        self.conv33 = nn.Conv2d(256, 128, 3,stride=1,padding=1) # 7-3+2=11
        self.pool3 = nn.MaxPool2d(3,stride=2) # (11-3)/2+1 = 5
        self.fc1 = nn.Linear(5*5*128, 1000)
        self.fc2 = nn.Linear(1000,256)
        self.fc3 = nn.Linear(256,10)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.pool1(self.batchnorm1(F.relu(self.conv1(x))))
        x = self.pool2(self.batchnorm2(F.relu(self.conv2(x))))
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sf(x)
        return x

def train(dataloader, epoch):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for epochnum in range(epoch):  # loop over the dataset multiple times
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            train_features, train_labels, img_paths, has_defects, defect_mask_paths, _= data
            img = sampler(train_features,227).float()
            reshapedlabel = torch.sum(train_labels * torch.linspace(0,9,10),axis=1).long()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(img)
            loss = criterion(outputs, reshapedlabel)
            loss.backward()
            optimizer.step()
            # print statistics
            if i % 100 == 99:    
                print(f'[{epochnum + 1}, {i + 1:5d}] loss: {loss.item():.3f}')

    print('Finished Training')
    return net
    
def test(net, dataloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            image, label, img_path, has_defect, defect_mask_path  = data
            img = sampler(image,227).float()
            reshapedlabel = torch.sum(label * torch.linspace(1,10,10),axis=1).long()
            # calculate outputs by running images through the network
            outputs = net(img)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += reshapedlabel.size(0)
            correct += (predicted == reshapedlabel).sum().item()
    return correct // total

