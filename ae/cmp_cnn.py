import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sampler import sampler
import matplotlib.pyplot as plt

NUM_CLASS = 10


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
        self.fc3 = nn.Linear(256,NUM_CLASS)
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


def train(dataloader, epoch, net=Net()):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for epochnum in range(epoch):  # loop over the dataset multiple times
        train_loss = 0.
        for i, data in enumerate(dataloader):
            train_features, train_labels, img_paths, has_defects, defect_mask_paths, defect_masks = data
            img = sampler(train_features, 227).float()
            outputs = net(img)
            loss = criterion(outputs, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # print(loss.item())
        train_loss = train_loss / len(dataloader)
        print("train_loss", train_loss)
    print('Finished Training')
    return net


def test(net, dataloader):
    confusion = torch.zeros([NUM_CLASS,NUM_CLASS])
    with torch.no_grad():
        for data in dataloader:
            image = data[0]
            label = data[1]
            img = sampler(image, 227).float()
            outputs = net(img)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(label.size(0)):
                confusion[label[i], predicted[i]] = confusion[label[i], predicted[i]] + 1
    sum = confusion.sum(axis=1).reshape([NUM_CLASS, 1])
    print_heatmap(confusion / sum)
    print("Overall accuracy: ", (confusion.diagonal().sum()/sum.sum()))
    return confusion/sum 

def print_heatmap(torchre):
    results = torchre.clone().detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(results)
    ax.set_xticks(range(NUM_CLASS))
    ax.set_yticks(range(NUM_CLASS))
    for i in range(NUM_CLASS):
        for j in range(NUM_CLASS):
            ax.text(j, i, round(results[i][j], 3), ha="center", va="center", color="w")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    plt.show()

