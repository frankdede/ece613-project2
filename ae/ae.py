import torch
import torch.nn as nn
from tqdm.notebook import tqdm

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.lyrs = torch.nn.ModuleList()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer9 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer12 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )

        self.layer14 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer15 = nn.Upsample(scale_factor=2)

        self.layer16 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer17 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer18 = nn.Upsample(scale_factor=2)

        self.layer19 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer20 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer21 = nn.Upsample(scale_factor=2)

        self.layer22 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer23 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer24 = nn.Upsample(scale_factor=2)

        self.layer25 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer26 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = x
        y = self.layer1(y)
        y = self.layer2(y)
        result2 = y
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        result5 = y
        y = self.layer6(y)
        y = self.layer7(y)
        y = self.layer8(y)
        result8 = y
        y = self.layer9(y)
        y = self.layer10(y)
        y = self.layer11(y)
        result11 = y
        y = self.layer12(y)
        y = self.layer13(y)
        y = self.layer14(y)
        y = self.layer15(y)
        y = torch.cat((result11, y), 1)
        y = self.layer16(y)
        y = self.layer17(y)
        y = self.layer18(y)
        y = torch.cat((y, result8), 1)
        y = self.layer19(y)
        y = self.layer20(y)
        y = self.layer21(y)
        y = torch.cat((y, result5), 1)
        y = self.layer22(y)
        y = self.layer23(y)
        y = self.layer24(y)
        y = torch.cat((y, result2), 1)
        y = self.layer25(y)
        y = self.layer26(y)
        y = self.softmax(y)
        y = torch.flatten(y, 2)
        return y[:, 0]

    def learn(self, torch_device, train_loader, optimizer, loss_fcn, epochs=30):
        train_loss_list = []
        for epoch in tqdm(range(epochs)):
            train_loss = 0.
            for i, (image, defect_mask) in enumerate(train_loader):
                image = image.to(torch_device)
                defect_mask = defect_mask.to(torch_device)
                output = self(image)
                loss = loss_fcn(output, defect_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # print(loss.item())
            train_loss = train_loss / len(train_loader)
            print("train_loss", train_loss)
            train_loss_list.append(train_loss)


class autoencoder_wasp(nn.Module):
    def __init__(self):
        super(autoencoder_wasp, self).__init__()
        channel = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channel, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=2 * channel, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer3 = nn.MaxPool2d(2, 2)

        coe = 2
        self.wasp4 = nn.Conv2d(in_channels=channel * 2, out_channels=channel * coe, kernel_size=3, padding=4,
                               dilation=4)
        self.wasp4_c1 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=1)
        self.wasp4_c2 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=1)

        self.wasp8 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=3, padding=8,
                               dilation=8)
        self.wasp8_c1 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=1)
        self.wasp8_c2 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=1)

        self.wasp16 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=3, padding=16,
                                dilation=16)
        self.wasp16_c1 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=1)
        self.wasp16_c2 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=1)

        self.wasp32 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=3, padding=32,
                                dilation=32)
        self.wasp32_c1 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=1)
        self.wasp32_c2 = nn.Conv2d(in_channels=channel * coe, out_channels=channel * coe, kernel_size=1)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(256),
            nn.Upsample(size=256)
        )

        self.layer4 = nn.Upsample(scale_factor=2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=channel * 12, out_channels=channel, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layerfinal = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=2, kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y2 = y
        y = self.layer3(y)

        _res4 = self.wasp4(y)
        res4 = self.wasp4_c1(_res4)
        res4 = self.wasp4_c2(res4)

        _res8 = self.wasp8(_res4)
        res8 = self.wasp8_c1(_res8)
        res8 = self.wasp8_c2(res8)

        _res16 = self.wasp16(_res8)
        res16 = self.wasp16_c1(_res16)
        res16 = self.wasp16_c2(res16)

        _res32 = self.wasp32(_res16)
        res32 = self.wasp32_c1(_res32)
        res32 = self.wasp32_c2(res32)

        avg = self.avgpool(y)
        # print(avg.shape,res4.shape,res8.shape,res16.shape,res32.shape)
        y = torch.cat((avg, res4, res8, res16, res32), 1)
        y = self.layer4(y)
        y = torch.cat((y, y2), 1)
        y = self.layer5(y)
        y = self.layerfinal(y)
        y = self.softmax(y)
        y = torch.flatten(y, 2)
        return y[:, 0]

    def learn(self, device, train_loader, optimizer, loss_fcn, epochs=30):
        train_loss_list = []
        for epoch in tqdm(range(epochs)):
            train_loss = 0.
            for i, (image, defect_mask) in enumerate(tqdm(train_loader)):
                image = image.to()
                defect_mask = defect_mask.to(device)
                output = self(image)
                loss = loss_fcn(output, defect_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # print(loss.item())
            train_loss = train_loss / len(train_loader)
            print("train_loss", train_loss)
            train_loss_list.append(train_loss)
