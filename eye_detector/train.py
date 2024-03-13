import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import os
from model import CNN,LeNet,resnet50

root = './data/'

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset,self).__init__()
        img_list = os.listdir(root)
        images = []
        i = 0
        for l in img_list:
            l = root + '/' + l
            for home, dirs, files in os.walk(l, topdown=True):
                for filename in files:
                    images.append((os.path.join(home, filename), i))
            i = i + 1

        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):  # index键
        fn, label = self.images[index]
        img = Image.open(fn).convert('RGB')
        img = img.resize((244, 244), Image.ANTIALIAS)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    train_data = MyDataset(root + 'train', transform=transforms.ToTensor())
    test_data = MyDataset(root + 'test', transform=transforms.ToTensor())
    print("len of train_data",len(train_data))
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)  # shuffle
    test_loader = DataLoader(dataset=test_data, batch_size=64)
    print("len of train_loader", len(train_loader))

    # model = CNN().to(device)
    # model = LeNet()

    model = resnet50() 
    model_weight_path = "resnet50-19c8e357.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # change fc layer structure
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 2)  # binary classification 
    model.to(device)
    print(model)

    # optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)  # Adam
    loss_func = torch.nn.CrossEntropyLoss()  # CE

    save_path = './resnet50_best.pth'
    best_accuracy = 0.0
    for epoch in range(10):
        print('epoch {}'.format(epoch + 1))
        # training
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x)  
            loss = loss_func(out, batch_y.squeeze()).to(device)
            train_loss += loss.item()  
            predict = torch.max(out, 1)[1]
            train_correct = (predict == batch_y).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step() 
        print('Train Accuracy: {:.6f}, Train Loss: {:.6f}'.format(train_acc / (len(train_data)),
                                                            train_loss / (len(train_data))))

        # evaluation--------------------------------
        model.eval()  
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = Variable(batch_x.to(device)), Variable(batch_y.to(device))
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            predict = torch.max(out, 1)[1]
            eval_correct = (predict == batch_y).sum()
            eval_acc += eval_correct.item()
        print('Test Accuracy: {:.6f}, Test Loss: {:.6f}'.format(eval_acc / (len(test_data)), eval_loss / (len(test_data))))

        if eval_acc > best_accuracy:
            best_accuracy = eval_acc
            print("best eval accuracy:",best_accuracy)
            torch.save(model.state_dict(), save_path)
            print("saved best resnet50 model of epoch", epoch+1)