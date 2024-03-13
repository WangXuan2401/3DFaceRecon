import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNN

transform = transforms.Compose(
    [transforms.Resize((48, 48)), 
     transforms.ToTensor()])

# example data 
im = Image.open('example/open1.jpg')
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  

# Load the model 
net = CNN()
net.load_state_dict(torch.load('mode_epoch29.pth'))

# predict 
classes = ('open','close')
with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, 1)[1]
    print("predict:",predict)
    # predict = torch.max(outputs, dim=1)[1].data.numpy()
print(classes[int(predict)])