import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

"""## Downloading dataset and calculate mean and standard deviation for each channel of the Cifar-10.  """

traindataset =  datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
testdataset =   datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
trainloader = DataLoader(traindataset, batch_size=128, shuffle=True)
testloader = DataLoader(testdataset, batch_size=128, shuffle=False)

image = traindataset[0][0]
print(image.shape)

#Code reference [1]
def cal_mean_std():
  cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

  imgs = [item[0] for item in cifar_trainset] 
  imgs = torch.stack(imgs, dim=0).numpy()

  #calculating mean and standard deviation across each channel
  mean_r = imgs[:,0,:,:].mean()
  mean_g = imgs[:,1,:,:].mean()
  mean_b = imgs[:,2,:,:].mean()
  print(mean_r,mean_g,mean_b)
  mean_channels = (mean_r,mean_g,mean_b)

  std_r = imgs[:,0,:,:].std()
  std_g = imgs[:,1,:,:].std()
  std_b = imgs[:,2,:,:].std()
  print(std_r,std_g,std_b)
  std_channels = (std_r,std_g,std_b)
  return mean_channels, std_channels

mean_channels, standard_deviation_channels = cal_mean_std()

"""##Preprocessing Dataset including spliting  data into training set and test set into ratio of 90:10 and performing transfromation including Data Augmentation and normalizing Cifar-10 dataset."""

#Trasforming operation such as data augmentation operation crop, horizontal flip.
#Normalizing images from mean and standard deviation.


transform_training = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_channels, std=standard_deviation_channels),
])

transform_testing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_channels, std=standard_deviation_channels),
])


trainset =  datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_training)
testset =   datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_testing)

"""##Extracting indices for splitting data into ratio of 80:20;
# Training dataset:80%
# Testing dataset:20%
"""

#taking training indices and splitting them in 80 to 20
train_indices = list(range(0, len(trainset)))
#shuffle training indices
np.random.shuffle(train_indices)
#splitting the data into 80 to 20 ratio;
split = int(0.2*len(train_indices))
X_idx = train_indices[split:]
Val_idx = train_indices[:split]

#Defining Sampler
X_train_idx = SubsetRandomSampler(X_idx)
Val_train_idx = SubsetRandomSampler(Val_idx)

"""## Defining DataLoader for Trainingset,Validationset and Testingset"""

trainload = DataLoader(traindataset, batch_size=128, sampler=X_train_idx)
valload = DataLoader(traindataset, batch_size=128,sampler=Val_train_idx)
testload = DataLoader(testdataset, batch_size=128, shuffle=False)

## classes in Cifar-10 dataset labeled as 0-9
classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

"""# Visualizing Cifar-10 Dataset"""

# Code reference [2], [3]
def visualize_Cifar(trainload):
  data  = iter(trainload)
  img, labels = data.next()
  r = 2 #row 
  c = 10 #column
  fig = plt.figure(figsize=(20,7))
  for i in range(0,20):
    fig.add_subplot(r, c, i+1)
    plt.imshow(np.transpose(img[i], (1,2,0))) #implement transpose to convert tensor image
    plt.axis("off")
    plt.title(classes[labels[i]])

  
visualize_Cifar(trainload)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes*BasicBlock.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes,kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
  return ResNet(BasicBlock, [2,1,1,1])

"""## Initializing Cuda or CPU if Not available"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = project1_model()
#Code reference [4]
#introduce dropout in full connected layer
net.linear.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
#check if cuda is available:
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

"""# Initializing Loss function and Optimizer"""

#using crossentropy loss
criterion = nn.CrossEntropyLoss() 
lrate = 0.001
opt = "Adam" ## opt = "Sgd" for using SGD
if opt == "Sgd":
    optimizer = optim.SGD(net.parameters(), lr=lrate, momentum=0.9, weight_decay=5e-4)

if opt == "Adam":
    optimizer = optim.Adam(net.parameters(),betas=[0.9,0.999],lr=lrate)       
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

len(trainload)

"""## Training data on project1_model()"""

def training_testing(epochs):
  train_loss_history = []
  val_loss_history = []
  val_accuracy = []
  train_acc_history = []
  val_acc_history = []
  best_accuracy = 0.0
  val_loss_max = np.inf;
  
  for epoch in range(epochs):
      train_loss = 0.0
      val_loss = 0.0
      net.train()
      for i, data in enumerate(trainload):
          images, labels = data
          images = images.cuda()
          labels = labels.cuda()
          optimizer.zero_grad()
          predicted_output = net(images)
          fit = criterion(predicted_output,labels)
          fit.backward()
          optimizer.step()
          train_loss += fit.item()
      net.eval()
      for i, data in enumerate(valload):
          with torch.no_grad():
              images, labels = data
              images = images.cuda()
              labels = labels.cuda()
              predicted_output = net(images)
              fit = criterion(predicted_output,labels)
              val_loss += fit.item()
      scheduler.step()
      train_loss = train_loss/len(trainload)
      val_loss = val_loss/len(valload)
      train_loss_history.append(train_loss)
      val_loss_history.append(val_loss)
      correct_points_train = 0
      correct_points_val = 0
      correct_points_train +=(torch.eq(torch.max(predicted_output, 1)[1],labels).sum()).data.cpu().numpy()
      correct_points_val +=(torch.eq(torch.max(predicted_output, 1)[1],labels).sum()).data.cpu().numpy()
      train_acc = correct_points_train/len(trainloader)
      val_acc = correct_points_val/len(testloader)
      train_acc_history.append(train_acc)
      val_acc_history.append(val_acc)
      print('Epoch %s, Train loss %s, Validation loss %s, Train Accuracy %s, Validation Accuracy %s'%(epoch, train_loss, val_loss, train_acc, val_acc))
      
      if val_loss_max >= val_loss:
        state = {
            'net': net.state_dict(),
            'acc': val_acc,
            'epoch': epoch,
        }
        print("Saving new best accuracy:{}=======>better than previous validation_accuracy:{}".format(val_acc, best_accuracy))
        best_accuracy = val_acc
        val_loss_max = val_loss
        ## save model
        model_path = './project1_model.pt'
        torch.save(net.state_dict(), model_path)
  print("Accuracy: ",(torch.eq(torch.max(predicted_output, 1)[1],labels).sum()/len(labels)*100).data.cpu().numpy())

  return train_loss_history, val_loss_history, val_accuracy, train_acc_history, val_acc_history

epochs=20;
train_loss_history, val_loss_history, val_accuracy, train_acc_history, val_acc_history = training_testing(epochs)

model_path = './project1_model.pt'
net.load_state_dict(torch.load(model_path, map_location=device), strict=False)

plt.plot(range(epochs),train_loss_history,'-',linewidth=3,label='Train error')
plt.plot(range(epochs),val_loss_history,'-',linewidth=3,label='val error')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()

plt.plot(range(epochs),train_acc_history,'-',linewidth=3,label='Train accuracy')
plt.plot(range(epochs),val_acc_history,'-',linewidth=3,label='val accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

def testing_best_model():
  test_loss = 0.0
  for i, data in enumerate(testload):
              images, labels = data
              images = images.cuda()
              labels = labels.cuda()
              predicted_output = net(images)
              fit = criterion(predicted_output,labels)
              test_loss += fit.item()
  test_loss = test_loss/len(testload)
  test_accuracy = (torch.eq(torch.max(predicted_output, 1)[1],labels).sum()/len(labels)*100).data.cpu().numpy()
  print('Test loss %s, Test Accuracy %s'%(test_loss, test_accuracy))
  print("Accuracy: ", test_accuracy)
testing_best_model()

"""## Printing model summary to analyze the trainable parameter"""

from torchsummary import summary
summary(net,(3,32,32))
model_summary = net,(3,32,32)
print(model_summary)

"""#Printing Whole model architecture"""

print(net)

"""## References
[1].https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data

[2].https://jamesmccaffrey.wordpress.com/2020/08/07/displaying-cifar-10-images-using-pytorch/


[3].https://linuxtut.com/en/feb9f49cdd8560b97667/

[4].https://discuss.pytorch.org/t/inject-dropout-into-resnet-or-any-other-network/66322
"""
