import torch
import torch.nn as nn
import torch.optim as optim

from net_NLB_TSM import resnet_tsn

from optimization import train_phase, test_phase
from datasets import VideoDataset, VideoDatasetVal, transform_train, transform_val

import argparse



parser = argparse.ArgumentParser(description="PyTorch implementation of TSN and TSM")
parser.add_argument('--strategy',default ='NLB', type=str)
args = parser.parse_args()



classez = ['Archery', 'BalanceBeam', 'BaseballPitch', 'BenchPress', 'Biking',
'Bowling', 'Fencing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'Kayaking',
'LongJump', 'PoleVault', 'PommelHorse', 'Rowing', 'SkateBoarding', 'SkyDiving',
'SumoWrestling', 'Surfing', 'TrampolineJumping']

# Init Network and optiization parameters
# You can tweak this parameters but that is not the homework

net = resnet_tsn(pretrained=True, progress=True)


#if True == 'TSN':

#elif args.strategy == 'TSM':
#    net = resnet_tsm(arch='resnet18',pretrained=True, progress=True)
print('Using {} strategy \n'.format(args.strategy))




criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# Set up datasets and pytorch loaders, adjust batchsize for your GPU, now it uses about 2.1 GB
# Tweak CPU usage here
train_set = VideoDataset('./data/train', transform_train, classez)
val_set = VideoDatasetVal('./data/val', transform_val, classez)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=24, shuffle=True,
                                          num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=24, shuffle=False,
                                          num_workers=4, pin_memory=True)

#GPU selection, if single gpu use cuda:0 instead of cuda:3
has_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if has_cuda else 'cpu')
net = net.to(device)
print(net)
# Training loop
loss_epoch_list = []
train_val_acc_list = []
test_val_acc_list = []


max_epoch = 100
for epoch in range(max_epoch):
    scheduler.step()
    print('Epoch ', epoch+1)
    loss_epoch = train_phase(net, device, train_loader, optimizer, criterion,args)
    
    val_acc    = test_phase(net, device, val_loader, criterion,args)

    train_acc = test_phase(net, device, train_loader, criterion,args)

    loss_epoch_list.append(loss_epoch)
    train_val_acc_list.append(train_acc)
    test_val_acc_list.append(val_acc)

import matplotlib.pyplot as plt
plt.plot(range(1,max_epoch+1), loss_epoch_list)
plt.xlabel('Epoch')
plt.title('NLB+TSM'+' Train Loss')
plt.ylabel('Loss')
plt.savefig('train_loss_'+'NLB+TSM'+'.png')
plt.close()

plt.plot(range(1,max_epoch+1),train_val_acc_list,'r',range(1,max_epoch+1),test_val_acc_list,'k')
plt.legend(['Train','Validation'])
plt.title(args.strategy+' Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuraccy')
plt.savefig('val_acc_'+'NLB+TSM'+'.png')
plt.close()


print('Finished Training')
