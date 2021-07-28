import os.path
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from AlexNet import Net
from solver import train, test
from util import get_accuracy, plot_losses
from datetime import datetime



#######################################################################################
#Some helpers


def parse_args():
    parser = argparse.ArgumentParser(description='AlexNet PyTorch Training')
    parser.add_argument('--type', '-t', default='train', type=str, help='type')
    parser.add_argument('--resume', '-r', default=-1, type=int,  help='resume after epoch')
    parser.add_argument('--epoch', '-e', default=15, type=int, help='epoch')
    parser.add_argument('--path', '-p', type=str, help='pth path')
    args = parser.parse_args()
    return args



#######################################################################################
#Configurations

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

args = parse_args()
resume_train = args.resume >= 0
resume_after_epoch = args.resume

BATCH_SIZE = 128
CLASSES = 10
LEARNING_RATE = 0.001
EPOCHS = args.epoch
TYPE = args.type

save_checkpoint = True
start_epoch = 0
checkpoint_per_epochs = 1
checkpoint_dir = r'./checkpoint'
pth_path = args.path

CLASSES_NAME = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#######################################################################################
##Data, transform, dataset and loader

print('==> Preparing data ..')

transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(227), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(root='CIFAR10_data', train= True, transform=transform, download= True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root='CIFAR10_data', train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


#######################################################################################
##Model, criterion and optimizer

print('==> Constructing model ..')

model = Net(CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #Adam Otimizer 사용
criterion = nn.CrossEntropyLoss() #loss Function은 CrossEntropy 사용



#######################################################################################
## Resume


if TYPE == 'train':

    if resume_train:
        epoch = resume_after_epoch
        checkpoint_file = os.path.join(checkpoint_dir, 'epoch' + str(epoch) + '.pth')

        print('==> Resuming from checkpoint after epoch {} ..'.format(epoch))
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of epoch {}'.format(epoch)

        checkpoint = torch.load(os.path.join(checkpoint_dir, 'epoch' + str(epoch) + '.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        EPOCHS -= start_epoch



    #######################################################################################
    ## Train and Validate

    print('==> Training ..')

    print_every = 1
    train_losses = []  # 훈련 손실값
    test_losses = []  # 검증 손실값

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model, optimizer, train_loss = train(model, criterion, optimizer, train_loader, DEVICE)
        train_losses.append(train_loss)

        with torch.no_grad():
            model, test_loss = test(model, criterion, test_loader, DEVICE)
            test_losses.append(test_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=DEVICE)
            valid_acc = get_accuracy(model, test_loader, device=DEVICE)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'  # 훈련 손실값
                  f'Valid loss: {test_loss:.4f}\t'  # 검증 손실값
                  f'Train accuracy: {train_acc:.2f}\t'  # 훈련 정확도
                  f'Valid accuracy: {valid_acc:.2f}')  # 검증 정확도


        if save_checkpoint and epoch % checkpoint_per_epochs == 0:
            if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
            checkpoint_file = os.path.join(checkpoint_dir, 'epoch' + str(epoch) + '.pth')
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_file)

    plot_losses(train_losses, test_losses)

else :
    #######################################################################################
    ## Test

    print('==> Testing ..')
    checkpoint = torch.load(os.path.abspath(pth_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    ROW_IMG = 10
    N_ROWS = 5


    fig = plt.figure()
    for index in range(1, ROW_IMG * N_ROWS + 1):
        plt.subplot(N_ROWS, ROW_IMG, index)
        plt.axis('off')
        plt.imshow(test_dataset.data[index])

        with torch.no_grad():
             model.eval()
             tmpTest = test_dataset[index][0].to(DEVICE)
             _, output = model(tmpTest.unsqueeze(0))

        title = f'{CLASSES_NAME[torch.argmax(output)]} ({torch.max(output * 100):.0f}%)'

        plt.title(title, fontsize=5)
    plt.show()
    plt.suptitle('AlexNet')


