import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from solver import train, Validate
from LeNet5 import Net
from util import get_accuracy, plot_losses
from datetime import datetime

import matplotlib.pyplot as plt

#######################################################################################
#Some helpers

def parse_args():
    parser = argparse.ArgumentParser(description='LeNet PyTorch Training')

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

IMG_SIZE = 32 #Input image Size
BATCH_SIZE = 32
LEARNING_RATE = 0.001
CLASSES = 10 #MNIST는 0~9의 숫자를 인식 때문에 총 10개
RANDOM_SEED = 42 #난수 발생을 위한 값
EPOCHS = args.epoch
TYPE = args.type

save_checkpoint = True
start_epoch = 0
checkpoint_per_epochs = 1
checkpoint_dir = r'./checkpoint'
pth_path = args.path


#######################################################################################
##Data, transform, dataset and loader

print('==> Preparing data ..')

transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transforms, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transforms)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


#######################################################################################
##Model, criterion and optimizer

print('==> Constructing model ..')

torch.manual_seed(RANDOM_SEED) #Random seed를 고정하기 위한 함수
model = Net(CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #Adam Otimizer 사용
criterion = nn.CrossEntropyLoss() #loss Function은 CrossEntropy 사용


if TYPE == 'train':

    #######################################################################################
    ## Resume


    if resume_train:
        epoch = resume_after_epoch
        checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')

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
    train_losses = []#훈련 손실값
    valid_losses = []#검증 손실값


    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model, optimizer, train_loss = train(model, criterion, optimizer,train_loader, DEVICE)
        train_losses.append(train_loss)

        with torch.no_grad():
            model, valid_loss = Validate(model, criterion, valid_loader, DEVICE)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=DEVICE)
            valid_acc = get_accuracy(model, valid_loader, device=DEVICE)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'  # 훈련 손실값
                  f'Valid loss: {valid_loss:.4f}\t'  # 검증 손실값
                  f'Train accuracy: {100 * train_acc:.2f}\t'  # 훈련 정확도
                  f'Valid accuracy: {100 * valid_acc:.2f}')  # 검증 정확도


        #.pth save 부분
        if save_checkpoint and epoch % checkpoint_per_epochs == 0:
            if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
            checkpoint_file = os.path.join(checkpoint_dir, 'epoch' + str(epoch) + '.pth')
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_file)


    plot_losses(train_losses, valid_losses)

else :
    #######################################################################################
    ## Test

    checkpoint_file = os.path.abspath(pth_path)
    checkpoint = torch.load(os.path.abspath(pth_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    print('==> Testing ..')

    ROW_IMG = 10
    N_ROWS = 5

    fig = plt.figure()
    for index in range(1, ROW_IMG * N_ROWS + 1):
        plt.subplot(N_ROWS, ROW_IMG, index)
        plt.axis('off')
        plt.imshow(valid_dataset.data[index], cmap='gray_r')

        with torch.no_grad():
            model.eval()
            _, probs = model(valid_dataset[index][0].unsqueeze(0))

        title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'

        plt.title(title, fontsize=7)
    plt.show()
    fig.suptitle('LeNet-5 - predictions');