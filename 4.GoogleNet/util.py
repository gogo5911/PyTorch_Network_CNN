import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_accuracy(model, data_loader, device):

    correct = 0

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(data_loader, 0):

            imgs, classes = data
            imgs, classes = imgs.to(device), classes.to(device)

            output, _, _ = model(imgs)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(classes.view_as(pred)).sum().item()

    test_accuracy = 100. * correct / len(data_loader.dataset)
    return test_accuracy



def plot_losses(train_losses, valid_losses):

    plt.style.use('seaborn')

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()
    plt.show()

    plt.style.use('default')