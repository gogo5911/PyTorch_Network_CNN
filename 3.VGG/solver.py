

def train(model, criterion, optimizer, train_loader, device):

    running_loss = 0

    for i, data in enumerate(train_loader, 0):

        imgs, classes = data
        imgs, classes = imgs.to(device), classes.to(device)


        output = model(imgs)
        loss = criterion(output, classes)

        running_loss += loss.item() * output.size(0)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


    epoch_loss = running_loss / len(train_loader.dataset)


    return model, optimizer, epoch_loss



def test(model, criterion, valid_loader, device):

    model.eval()
    running_loss = 0

    for i, data in enumerate(valid_loader, 0):

        imgs, classes = data
        imgs, classes = imgs.to(device), classes.to(device)

        # 순전파
        output  = model(imgs)
        loss = criterion(output, classes)
        running_loss += loss.item() * output.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return  model, epoch_loss