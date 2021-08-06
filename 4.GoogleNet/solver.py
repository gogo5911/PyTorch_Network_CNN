def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0

    for i, data in enumerate(train_loader, 0):

        imgs, classes = data
        imgs, classes = imgs.to(device), classes.to(device)


        output, aux1 , aux2 = model(imgs)
        output_loss = criterion(output, classes)
        aux1_loss = criterion(aux1, classes)
        aux2_loss = criterion(aux2, classes)

        loss = output_loss + 0.3*(aux1_loss + aux2_loss)
        running_loss += loss.item() * output.size(0)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


    epoch_loss = running_loss / len(train_loader.dataset)


    return model, optimizer, epoch_loss


def test(model, criterion, test_loader, device):

    model.eval()
    running_loss = 0

    for i, data in enumerate(test_loader, 0):
        imgs, classes = data
        imgs, classes = imgs.to(device), classes.to(device)

        output, _, _ = model(imgs)
        loss = criterion(output, classes)
        running_loss += loss.item() * output.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)

    return model, epoch_loss