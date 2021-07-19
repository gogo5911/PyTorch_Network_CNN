

def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0

    for X, Y_true in train_loader:

        # 역전파 단계 전에, Optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인)
        # 갱신할 변수들에 대한 모든 변화도를 0으로 만듭니다. 이렇게 하는 이유는
        # 기본적으로 .backward()를 호출할 때마다 변화도가 버퍼(buffer)에 (덮어쓰지 않고)
        # 누적되기 때문입니다.
        optimizer.zero_grad()

        X = X.to(device)
        Y_true = Y_true.to(device)

        #순전파
        y_hat, _ = model(X)
        loss = criterion(y_hat, Y_true) #CrossEntropy
        running_loss += loss.item() * X.size(0)

        #역전파
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)



    return model, optimizer, epoch_loss


def Validate(model, criterion, valid_loader, device):

    model.eval()
    running_loss = 0

    for X, Y_true in valid_loader:

        X = X.to(device)
        Y_true = Y_true.to(device)

        #순전파와 손실 기록하기
        y_hat, _ = model(X) #예측값
        loss = criterion(y_hat, Y_true)  # CrossEntropy
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return  model, epoch_loss