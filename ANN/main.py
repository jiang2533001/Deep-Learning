import numpy
import torch
import torch as nn
from torch.autograd import Variable
import data_preprocess as preprocess
from ANN import ANN

def main():
    unneeded = ['RowNumber', 'CustomerId', 'Surname']
    encode = ['Geography','Gender']
    data_set = preprocess.read_data('data_set.csv', unneeded, encode)
    train_x, train_y, test_x, test_y = preprocess.train_test_split(data_set, 0.8)
    train_x = preprocess.normalize_data(train_x)
    test_x = preprocess.normalize_data(test_x)

    # N is batch size
    # D_in is input dimension
    # H is hidden dimension
    # D_out is output dimension
    N, D_in, H, D_out = len(train_x), len(train_x[1,:]), 100, 1

    x = Variable(torch.FloatTensor(train_x))
    y = Variable(torch.FloatTensor(train_y), requires_grad=False) 


    model = ANN(D_in, H, D_out)

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(500):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct = 0.0
    for i in range(N):
        if (float(y_pred.data[i]) + 0.5 > 1 and  y.data[i] == 1) or (float(y_pred.data[i]) - 0.5 <= 0 and  y.data[i] == 0):
            correct += 1.0
    
    print correct/float(N)

if __name__ == '__main__':
    main()
