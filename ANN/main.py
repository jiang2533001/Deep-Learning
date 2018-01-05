import torch
import torch as nn
import numpy
from torch.autograd import Variable
import data_preprocess as preprocess
from ANN import ANN
import os.path

def main():
    unneeded = ['RowNumber', 'CustomerId', 'Surname']
    encode = ['Geography','Gender']
    data_set = preprocess.read_data('data_set.csv', unneeded, encode)
    train_x, train_y, test_x, test_y = preprocess.train_test_split(data_set, 0.8)
    train_x = preprocess.normalize_data(train_x)
    test_x = preprocess.normalize_data(test_x)

    if not os.path.isfile('ANN.pt'):     
        process(train_x, train_y)

    train_y_pred = test(train_x)
    check_result(train_y_pred, train_y)

    test_y_pred = test(test_x)
    check_result(test_y_pred, test_y)
    
    #test_data =[[600, 1, 1, 40, 3, 60000, 2, 1, 1, 50000]]
    #test_data = preprocess.normalize_data(test_data) 
    #print test(test_data)

def process(data_x, data_y):
    N, D_in, H, D_out = len(data_x), len(data_x[1,:]), 100, 1

    x = Variable(torch.FloatTensor(data_x))
    y = Variable(torch.FloatTensor(data_y), requires_grad=False) 

    model = ANN(D_in, H, D_out)

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    
    for t in range(500):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model, 'ANN.pt')

def test(data):
    x = Variable(torch.FloatTensor(data)) 
    model = torch.load('ANN.pt')
    return model(x)

def check_result(y_pred, y):
    correct = 0.0
    total = y.size
    y_pred = y_pred.data.numpy()

    for i in range(total):
        if (y_pred[i] + 0.5 > 1 and  y[i] == 1) or (y_pred[i] - 0.5 <= 0 and  y[i] == 0):
            correct += 1.0
    
    print correct/float(total)

if __name__ == '__main__':
    main()
