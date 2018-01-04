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
        #print y_pred

        loss = criterion(y_pred, y)
        print(t, loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
