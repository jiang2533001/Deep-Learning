import torch
import torch.nn as nn

class ANN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(ANN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h_relu_1 = self.linear1(x).clamp(min=0)
        h_relu_2 = self.linear2(h_relu_1).clamp(min=0)
        y_pred = self.linear3(h_relu_1).sigmoid()
   
        return y_pred
