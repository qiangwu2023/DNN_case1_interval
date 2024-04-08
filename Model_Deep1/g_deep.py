
import torch
from torch import nn

def g_D(train_data,X_test,Lambda_U,Beta,Beta0,n_layer,n_node,n_lr,n_epoch):
    Z_train = torch.Tensor(train_data['Z'])
    X_train = torch.Tensor(train_data['X'])
    U_train = torch.Tensor(train_data['U'])
    De_train = torch.Tensor(train_data['De'])
    g_train_true = torch.Tensor(train_data['g_X'])
    X_test = torch.Tensor(X_test)
    Lambda_U = torch.Tensor(Lambda_U)
    Beta0 = torch.Tensor(Beta0)

    class DNNModel(torch.nn.Module):
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(5, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 1))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred

    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def my_loss(De, Z, Beta, Lambda_U, g_X):
        Lam1 = Lambda_U * torch.exp(Z*Beta + g_X)
        loss_fun = -torch.mean(De*torch.log(1-torch.exp(-Lam1)+1e-5) - (1-De)*Lam1)
        return loss_fun

    for epoch in range(n_epoch):
        pred_g_X = model(X_train)
        loss = my_loss(De_train, Z_train, Beta0, Lambda_U, pred_g_X[:, 0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    g_train = model(X_train)
    g_test = model(X_test)
    g_train = g_train[:,0].detach().numpy()
    g_test = g_test[:,0].detach().numpy()
    return {
        'g_train': g_train,
        'g_test': g_test
    }
