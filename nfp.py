import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NFP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, depth=2, nbits=16):
        super(NFP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, nbits)
        self.softmax = nn.Softmax(dim=1)

        self.depth = depth
        self.nbits = nbits

        self.linear3 = nn.Linear(nbits, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

    def forward(self, g, n_feat):
        with g.local_scope():
            fps = torch.zeros([1, self.nbits])
            for _ in range(self.depth):
                g.ndata['h'] = n_feat
                g.update_all(gcn_msg, gcn_reduce)
                h = g.ndata['h']

                r = F.relu(self.linear1(h))
                i = self.softmax(self.linear2(r))
                fps += torch.sum(i, dim=0)

            out = F.relu(self.linear3(fps))
            out = self.linear4(out).squeeze(0)
        return fps, out


def train(model, train_loader, epochs=10):
    loss_func = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for ite, batch in enumerate(train_loader):
            _, bg, label, masks = batch
            n_feat = bg.ndata['h']
            fps, prediction = model(bg, n_feat)
            loss = (loss_func(prediction, label) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (ite + 1)
        print(f'Epoch {epoch}, loss {epoch_loss:.4f}')
        epoch_losses.append(epoch_loss)


def eval_pred(model, data_loader):
    preds = []
    fps = []
    model.eval()
    for _, (smi, bg, label, mask) in enumerate(data_loader):
        n_feat = bg.ndata['h']
        fp, pred = model(bg, n_feat)
        fps.append(fp.detach().numpy()[0])
        preds.append(pred.detach().numpy()[0])

    fps = numpy.vstack(fps)
    preds = numpy.array(preds)[:, numpy.newaxis]

    return preds, fps
