import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F



class GraphAttentionLayer(nn.Module):

    def __init__(self, requires_grad=True):
        super(GraphAttentionLayer, self).__init__()
        if requires_grad:
            # unifrom initialization
            self.beta = Parameter(torch.Tensor(1).uniform_(
                0, 1), requires_grad=requires_grad)
        else:
            self.beta = Variable(torch.zeros(1), requires_grad=requires_grad)

    def forward(self, x, adj, aff_cropping):

        norm2 = torch.norm(x, 2, 1).view(-1, 1)
        cos = torch.div(torch.mm(x, x.t()), torch.mm(norm2, norm2.t()) + 1e-7)

        mask = torch.zeros_like(aff_cropping).cuda()
        mask[aff_cropping == 0] = -1e9
        mask[cos<0] = -1e9
        cos = self.beta.cuda() * cos
        masked = cos + mask + 10 * adj

        # propagation matrix
        P = F.softmax(masked, dim=1)

        # attention-guided propagation
        output = torch.mm(P, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (16 -> 16)'


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, initializer=nn.init.xavier_uniform_):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(initializer(
            torch.Tensor(in_features, out_features)))

    def forward(self, input):
        # no bias
        return torch.mm(input, self.weight)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class A2GNN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers, dropout_rate):
        super(A2GNN, self).__init__()

        self.layers = nlayers
        self.dropout_rate = dropout_rate

        self.embeddinglayer = LinearLayer(nfeat, nhid)
        nn.init.xavier_uniform_(self.embeddinglayer.weight)

        self.attentionlayers = nn.ModuleList()
        # for Cora dataset, the first propagation layer is non-trainable
        # and beta is fixed at 0
        self.attentionlayers.append(GraphAttentionLayer(requires_grad=True))
        for i in range(1, self.layers):
            self.attentionlayers.append(GraphAttentionLayer())

        self.outputlayer = LinearLayer(nhid, nclass)
        nn.init.xavier_uniform_(self.outputlayer.weight)

    def forward(self, x, adj, aff_cropping):
        x = F.relu(self.embeddinglayer(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)

        for i in range(self.layers):
            x = self.attentionlayers[i](x, adj, aff_cropping)
        fts = x.clone()

        x = self.outputlayer(x)

        return x,fts
