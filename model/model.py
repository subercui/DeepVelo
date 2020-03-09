import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class VeloModel(BaseModel):
    def __init__(self, n_genes, layers = [256, 64]):
        super().__init__()
        self.fc1 = nn.Linear(2*n_genes, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], 2*n_genes)

    def forward(self, x_u, x_s):
        """
        right now it is jus mlp, and the complexity of the middle part does not make sense; 
        Change it to the attention model and constrain the information flow
        """
        batch, n_gene = x_u.shape
        # x should be (batch, features=2*n_gene)
        x = torch.cat([x_u, x_s], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # (batch, 64)
        x = self.fc3(x)  # (batch, genes*2)
        beta = x[:, 0:n_gene] # (batch, genes)
        gamma = x[:, n_gene:2*n_gene]
        pred = beta * x_u + gamma * x_s
        return pred

class VeloTransformer(BaseModel):
    def __init__(self, n_genes, layers = [2, 64], emb_dim=32):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=2*n_genes, embedding_dim=emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=1)
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, layers[0])
        self.fc3 = nn.Linear(layers[0]*2*n_genes, 2*n_genes)

    def forward(self, x_u, x_s):
        """
        right now it is jus mlp, and the complexity of the middle part does not make sense; 
        Change it to the attention model and constrain the information flow
        """
        batch, n_gene = x_u.shape
        # x should be (batch, features=2*n_gene)
        x = torch.cat([x_u, x_s], dim=1)
        x = x.unsqueeze(2)  # (batch, 2*genes, 1)
        x_emb = x * self.emb.weight  # (batch, 2*genes, emb_dim=32)
        x_emb = F.relu(self.fc1(x_emb))
        x_emb = x_emb.permute(1,0,2)  # (2*genes, batch, emb_dim)

        x, attn_output_weights = self.attn(x_emb, x_emb, x_emb)  # (2*genes, batch, emb_dim)
        # TODO: try an additional activation on the attention
        x = x.permute(1,0,2)  # (batch, 2*genes, emb_dim=32)
        x = F.relu(self.fc2(x)).reshape([batch, -1]) # (batch, 2*genes*d)
        x = self.fc3(x)  # (batch, genes*2)
        beta = x[:, 0:n_gene] # (batch, genes)
        gamma = x[:, n_gene:2*n_gene]
        pred = beta * x_u + gamma * x_s
        return pred


from dgl.nn.pytorch import GraphConv
class VeloGCN(BaseModel):
    def __init__(self,
                 g,
                 n_genes,
                 layers = [64],
                 activation =F.relu):
        super(VeloGCN, self).__init__()
        self.g = g # the graph  
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(n_genes*2, layers[0], activation=activation))
        # hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(GraphConv(layers[i], layers[i+1], activation=activation))
        # output layer
        self.layers.append(GraphConv(layers[-1], n_genes*2))

    def forward(self, x_u, x_s):
        """
        right now it is jus mlp, and the complexity of the middle part does not make sense; 
        Change it to the attention model and constrain the information flow
        """
        batch, n_gene = x_u.shape
        # h should be (batch, features=2*n_gene)
        h = torch.cat([x_u, x_s], dim=1) # features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        x = h

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))  # (batch, 64)
        # x = self.fc3(x)  # (batch, genes*2)
        beta = x[:, 0:n_gene] # (batch, genes)
        gamma = x[:, n_gene:2*n_gene]
        pred = beta * x_u + gamma * x_s
        return pred