from typing import Optional, Union

import torch
import torch.nn
import torch.nn.functional
import torch.nn.modules.loss
import torch.optim

import torch_geometric.nn
import torch_geometric.nn.norm
import torch_geometric.data
import torch_geometric.data.batch


# for type hint
DataBatch = Union[torch_geometric.data.Batch, torch_geometric.data.Data]

# class GCN(torch.nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.conv1 = torch_geometric.nn.GCNConv(in_channels, 32)
#         self.conv2 = torch_geometric.nn.GCNConv(32, 64)
#         self.relu = torch.nn.ReLU()
#         self.linear = torch.nn.Linear(64 * , out_channels)

#     def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
#         x: torch.Tensor = data.x
#         edge_index: torch.Tensor = data.edge_index
#         edge_attr: Optional[torch.Tensor] = data.edge_attr
#         # x: Node feature matrix of shape [num_nodes, in_channels]
#         # edge_index: Graph connectivity matrix of shape [2, num_edges]
#         x_conv1 = self.conv1(x, edge_index, edge_attr)
#         x_conv1 = self.relu(x_conv1)
#         x_conv2 = self.conv2(x_conv1, edge_index, edge_attr)
#         x_conv2 = self.relu(x_conv2)
#         x_linear = self.linear(torch.flatten(x_conv2))
#         return x_linear


class GCN(torch.nn.Module):
    def __init__(
        self, in_channels: int, embedding_size: int = 64, debug: bool = False
    ):
        """Graph Convolutional Network

        Reference
        - https://seunghan96.github.io/gnn/PyG_review1/

        Parameters
        ----------
        in_channels : int
            channel size of input
        embedding_size : int, optional
            embedding size of hidden layer, by default 64
        debug : bool, optional
            debug mode, by default False
        """
        # Init parent
        super(GCN, self).__init__()
        self.debug = debug

        # GCN layers ( for Message Passing )
        self.initial_conv = torch_geometric.nn.GCNConv(
            in_channels, embedding_size
        )
        self.batch_norm0 = torch_geometric.nn.norm.BatchNorm(embedding_size)
        self.conv1 = torch_geometric.nn.GCNConv(embedding_size, embedding_size)
        self.batch_norm1 = torch_geometric.nn.norm.BatchNorm(embedding_size)
        self.conv2 = torch_geometric.nn.GCNConv(embedding_size, embedding_size)
        self.conv3 = torch_geometric.nn.GCNConv(embedding_size, embedding_size)

        self.dropout = torch.nn.Dropout(0.01)
        self.relu = torch.nn.LeakyReLU()

        # Output layer ( for scalar output ... REGRESSION )
        self.out = torch.nn.Linear(embedding_size * 2, 1)

    def forward(self, data: DataBatch) -> float:
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        batch_index: torch.Tensor = data.batch
        edge_attr: Optional[torch.Tensor] = data.edge_attr

        x_conv0 = self.initial_conv(
            x=x, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv0 = self.batch_norm0(x_conv0)
        x_conv0 = self.relu(x_conv0)
        x_conv1 = self.conv1(
            x=x_conv0, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv1 = self.batch_norm1(x_conv1)
        x_conv1 = self.relu(x_conv1)
        x_conv2 = self.conv2(
            x=x_conv1, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv2 = self.relu(x_conv2)
        x_conv2 = self.dropout(x_conv2)
        x_conv3 = self.conv3(
            x=x_conv2, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv3 = self.relu(x_conv3)

        # Global Pooling (stack different aggregations)
        # (reason) multiple nodes in one graph....
        # how to make 1 representation for graph??
        # use POOLING!
        # ( gmp : global MAX pooling, gap : global AVERAGE pooling )
        x_pool = torch.cat(
            [
                torch_geometric.nn.global_max_pool(x_conv3, batch_index),
                torch_geometric.nn.global_mean_pool(x_conv3, batch_index),
            ],
            dim=1,
        )
        out = self.out(x_pool).flatten()

        if self.debug:
            print("x.shape", x.shape)
            print("x_conv0.shape", x_conv0.shape)
            print("x_conv1.shape", x_conv1.shape)
            print("x_conv2.shape", x_conv2.shape)
            print("x_conv3.shape", x_conv3.shape)
            print("x_pool.shape", x_pool.shape)
            print("out.shape", out.shape)

        return out
        # return out, hidden
