import sys
from typing import Optional, Type, Union

if sys.version_info >= (3, 12):
    pass
elif sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import torch
import torch.nn
import torch.nn.functional
import torch.nn.modules.loss
import torch.optim
import torch_geometric.data
import torch_geometric.data.batch
import torch_geometric.nn
import torch_geometric.nn.norm

# for type hint
if sys.version_info >= (3, 12):
    type DataBatch = Union[
        torch_geometric.data.Batch, torch_geometric.data.Data
    ]
else:
    DataBatch: TypeAlias = Union[
        torch_geometric.data.Batch, torch_geometric.data.Data
    ]

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

        self.dropout = torch.nn.Dropout(0.05)
        self.relu = torch.nn.LeakyReLU()

        # Output layer ( for scalar output ... REGRESSION )
        self.fc = torch.nn.Linear(embedding_size * 2, 1)

    def forward(self, data: DataBatch) -> torch.Tensor:
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        batch_index: torch.Tensor = data.batch
        edge_attr: Optional[torch.Tensor] = data.edge_attr

        x_conv0: torch.Tensor = self.initial_conv(
            x=x, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv0: torch.Tensor = self.batch_norm0(x_conv0)
        x_conv0: torch.Tensor = self.relu(x_conv0)
        x_conv1: torch.Tensor = self.conv1(
            x=x_conv0, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv1: torch.Tensor = self.batch_norm1(x_conv1)
        x_conv1: torch.Tensor = self.relu(x_conv1)
        x_conv2: torch.Tensor = self.conv2(
            x=x_conv1, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv2: torch.Tensor = self.relu(x_conv2)
        x_conv2: torch.Tensor = self.dropout(x_conv2)
        x_conv3: torch.Tensor = self.conv3(
            x=x_conv2, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv3: torch.Tensor = self.relu(x_conv3)

        # Global Pooling (stack different aggregations)
        # (reason) multiple nodes in one graph....
        # how to make 1 representation for graph??
        # use POOLING!
        # ( gmp : global MAX pooling, gap : global AVERAGE pooling )
        x_pool: torch.Tensor = torch.cat(
            [
                torch_geometric.nn.global_max_pool(x_conv3, batch_index),
                torch_geometric.nn.global_mean_pool(x_conv3, batch_index),
            ],
            dim=1,
        )
        out: torch.Tensor = self.fc(x_pool)

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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @device.setter
    def device(self, __value) -> None:
        raise AttributeError("can't attribute.")


class GNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_size: int = 64,
        n_features_edge: int = 11,
        debug: bool = False,
    ):
        """Graph Convolutional Network

        Reference
        - https://seunghan96.github.io/gnn/PyG_review1/

        Parameters
        ----------
        in_channels : int
            channel size of input (node feature size)
        embedding_size : int, optional
            embedding size of hidden layer, by default 64
        n_features_edge : int, optional
            channel size of edge feature, by default 11
        debug : bool, optional
            debug mode, by default False
        """
        # Init parent
        super().__init__()
        self.debug = debug

        # GCN layers ( for Message Passing )
        self.initial_conv = torch_geometric.nn.NNConv(
            in_channels,
            embedding_size,
            nn=torch.nn.Sequential(
                torch.nn.Linear(n_features_edge, embedding_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(embedding_size, embedding_size * in_channels),
            ),
        )
        self.batch_norm0 = torch_geometric.nn.norm.BatchNorm(embedding_size)
        self.conv1 = torch_geometric.nn.NNConv(
            embedding_size,
            embedding_size,
            nn=torch.nn.Sequential(
                torch.nn.Linear(n_features_edge, embedding_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    embedding_size, embedding_size * embedding_size
                ),
            ),
        )
        self.batch_norm1 = torch_geometric.nn.norm.BatchNorm(embedding_size)
        self.conv2 = torch_geometric.nn.NNConv(
            embedding_size,
            embedding_size,
            nn=torch.nn.Sequential(
                torch.nn.Linear(n_features_edge, embedding_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    embedding_size, embedding_size * embedding_size
                ),
            ),
        )
        self.conv3 = torch_geometric.nn.NNConv(
            embedding_size,
            embedding_size,
            nn=torch.nn.Sequential(
                torch.nn.Linear(n_features_edge, embedding_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    embedding_size, embedding_size * embedding_size
                ),
            ),
        )

        self.dropout = torch.nn.Dropout(0.05)
        self.relu = torch.nn.LeakyReLU()

        # Output layer ( for scalar output ... REGRESSION )
        self.fc = torch.nn.Linear(embedding_size * 2, 1)

    def forward(self, data: DataBatch) -> torch.Tensor:
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        batch_index: torch.Tensor = data.batch
        edge_attr: Optional[torch.Tensor] = data.edge_attr

        x_conv0: torch.Tensor = self.initial_conv(
            x=x, edge_index=edge_index, edge_attr=edge_attr
        )
        x_conv0: torch.Tensor = self.batch_norm0(x_conv0)
        x_conv0: torch.Tensor = self.relu(x_conv0)
        x_conv1: torch.Tensor = self.conv1(
            x=x_conv0, edge_index=edge_index, edge_attr=edge_attr
        )
        x_conv1: torch.Tensor = self.batch_norm1(x_conv1)
        x_conv1: torch.Tensor = self.relu(x_conv1)
        x_conv2: torch.Tensor = self.conv2(
            x=x_conv1, edge_index=edge_index, edge_attr=edge_attr
        )
        x_conv2: torch.Tensor = self.relu(x_conv2)
        x_conv2: torch.Tensor = self.dropout(x_conv2)
        x_conv3: torch.Tensor = self.conv3(
            x=x_conv2, edge_index=edge_index, edge_attr=edge_attr
        )
        x_conv3: torch.Tensor = self.relu(x_conv3)

        # Global Pooling (stack different aggregations)
        # (reason) multiple nodes in one graph....
        # how to make 1 representation for graph??
        # use POOLING!
        # ( gmp : global MAX pooling, gap : global AVERAGE pooling )
        x_pool: torch.Tensor = torch.cat(
            [
                torch_geometric.nn.global_max_pool(x_conv3, batch_index),
                torch_geometric.nn.global_mean_pool(x_conv3, batch_index),
            ],
            dim=1,
        )
        out: torch.Tensor = self.fc(x_pool)

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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @device.setter
    def device(self, __value) -> None:
        raise AttributeError("can't attribute.")


class GINE(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_size: int = 64,
        n_features_edge: int = 11,
        debug: bool = False,
    ):
        """Graph Isomorphism Network

        Reference
        - https://seunghan96.github.io/gnn/PyG_review1/

        Parameters
        ----------
        in_channels : int
            channel size of input (node feature size)
        embedding_size : int, optional
            embedding size of hidden layer, by default 64
        n_features_edge : int, optional
            channel size of edge feature, by default 11
        debug : bool, optional
            debug mode, by default False
        """
        # Init parent
        super().__init__()
        self.debug = debug

        # GIN layers ( for Message Passing )
        self.initial_conv = torch_geometric.nn.GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, embedding_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(embedding_size, embedding_size * in_channels),
            ),
            edge_dim=n_features_edge,
        )
        self.batch_norm0 = torch_geometric.nn.norm.BatchNorm(embedding_size)
        self.conv1 = torch_geometric.nn.GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(embedding_size, embedding_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    embedding_size, embedding_size * embedding_size
                ),
            ),
            edge_dim=n_features_edge,
        )
        self.batch_norm1 = torch_geometric.nn.norm.BatchNorm(embedding_size)
        self.conv2 = torch_geometric.nn.GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(embedding_size, embedding_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    embedding_size, embedding_size * embedding_size
                ),
            ),
            edge_dim=n_features_edge,
        )
        self.conv3 = torch_geometric.nn.GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(embedding_size, embedding_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    embedding_size, embedding_size * embedding_size
                ),
            ),
            edge_dim=n_features_edge,
        )

        self.dropout = torch.nn.Dropout(0.05)
        self.relu = torch.nn.LeakyReLU()

        # Output layer ( for scalar output ... REGRESSION )
        self.fc = torch.nn.Linear(embedding_size * 2, 1)

    def forward(self, data: DataBatch) -> torch.Tensor:
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        batch_index: torch.Tensor = data.batch
        edge_attr: Optional[torch.Tensor] = data.edge_attr

        x_conv0: torch.Tensor = self.initial_conv(
            x=x, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv0: torch.Tensor = self.batch_norm0(x_conv0)
        x_conv0: torch.Tensor = self.relu(x_conv0)
        x_conv1: torch.Tensor = self.conv1(
            x=x_conv0, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv1: torch.Tensor = self.batch_norm1(x_conv1)
        x_conv1: torch.Tensor = self.relu(x_conv1)
        x_conv2: torch.Tensor = self.conv2(
            x=x_conv1, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv2: torch.Tensor = self.relu(x_conv2)
        x_conv2: torch.Tensor = self.dropout(x_conv2)
        x_conv3: torch.Tensor = self.conv3(
            x=x_conv2, edge_index=edge_index, edge_weight=edge_attr
        )
        x_conv3: torch.Tensor = self.relu(x_conv3)

        # Global Pooling (stack different aggregations)
        # (reason) multiple nodes in one graph....
        # how to make 1 representation for graph??
        # use POOLING!
        # ( gmp : global MAX pooling, gap : global AVERAGE pooling )
        x_pool: torch.Tensor = torch.cat(
            [
                torch_geometric.nn.global_max_pool(x_conv3, batch_index),
                torch_geometric.nn.global_mean_pool(x_conv3, batch_index),
            ],
            dim=1,
        )
        out: torch.Tensor = self.fc(x_pool)

        if self.debug:
            print("x.shape", x.shape)
            print("x_conv0.shape", x_conv0.shape)
            print("x_conv1.shape", x_conv1.shape)
            print("x_conv2.shape", x_conv2.shape)
            print("x_conv3.shape", x_conv3.shape)
            print("x_pool.shape", x_pool.shape)
            print("out.shape", out.shape)

        return out


class GraphConvLayer(torch.nn.Module):
    def __init__(
        self,
        message_passing_class: Type[
            Union[
                torch_geometric.nn.GCNConv,
                torch_geometric.nn.NNConv,
                torch_geometric.nn.GINConv,
            ]
        ],
        in_channels: int = 30,
        out_channels: int = 64,
        drop_rate: float = 0.05,
        use_edges: bool = False,
        n_edge_feature: int = 11,
    ) -> None:
        super().__init__()
        self.message_passing_class = message_passing_class
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_rate = drop_rate
        self.use_edges = use_edges
        self.n_edge_feature = n_edge_feature

        if self.message_passing_class is torch_geometric.nn.NNConv:
            if not self.use_edges:
                raise ValueError(
                    "NNConv requires edge features, but use_edges is False."
                )
            self.message_passing = self.message_passing_class(
                self.in_channels,
                self.out_channels,
                nn=torch.nn.Sequential(
                    torch.nn.Linear(self.n_edge_feature, self.out_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=self.drop_rate),
                    torch.nn.Linear(
                        self.out_channels,
                        self.out_channels * self.in_channels,
                    ),
                ),
            )
        elif self.message_passing_class is torch_geometric.nn.GCNConv:
            self.message_passing = self.message_passing_class(
                self.in_channels, self.out_channels
            )
        elif self.message_passing_class is torch_geometric.nn.GINConv:
            if self.use_edges:
                raise ValueError(
                    "GINConv does not require edge features, "
                    "but use_edges is True."
                )
            self.message_passing = self.message_passing_class(
                torch.nn.Sequential(
                    torch.nn.Linear(self.in_channels, self.out_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=self.drop_rate),
                    torch.nn.Linear(
                        self.out_channels, self.out_channels * self.in_channels
                    ),
                )
            )
        else:
            raise ValueError(
                f"Unknown message passing class {self.message_passing_class}."
            )

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        edge_attr: Optional[torch.Tensor] = data.edge_attr
        if self.message_passing_class is torch_geometric.nn.NNConv:
            return self.message_passing(
                x=x, edge_index=edge_index, edge_attr=edge_attr
            )
        elif self.message_passing_class is torch_geometric.nn.GCNConv:
            return self.message_passing(
                x=x, edge_index=edge_index, edge_weight=edge_attr
            )
        elif self.message_passing_class is torch_geometric.nn.GINConv:
            return self.message_passing(x=x, edge_index=edge_index)
        else:
            return self.message_passing(x=x, edge_index=edge_index)


if __name__ == "__main__":
    gc = GraphConvLayer(torch_geometric.nn.GCNConv)
