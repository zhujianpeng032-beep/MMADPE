import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
from torch_geometric.nn import Linear, GINEConv, MessagePassing
from torch_geometric.utils import to_dense_batch
from performer_pytorch import SelfAttention       # pip install performer-pytorch


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        # 对图曼巴进行初始化
        self.drug_GMA_input_dim, self.drug_GMA_output_dim = args.drug_GMA_input_dim, args.drug_GMA_output_dim
        self.dis_GMA_input_dim, self.dis_GMA_output_dim = args.dis_GMA_input_dim, args.dis_GMA_output_dim
        self.hops = args.hops+1
        self.drug_graphmamba = GraphMamba(self.drug_GMA_input_dim, self.drug_GMA_output_dim, self.hops)
        self.dis_graphmamba = GraphMamba(self.dis_GMA_input_dim, self.dis_GMA_output_dim, self.hops)
        # 对图GPS进行初始化
        self.cfg = GPSConfig()  # dim_in = 64, dim_out = 64
        self.cfg.dim_in, self.cfg.dim_out = args.GGPS_input_dim, args.GGPS_output_dim
        self.drug_graphgps = GPSModel(self.cfg)
        self.dis_graphgps = GPSModel(self.cfg)

    def forward(self, data, drs_data, dis_data):

        # 图mamba
        dis_balance_edge_index = data['dis_balance_edge_index']  # 2
        drs_balance_edge_index = data['drs_balance_edge_index']  # 2*
        drug_multi_hop_aggregation = data['drug_multi_hop_aggregation']
        disease_multi_hop_aggregation = data['disease_multi_hop_aggregation']
        drug_x1 = self.drug_graphmamba(drug_multi_hop_aggregation, drs_balance_edge_index)  # 输出大小为663*64
        dis_x1 = self.dis_graphmamba(disease_multi_hop_aggregation, dis_balance_edge_index)  # 输出大小为409*64

        # 图GPS
        drug_x2 = self.drug_graphgps(drs_data.x, drs_data.edge_index, drs_data.edge_attr, pe=drs_data.pe,
                                     rwse=drs_data.rwse, batch=drs_data.batch)  # 输出大小为663*64
        dis_x2 = self.dis_graphgps(dis_data.x, dis_data.edge_index, dis_data.edge_attr, pe=dis_data.pe,
                                   rwse=dis_data.rwse, batch=dis_data.batch)  # 输出大小为409*64


        drug_embedding = drug_x1 * 0.5 + drug_x2 * 0.5
        dis_embedding = dis_x1 * 0.5 + dis_x2 * 0.5

        output = torch.mm(drug_embedding, dis_embedding.t())

        return output

    def decode(self, output, edge_index):   # 点积解码器
        src_nodes = edge_index[0]  # 起始节点
        dst_nodes = edge_index[1]  # 终止节点
        # link_logits是一个长度为4557的一维张量(若为C数据集)
        link_logits = output[src_nodes, dst_nodes]  # 一维张量
        # 如果 link_logits 的形状不满足后续操作的需求，进行重塑
        # 例如，如果后续操作需要一维张量，可以执行如下操作
        if link_logits.dim() > 1:
            link_logits = link_logits.squeeze()

        return link_logits


# ---------------------------------------------GraphGPS---------------------------------------------------------------------

class GPSConfig:
    """
    Configuration class for GraphGPS model
    """

    def __init__(self):
        # Input dimensions
        self.dim_in = 1  # Input node feature dimension
        self.dim_edge = 1  # Input edge feature dimension
        self.dim_pe = 16  # Positional encoding dimension
        self.dim_rwse = 8  # RWSE dimension

        # Model architecture
        self.dim_hidden = 128  # Hidden dimension
        self.dim_out = 1  # Output dimension
        self.num_layers = 6  # Number of GPS layers
        self.num_heads = 8  # Number of attention heads

        # Component types
        self.local_gnn_type = 'GatedGCN'  # Options: 'GatedGCN', 'GINE'
        self.global_model_type = 'Performer'  # Options: 'Transformer', 'Performer'

        # Regularization
        self.dropout = 0.2  # Dropout rate
        self.attn_dropout = 0.2  # Attention dropout rate

        # PNA-specific (if used)
        self.pna_degrees = [1, 2, 3]  # Degrees for PNA aggregators


# 修正的GatedGCNLayer实现
class GatedGCNLayer(MessagePassing):
    """
    Gated Graph Convolutional Layer with edge features
    Based on the GatedGCN implementation from the paper
    """

    def __init__(self, in_dim, out_dim, edge_dim=1, dropout=0.1, residual=True, act='relu'):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.residual = residual

        # Edge feature transformation
        self.edge_mlp = nn.Sequential(
            Linear(in_dim * 2 + edge_dim, out_dim),  # 修改这里，使用edge_dim而不是固定+1
            nn.ReLU() if act == 'relu' else nn.GELU(),
            Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )

        # Node feature transformation
        self.node_mlp = nn.Sequential(
            Linear(in_dim + out_dim, out_dim),
            nn.ReLU() if act == 'relu' else nn.GELU(),
            Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_mlp[0].weight)
        nn.init.xavier_uniform_(self.edge_mlp[2].weight)
        nn.init.xavier_uniform_(self.node_mlp[0].weight)
        nn.init.xavier_uniform_(self.node_mlp[2].weight)

    def forward(self, x, edge_index, edge_attr):
        # Save input for residual connection
        x_in = x

        # Message passing with edge features
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Apply node MLP
        out = self.node_mlp(torch.cat([x, out], dim=-1))

        # Add residual connection and apply layer norm
        if self.residual:
            out = x_in + out if x_in.shape == out.shape else out

        out = self.norm(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        # Combine node features and edge features
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(msg)


class FeatureEncoder(nn.Module):
    """
    Encodes node features, edge features, and positional/structural encodings
    """

    def __init__(self, dim_in, dim_pe, dim_rwse, dim_edge, dim_hidden):
        super().__init__()
        # Node feature encoder
        self.node_encoder = Linear(dim_in, dim_hidden)
        self.node_norm = nn.BatchNorm1d(dim_hidden)

        # Edge feature encoder
        self.edge_encoder = Linear(dim_edge, dim_hidden)
        self.edge_norm = nn.BatchNorm1d(dim_hidden)

        # Positional encoding (PE) encoders
        if dim_pe > 0:
            self.pe_encoder = Linear(dim_pe, dim_hidden)
            self.pe_norm = nn.BatchNorm1d(dim_hidden)

        # Random Walk Structural Encoding (RWSE) encoder
        if dim_rwse > 0:
            self.rwse_encoder = Linear(dim_rwse, dim_hidden)
            self.rwse_norm = nn.BatchNorm1d(dim_hidden)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_attr=None, pe=None, rwse=None):
        # Encode node features
        h = self.node_encoder(x)
        h = self.node_norm(h)
        h = self.act(h)
        h = self.dropout(h)

        # Add positional encodings if available
        if pe is not None and hasattr(self, 'pe_encoder'):
            h_pe = self.pe_encoder(pe)
            h_pe = self.pe_norm(h_pe)
            h_pe = self.act(h_pe)
            h_pe = self.dropout(h_pe)
            h = h + h_pe

        # Add structural encodings if available
        if rwse is not None and hasattr(self, 'rwse_encoder'):
            h_rwse = self.rwse_encoder(rwse)
            h_rwse = self.rwse_norm(h_rwse)
            h_rwse = self.act(h_rwse)
            h_rwse = self.dropout(h_rwse)
            h = h + h_rwse

        # Encode edge features if available
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = self.edge_norm(edge_attr)
            edge_attr = self.act(edge_attr)
            edge_attr = self.dropout(edge_attr)

        return h, edge_attr


class GPSLayer(nn.Module):
    """
    GraphGPS Layer combining local MPNN and global attention
    """

    def __init__(self, dim_h, local_gnn_type, global_model_type, num_heads, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.dropout = dropout

        # Local MPNN component
        if local_gnn_type == 'GatedGCN':
            self.local_gnn = GatedGCNLayer(dim_h, dim_h, edge_dim=dim_h, dropout=dropout)  # 添加edge_dim参数
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(
                Linear(dim_h, dim_h),
                nn.ReLU(),
                Linear(dim_h, dim_h))
            self.local_gnn = GINEConv(gin_nn, edge_dim=dim_h)
        else:
            raise ValueError(f"Unknown local GNN type: {local_gnn_type}")

        # Global attention component
        if global_model_type == 'Transformer':
            self.global_attn = nn.MultiheadAttention(
                dim_h, num_heads,
                dropout=attn_dropout,
                batch_first=True)
        elif global_model_type == 'Performer':
            self.global_attn = SelfAttention(
                dim=dim_h,
                heads=num_heads,
                dropout=attn_dropout,
                causal=False)
        else:
            raise ValueError(f"Unknown global model type: {global_model_type}")

        # Normalization layers
        self.norm_local = nn.LayerNorm(dim_h)
        self.norm_attn = nn.LayerNorm(dim_h)

        # Feed-forward network
        self.ffn = nn.Sequential(
            Linear(dim_h, dim_h * 2),
            nn.ReLU(),
            Linear(dim_h * 2, dim_h),
            nn.Dropout(dropout))
        self.norm_ffn = nn.LayerNorm(dim_h)

    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        h_in = h

        # Local message passing
        h_local = self.local_gnn(h, edge_index, edge_attr)
        h_local = self.norm_local(h_in + h_local)

        # Global attention
        h_dense, mask = to_dense_batch(h, batch)
        if isinstance(self.global_attn, SelfAttention):  # Performer
            h_attn = self.global_attn(h_dense, mask=mask)
        else:  # Transformer
            h_attn, _ = self.global_attn(h_dense, h_dense, h_dense, key_padding_mask=~mask)
        h_attn = h_attn[mask]
        h_attn = self.norm_attn(h_in + h_attn)

        # Combine local and global features
        h = h_local + h_attn

        # Feed-forward network
        h = self.norm_ffn(h + self.ffn(h))

        return h


class GPSModel(nn.Module):
    """
    Complete GraphGPS model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Feature encoder
        self.encoder = FeatureEncoder(
            dim_in=config.dim_in,
            dim_pe=config.dim_pe,
            dim_rwse=config.dim_rwse,
            dim_edge=config.dim_edge,
            dim_hidden=config.dim_hidden
        )

        # Pre-MPNN processing
        self.pre_mp = nn.Sequential(
            Linear(config.dim_hidden, config.dim_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(config.dim_hidden),
            nn.Dropout(config.dropout)
        )

        # GPS layers
        self.layers = nn.ModuleList([
            GPSLayer(
                dim_h=config.dim_hidden,
                local_gnn_type=config.local_gnn_type,
                global_model_type=config.global_model_type,
                num_heads=config.num_heads,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout
            )
            for _ in range(config.num_layers)
        ])

        # Prediction head
        self.head = nn.Sequential(
            Linear(config.dim_hidden, config.dim_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            Linear(config.dim_hidden, config.dim_out)
        )

    def forward(self, x, edge_index, edge_attr, pe, rwse, batch):
        # Initialize batch if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Encode features
        h, edge_attr = self.encoder(x, edge_attr, pe = None, rwse = rwse)
        h = self.pre_mp(h)

        # Process through GPS layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, batch)

        # Final prediction
        out = self.head(h)
        return out


# ---------------------------------------------GraphMamba---------------------------------------------------------------------


class MambaBlock(nn.Module):
    def __init__(self, dim_node, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = dim_node
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * dim_node

        # 投影层
        self.in_proj = nn.Linear(dim_node, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=False
        )
        self.x_proj = nn.Linear(self.d_inner, d_state + 1, bias=False)  # 减少参数
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.zeros(self.d_inner, d_state))  # 参数初始化
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim_node)
        self.act = nn.SiLU()

        # 初始化参数
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 卷积实现
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = self.act(x)

        # 状态空间模型
        x_dbl = self.x_proj(x)
        dt, B = torch.split(x_dbl, [1, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A)  # 更稳定的参数化

        # 扫描实现
        y = self.selective_scan(x, dt, A, B)
        y = y * self.act(z)
        return self.out_proj(y)

    def selective_scan(self, u, delta, A, B):
        batch, seq_len, dim = u.shape
        h = torch.zeros(batch, dim, self.d_state, device=u.device)
        outputs = []

        # 使用cumsum优化扫描过程
        for i in range(seq_len):
            h = torch.exp(A * delta[:, i].unsqueeze(-1)) * h + delta[:, i].unsqueeze(-1) * B[:, i].unsqueeze(1)
            outputs.append(torch.einsum("bdn,bn->bd", h, B[:, i]))

        return torch.stack(outputs, dim=1) + u * self.D


class NodeProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)  # 添加dropout增加稳定性

    def forward(self, x, edge_index):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        batch, num_nodes, _ = x.shape

        # 计算节点度
        node_degree = torch.zeros(batch, num_nodes, device=x.device)
        src_nodes = edge_index[0]
        node_degree.scatter_add_(1, src_nodes.unsqueeze(0).expand(batch, -1),
                                 torch.ones_like(src_nodes).float().unsqueeze(0).expand(batch, -1))

        # 添加噪声和dropout
        noise = self.dropout(torch.rand_like(node_degree) * 0.1)
        sorted_idx = torch.argsort(node_degree + noise, dim=-1)

        # 更高效的重排实现
        batch_indices = torch.arange(batch, device=x.device)[:, None]
        x_sorted = x[batch_indices, sorted_idx]
        return x_sorted, sorted_idx


class GraphMamba(nn.Module):
    def __init__(self, input_dim, output_dim, num_hops, num_mamba_layers=1, num_permutations=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hops = num_hops
        self.num_permutations = num_permutations
        self.num_mamba_layers = num_mamba_layers  # !!! 记录层数

        # 跳数融合
        self.hop_proj = nn.Sequential(
            nn.Linear(num_hops * input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU()
        ) if num_hops > 1 else nn.Identity()

        # 网络结构
        self.mpnn = GATConv(input_dim, input_dim, heads=1, dropout=0.1)
        self.node_processor = NodeProcessor()
        self.mamba_layers = nn.ModuleList([  # !!! 替换为ModuleList实现多层
            MambaBlock(input_dim) for _ in range(num_mamba_layers)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim)
        )

        # 输出层
        self.dim_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU()
        ) if input_dim != output_dim else nn.Identity()

    def forward(self, x, edge_index, inference=False):
        # 输入检查
        assert len(x.shape) == 3, "输入特征必须是3维的[num_hops, num_nodes, input_dim]"
        num_nodes = x.shape[1]

        # 跳数融合
        x = x.permute(1, 0, 2).reshape(num_nodes, -1)
        x = self.hop_proj(x)

        x_mpnn = self.mpnn(x, edge_index)

        if self.training or not inference:
            x_sorted, sorted_idx = self.node_processor(x.unsqueeze(0), edge_index)
            x_mamba = x_sorted
            for mamba_layer in self.mamba_layers:  # !!! 多层处理
                x_mamba = mamba_layer(x_mamba)
            x_mamba = x_mamba.squeeze(0)

            reverse_idx = torch.argsort(sorted_idx.squeeze(0), dim=0)
            x_mamba = x_mamba[reverse_idx]

            x_out = self.mlp(torch.cat([x_mpnn, x_mamba], dim=-1))
        else:
            with torch.no_grad():
                outputs = []
                for _ in range(self.num_permutations):
                    x_sorted, sorted_idx = self.node_processor(x.unsqueeze(0), edge_index)
                    x_mamba = x_sorted
                    for mamba_layer in self.mamba_layers:  # !!! 多层处理
                        x_mamba = mamba_layer(x_mamba)
                    x_mamba = x_mamba.squeeze(0)
                    reverse_idx = torch.argsort(sorted_idx.squeeze(0), dim=0)
                    outputs.append(self.mlp(torch.cat([x_mpnn, x_mamba[reverse_idx]], dim=-1)))
                x_out = torch.mean(torch.stack(outputs), dim=0)

        return self.dim_proj(x_out)

