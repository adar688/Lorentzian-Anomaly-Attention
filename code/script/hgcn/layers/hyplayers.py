"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot, zeros
from script.hgcn.manifolds import PoincareBall
import itertools


class HGATConv(nn.Module):
    """
    Hyperbolic graph convolution layer.。
    """
    # שכבת GNN היפרבולית עם Attention

    def __init__(self, manifold, in_features, out_features, c_in, c_out, act=F.leaky_relu,
                 dropout=0.6, att_dropout=0.6, use_bias=True, heads=2, concat=False):
        """
        בונה שלוש תת־שכבות:
HypLinear (טרנספורמציה לינארית היפרבולית),
HypAttAgg (אגרגציית שכנים עם קשב),
HypAct (אקטיבציה היפרבולית). תומך ב־heads/concat.
        """
        super(HGATConv, self).__init__()
        out_features = out_features * heads
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout=dropout, use_bias=use_bias)
        self.agg = HypAttAgg(manifold, c_in, out_features, att_dropout, heads=heads, concat=concat)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in

    def forward(self, x, edge_index):
        """
        מפעיל לפי הסדר: Linear → Attention-Aggregation → Activation ומחזיר את האמבדינג החדש.
        """
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HGCNConv(nn.Module):
    """
    Hyperbolic graph convolution layer, from hgcn。
    """

    #שכבת GNN היפרבולית בסגנון HGCN (בלי קשב):

    def __init__(self, manifold, in_features, out_features, c_in=1.0, c_out=1.0, dropout=0.6, act=F.leaky_relu,
                 use_bias=True):
        super(HGCNConv, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout=dropout)
        self.agg = HypAgg(manifold, c_in, out_features, bias=use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """
    #מממש את הנוסחה שנמאת בסעיך 3.2.2.2 בספר א'

    def __init__(self, manifold, in_features, out_features, c, dropout=0.6, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    """
    אקטיבציה היפרבולית:

__init__: מקבל פונקציית אקטיבציה רגילה (למשל leaky_relu) ועוד עקמומיות כניסה/יציאה.

forward(x):

logmap0: מעביר ל־tangent ב־0,

מפעיל act (רגילה, אוקלידית),

expmap0 מחזיר למניפולד + proj.
מאפשר אקטיבציה “כאילו אוקלידית” אבל בצורה תקינה היפרבולית.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypAggAtt(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """

    """
    אגרגציה עם משקל קשב שמחושב ע"י MLP על [x_i, x_j]:

__init__: MLP קטן שמחזיר משקל לכל קשת.

forward(x, edge_index):

logmap0(x): לטנגנט,

מוסיף self-loops, מוציא כפולים,

לכל קשת (i←j) מחשב score = MLP([x_i,x_j]),

softmax על הקשתות שנכנסות ל־i,

שוקל את x_j לפי המשקלים ומחבר ב־scatter,

expmap0 + proj חזרה למניפולד.
תוצאה: ממוצע משוקלל (attention) של שכנים, היפרבולי.
    """

    def __init__(self, manifold, c, out_features, bias=True):
        super(HypAggAtt, self).__init__()
        self.manifold = manifold
        self.c = c
        self.use_bias = bias
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1))

    def forward(self, x, edge_index=None):
        x_tangent = self.manifold.logmap0(x, c=self.c)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        x_j = torch.nn.functional.embedding(edge_j, x_tangent)
        x_i = torch.nn.functional.embedding(edge_i, x_tangent)

        norm = self.mlp(torch.cat([x_i, x_j], dim=1))
        norm = softmax(norm, edge_i, x_i.size(0)).view(-1, 1)
        support = norm.view(-1, 1) * x_j
        support_t_curv = scatter(support, edge_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
        output = self.manifold.proj(self.manifold.expmap0(support_t_curv, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAgg(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """

    """
    אגרגציה בסגנון GCN (נורמליזציית דרגות):

norm(...): מחשב מטריצת אדג’סים מנורמללת (עם self-loops) ומחזיר גם וקטור נורמליזציה לכל קשת (D^{-1/2} A D^{-1/2}).

forward(x, edge_index):

logmap0(x),

מחשב norm לקשתות,

לוקח את x_j של כל שכן ומכפיל בנורמליזציה,

scatter לפי i כדי לסכם שכנים,

expmap0 + proj.
תוצאה: סכימת שכנים מנורמלת (GCN) אבל באופן היפרבולי.
"""

    def __init__(self, manifold, c, out_features, bias=True):
        super(HypAgg, self).__init__()
        
        # self.manifold = PoincareBall()
        self.manifold = manifold
        self.c = c
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        zeros(self.bias)
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1))

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index=None):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_j = torch.nn.functional.embedding(node_j, x_tangent)
        support = norm.view(-1, 1) * x_j
        support_t = scatter(support, node_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAttAgg(MessagePassing):

    """
    אגרגציית GAT היפרבולית (Heads אמיתיים):

__init__: מגדיר מספר ראשים, וקטורי קשב att_i, att_j לכל ראש, ועוד פרמטרים (dropout, concat).

forward(x, edge_index):

מוסיף self-loops,

logmap0(x), משכפל לפי הקשתות,

מחלק את הייצוג ל־heads: צורה [E, heads, out_channels],

מחשב α = ⟨x_i, att_i⟩ + ⟨x_j, att_j⟩ לכל קשת וראש,

LeakyReLU → softmax על קשתות שנכנסות ל־i → Dropout,

סכימה משוקללת של x_j לפי α ב־scatter,

מחבר ראשים (concat או mean),

expmap0 + proj.
תוצאה: GAT קלאסי אבל בטנגנט+חזרה להיפרבולי.
    """
    def __init__(self, manifold, c, out_features, att_dropout=0.6, heads=1, concat=False):
        super(HypAttAgg, self).__init__()
        self.manifold = manifold
        self.dropout = att_dropout
        self.out_channels = out_features // heads
        self.negative_slope = 0.2
        self.heads = heads
        self.c = c
        self.concat = concat
        self.att_i = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        self.att_j = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x.size(self.node_dim))

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0)

        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)
        support_t = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)

        return support_t

    '''
    def forward(self, x, edge_index):
        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        out = self.propagate(edge_index, x=x_tangent0, num_nodes=x.size(0),original_x=x)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)
        return out
    def message(self, edge_index_i, x_i, x_j, num_nodes,original_x_i, original_x_j):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
        if False:  # Compute distance
            dist = self.manifold.dist(original_x_i, original_x_j, self.c)
            dist = softmax(dist, edge_index_i, num_nodes).reshape(-1, 1)
            alpha = alpha * dist

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)
    '''


# refer to: https://github.com/ferrine/hyrnn/blob/master/hyrnn/nets.py
class HypGRU(nn.Module):
    """
    תא GRU היפרבולי (על PoincaréBall):

__init__: מגדיר משקלי GRU (W,U) ומכין bias כבר על המניפולד (עם expmap0).

reset_parameters: אתחול אחיד לפי ממד.

forward(hyperx, hyperh): מפעיל תא GRU היפרבולי על כניסה ומצב קודם.

mobius_gru_cell(...):

מפרק את המשקולות לשערים (z, r, h̃),

מחשב z_t ו־r_t ע"י טרנספורמציות היפרבוליות ואז sigmoid בטנגנט,

מחשב h_tilde תוך שימוש ב־r_t ⊙ hx (מוגדר כ־mobius_pointwise_mul),

מוציא מצב חדש ע"י חיבורי מוביוס בין hx לתיקון.

one_rnn_transform(...): W⊗h ⊕ U⊗x ⊕ b במוביס (mobius_matvec + mobius_add).

mobius_linear(...): שכבה לינארית היפרבולית כללית (עם Bias/Nonlinearity אופציונליים) + הקרנה.
"""
    def __init__(self, args):
        super(HypGRU, self).__init__()
        self.manifold = PoincareBall()
        self.nhid = args.nhid
        self.weight_ih = Parameter(torch.Tensor(3 * args.nhid, args.nhid), requires_grad=True).to(args.device)
        self.weight_hh = Parameter(torch.Tensor(3 * args.nhid, args.nhid), requires_grad=True).to(args.device)
        if args.bias:
            bias = nn.Parameter(torch.zeros(3, args.nhid) * 1e-5, requires_grad=False)
            self.bias = self.manifold.expmap0(bias).to(args.device)
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nhid)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, hyperx, hyperh):
        out = self.mobius_gru_cell(hyperx, hyperh, self.weight_ih, self.weight_hh, self.bias)
        return out

    def mobius_gru_cell(self, input, hx, weight_ih, weight_hh, bias, nonlin=None, ):
        W_ir, W_ih, W_iz = weight_ih.chunk(3)
        b_r, b_h, b_z = bias
        W_hr, W_hh, W_hz = weight_hh.chunk(3)

        z_t = self.manifold.logmap0(self.one_rnn_transform(W_hz, hx, W_iz, input, b_z)).sigmoid()
        r_t = self.manifold.logmap0(self.one_rnn_transform(W_hr, hx, W_ir, input, b_r)).sigmoid()

        rh_t = self.manifold.mobius_pointwise_mul(r_t, hx)
        h_tilde = self.one_rnn_transform(W_hh, rh_t, W_ih, input, b_h)

        if nonlin is not None:
            h_tilde = self.manifold.mobius_fn_apply(nonlin, h_tilde)
        delta_h = self.manifold.mobius_add(-hx, h_tilde)
        h_out = self.manifold.mobius_add(hx, self.manifold.mobius_pointwise_mul(z_t, delta_h))
        return h_out

    def one_rnn_transform(self, W, h, U, x, b):
        W_otimes_h = self.manifold.mobius_matvec(W, h)
        U_otimes_x = self.manifold.mobius_matvec(U, x)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x)
        return self.manifold.mobius_add(Wh_plus_Ux, b)

    def mobius_linear(self, input, weight, bias=None, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None):
        if hyperbolic_input:
            output = self.manifold.mobius_matvec(weight, input)
        else:
            output = torch.nn.functional.linear(input, weight)
            output = self.manifold.expmap0(output)
        if bias is not None:
            if not hyperbolic_bias:
                bias = self.manifold.expmap0(bias)
            output = self.manifold.mobius_add(output, bias)
        if nonlin is not None:
            output = self.manifold.mobius_fn_apply(nonlin, output)
        output = self.manifold.project(output)
        return output


class TemporalAttentionLayer(nn.Module):
    #מתאים לשכבה 2 במודל (השכבה הטמפורלית)

    """
    Self-Attention על ציר הזמן (לרצף של T snapshots לכל צומת):
    """
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                attn_drop, 
                residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))  # 位置embedding信息[16, 128]
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))  # [128, 128]; W*Q
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input; [143, 16]: 143个节点，每个节点16个位置信息
        position_inputs = torch.arange(0,self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)  # 重复143个节点; 每个节点有16个时间步
        temporal_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F]; 每个节点在各个时刻对应到的128维向量

        # 2: Query, Key based multi-head self attention. [143, 16, 128]
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]; 第一个矩阵第2个维度，乘以，第二个矩阵的第0个维度
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)  # 每个head的维度
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)  # Q*K
        # 4: Masked (causal) softmax to compute attention weights. 目的是将之前没有出现的时间步，设置为0;
        diag_val = torch.ones_like(outputs[0])  # [16,16]的全1向量
        tril = torch.tril(diag_val)  # 下三角阵
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]  重复N次（2288）; [2288, 16, 16]
        padding = torch.ones_like(masks) * (-2**32+1)  # 负无穷
        outputs = torch.where(masks==0, padding, outputs)  # outputs中mask为0的地方，填充padding中负无穷的数值
        outputs = F.softmax(outputs, dim=2)  # output:[2288, 16, 16]
        self.attn_wts_all = outputs # [h*N, T, T]
                
        # 5: Dropout on attention weights.
        # if self.training:
        #     outputs = self.attn_dp(outputs)  # dropout
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]  # (K*Q)*V; ouput-经过归一化后的attention系数[2288, 16, 16]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]
        
        # 6: Feedforward and residual
        #outputs = self.feedforward(outputs)
        # if self.residual:
        #     outputs = outputs + temporal_inputs
        return outputs  # 所有节点聚合时序self-attention后的节点embedding，所有时间

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs


    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)