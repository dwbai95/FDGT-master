import torch.nn as nn

from model import layers


class FDGT(nn.Module):

    def __init__(
        self,
        future_guided,
        order,
        Kt,
        Ks,
        blocks,
        batch_size,
        n_feature,
        T,
        n_vertex,
        gated_act_func,
        graph_conv_type,
        chebconv_matrix,
        drop_rate,
        device,
    ):
        super(FDGT, self).__init__()
        self.future_guided = future_guided
        Ko = T - 3 * (Kt - 1)
        self.Ko = Ko
        self.order = order
        self.device = device
        self.n_feature = n_feature

        self.encode = layers.Encode(
            device,
            Kt,
            Ks,
            order,
            n_feature,
            T,
            n_vertex,
            blocks[0][0],
            blocks[1],
            gated_act_func,
            graph_conv_type,
            chebconv_matrix,
            drop_rate,
        )
        self.decode = layers.Decode(
            future_guided,
            Kt,
            Ko,
            blocks[1][-1],
            blocks[2],
            blocks[3][0],
            n_vertex,
            gated_act_func,
            drop_rate,
        )

    def forward(self, x):
        if self.future_guided is True:

            x_stbs = self.encode(x[:, :-1, :, :])
        else:
            x_stbs = self.encode(x)
        x_out = self.decode(x_stbs, x[:, -1, :, :])
        return x_out
