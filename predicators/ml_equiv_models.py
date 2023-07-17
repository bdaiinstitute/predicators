import numpy as np
import torch
from torch import nn
import escnn
import escnn.group
from escnn import nn as esnn
from escnn import gspaces


class EquivMLPWrapper:
    """
    The goal of this wrapper is to provide an interface that is identical to normal MLP for easier use
    """

    def __init__(self, g_name, hid_num, input_def=None, output_def=None):
        super().__init__()

        self.group = get_group(g_name=g_name)
        self.g_space = gspaces.no_base_space(self.group)

        # TODO hardcode G-representations for input and output
        # FIXME we will later need to input what can be "rotated" to the model
        stand_repr = self.group.irrep(1) if

        self.in_repr = self.g_space.type(
            *[self.group.irrep(1), self.group.trivial_representation]
            + [self.group.trivial_representation] * 3
        )
        self.out_repr = self.g_space.type(
            *[self.group.irrep(1), self.group.trivial_representation]
        )

        self.hid_dim = get_latent_num(
            g_space=self.g_space,
            h_dim=hid_num,
            h_repr=self.group.regular_representation
        )

        self.mlp = sym_mlp(
            g_space=self.g_space,
            in_field=self.in_repr,
            out_field=self.out_repr,
            h_num=self.hid_num
        )

    def _setup_group(self, g_name):


        assert any([
            # 2D discrete subgroups
            self.g_name.startswith("c"), self.g_name.startswith("d"),
            # 3D discrete subgroups
            self.g_name in ['ico', 'octa', 'full_ico', 'full_octa']
        ])

        # 2D discrete subgroups
        if g_name.startswith("c") or self.g_name.startswith("d"):
            self.dimensionality = 2
            self.rot_num = int(self.g_name[1:])
            self.enable_reflection = "d" in self.g_name  # for dihedral group
            self.group_size = (
                self.rot_num if not self.enable_reflection else (self.rot_num * 2)
            )

            if not self.enable_reflection:
                self.group = escnn.group.cyclic_group(N=self.rot_num)
            else:
                self.group = escnn.group.dihedral_group(N=self.rot_num)

        # 3D discrete subgroups
        else:
            self.dimensionality = 3
            self.enable_reflection = self.g_name.startswith('full')

            name2group = {
                'ico': escnn.group.ico_group(),
                'full_ico': escnn.group.full_ico_group(),
                'octa': escnn.group.octa_group(),
                'full_octa': escnn.group.full_octa_group(),
            }
            self.group = name2group[self.g_name]

        # Create 0D point base space, only for MLP
        self.g_space = escnn.gspaces.no_base_space(self.group)

        print("> Group:", self.group)
        print("> Group space:", self.g_space)

    def forward(self, x):
        x_wrap = self.in_repr(x)
        x_out = self.mlp(x_wrap)
        x_unwrap = x_out.tensor
        return x_unwrap


def get_latent_num(cfg, g_space, h_dim, h_repr=None, multiply_repr_size=False):
    if h_repr is None:
        h_repr = cfg.latent_repr

    if h_repr == 'regular':

        if cfg.latent_dim_factor == 'linear':
            # This keeps the same latent size, but equivariant methods have less learnable parameters
            h_dim = h_dim // g_space.regular_repr.size

        elif cfg.latent_dim_factor == 'sqrt':
            # This option uses sqrt(size) to keep same # of free parameters; fixed: divided then round
            h_dim = int(h_dim / np.sqrt(g_space.regular_repr.size)) + 1

        elif cfg.latent_dim_factor == 'sqrt-1.2x':
            h_dim = int(1.2 * h_dim / np.sqrt(g_space.regular_repr.size))

        elif cfg.latent_dim_factor == 'sqrt-1.5x':
            h_dim = int(1.5 * h_dim / np.sqrt(g_space.regular_repr.size))

        elif cfg.latent_dim_factor == 'const':
            h_dim = h_dim

        else:
            raise ValueError

        repr_size = g_space.regular_repr.size

    elif h_repr == 'trivial':
        h_dim = h_dim
        repr_size = 1

    else:
        raise NotImplementedError("Unsupported latent space representation")

    return h_dim if not multiply_repr_size else h_dim * repr_size


def sym_mlp(g_space, in_field, out_field, h_num, act_fn=esnn.ELU):
    """
    Return an equivariant MLP using equivariant linear layer
    """
    if isinstance(h_num, int):
        h_num = [h_num, h_num]

    # Hidden space
    h_reprs = [d * [g_space.regular_repr] for d in h_num]
    h_field = [g_space.type(*h_repr) for h_repr in h_reprs]

    # TODO hardcode to be 1 hidden layer + input+output layers
    return esnn.SequentialModule(
        esnn.Linear(in_field, h_field[0]),
        # esnn.IIDBatchNorm1d(h_field[0]),
        act_fn(h_field[0]),
        esnn.Linear(h_field[0], h_field[0]),
        # esnn.IIDBatchNorm1d(h_field[1]),
        act_fn(h_field[1]),
        esnn.Linear(h_field[1], out_field),
    )


# def sym_enc(cfg, g_space, in_field, out_field, use_state=True):
#     if use_state:
#         h_num = get_latent_num(cfg, g_space=g_space, h_dim=cfg.enc_dim)
#         h_repr = h_num * [g_space.regular_repr]
#         h_field = g_space.type(*h_repr)
#
#         layers = [
#             esnn.Linear(in_field, h_field),
#             esnn.ELU(h_field),
#             esnn.Linear(h_field, out_field),
#         ]
#
#     else:
#         raise ValueError
#
#     return esnn.SequentialModule(*layers)


# class NormalizeImg(nn.Module):
#     """Normalizes pixel observations to [0,1) range."""
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x.div(255.0)


# def _get_sym_out_shape(in_shape, layers, in_field):
#     """Utility function. Returns the output shape of a network for a given input shape."""
#     x = torch.randn(*in_shape).unsqueeze(0)
#     x = esnn.GeometricTensor(x, in_field)
#     return (
#         (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x)
#         .tensor.squeeze(0)
#         .shape
#     )


if __name__ == '__main__':
    equiv_mlp = EquivMLPWrapper(g_name='d4', hid_num=10)