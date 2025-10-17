from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class PhyCell_Cell(nn.Module):
    def __init__(self, in_channels, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = in_channels
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential()
        self.F.add_module(
            "conv1",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=F_hidden_dim,
                kernel_size=self.kernel_size,
                stride=(1, 1),
                padding=self.padding,
            ),
        )
        self.F.add_module("bn1", nn.GroupNorm(7, F_hidden_dim))
        self.F.add_module(
            "conv2",
            nn.Conv2d(
                in_channels=F_hidden_dim,
                out_channels=in_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
        )

        self.convgate = nn.Conv2d(
            in_channels=self.input_dim + self.input_dim,
            out_channels=self.input_dim,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=self.bias,
        )

    def forward(self, x, hidden):  # x [batch_size, hidden_dim, height, width]
        # print("x shape: ", x.shape)
        # print("hidden shape: ", hidden.shape)
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)  # prediction
        next_hidden = hidden_tilde + K * (
            x - hidden_tilde
        )  # correction , Haddamard product
        return next_hidden


class PhyCell(nn.Module):
    def __init__(
        self,
        input_shape,
        in_channels,
        F_hidden_dims,
        n_layers,
        kernel_size,
    ):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(
                PhyCell_Cell(
                    in_channels=in_channels,
                    F_hidden_dim=self.F_hidden_dims[i],
                    kernel_size=self.kernel_size,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self, input, first_timestep=False
    ):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input.data.size()[0]
        if first_timestep:
            self.initHidden(
                batch_size, input.device
            )  # init Hidden at each forward start
        for j, cell in enumerate(self.cell_list):
            self.H[j] = self.H[j].to(input.device)
            if j == 0:  # bottom layer
                self.H[j] = cell(input, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], self.H[j])
        return self.H, self.H

    def initHidden(self, batch_size, device):
        self.H = []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(
                    batch_size,
                    self.in_channels,
                    self.input_shape[0],
                    self.input_shape[1],
                ).to(device)
            )

    def setHidden(self, H):
        self.H = H


class PhyD_ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(PhyD_ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden):  # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class PhyD_ConvLSTM(nn.Module):
    def __init__(self, input_shape, in_channels, hidden_dims, n_layers, kernel_size):
        super(PhyD_ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = in_channels
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            print(
                "layer ",
                i,
                "input dim ",
                cur_input_dim,
                " hidden dim ",
                self.hidden_dims[i],
            )
            cell_list.append(
                PhyD_ConvLSTM_Cell(
                    input_shape=self.input_shape,
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_size,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self, input_, first_timestep=False
    ):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if first_timestep:
            self.initHidden(
                batch_size, input_.device
            )  # init Hidden at each forward start
        for j, cell in enumerate(self.cell_list):
            self.H[j] = self.H[j].to(input_.device)
            self.C[j] = self.C[j].to(input_.device)
            if j == 0:  # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))
        return (self.H, self.C), self.H  # (hidden, output)

    def initHidden(self, batch_size, device):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(
                    batch_size,
                    self.hidden_dims[i],
                    self.input_shape[0],
                    self.input_shape[1],
                ).to(device)
            )
            self.C.append(
                torch.zeros(
                    batch_size,
                    self.hidden_dims[i],
                    self.input_shape[0],
                    self.input_shape[1],
                ).to(device)
            )

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


class dcgan_conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=1,
            ),
            nn.GroupNorm(16, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if stride == 2:
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nin,
                out_channels=nout,
                kernel_size=(3, 3),
                stride=stride,
                padding=1,
                output_padding=output_padding,
            ),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class encoder_E(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, patch_size=4):
        super(encoder_E, self).__init__()
        assert patch_size in [2, 4]
        stride_2 = patch_size // 2
        # input is (1) x 64 x 64
        mid_channel = out_channel // 2
        self.c1 = dcgan_conv(in_channel, mid_channel, stride=2)  # (32) x 32 x 32
        self.c2 = dcgan_conv(mid_channel, mid_channel, stride=1)  # (32) x 32 x 32
        self.c3 = dcgan_conv(
            mid_channel, out_channel, stride=stride_2
        )  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_D(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, patch_size=4):
        super(decoder_D, self).__init__()
        assert patch_size in [2, 4]
        stride_2 = patch_size // 2
        output_padding = 1 if stride_2 == 2 else 0

        mid_channel = out_channel // 2

        self.upc1 = dcgan_upconv(out_channel, mid_channel, stride=2)  # (32) x 32 x 32
        self.upc2 = dcgan_upconv(mid_channel, mid_channel, stride=1)  # (32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(
            in_channels=mid_channel,
            out_channels=in_channel,
            kernel_size=(3, 3),
            stride=stride_2,
            padding=1,
            output_padding=output_padding,
        )  # (nc) x 64 x 64

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(in_channel, out_channel, stride=1)  # (64) x 16 x 16
        self.c2 = dcgan_conv(out_channel, out_channel, stride=1)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2


class decoder_specific(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(out_channel, out_channel, stride=1)  # (64) x 16 x 16
        self.upc2 = dcgan_upconv(out_channel, in_channel, stride=1)  # (32) x 32 x 32

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2


class PhyD_EncoderRNN(torch.nn.Module):
    def __init__(self, phycell, convcell, in_channel=1, patch_size=4):
        super(PhyD_EncoderRNN, self).__init__()
        self.encoder_E = encoder_E(
            in_channel=in_channel, out_channel=64, patch_size=patch_size
        )  # general encoder 64x64x1 -> 32x32x32
        self.encoder_Ep = encoder_specific(
            in_channel=64, out_channel=64
        )  # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific(in_channel=64, out_channel=64)
        self.decoder_Dp = decoder_specific(
            in_channel=64, out_channel=64
        )  # specific image decoder 16x16x64 -> 32x32x32
        self.decoder_Dr = decoder_specific(in_channel=64, out_channel=64)
        self.decoder_D = decoder_D(
            in_channel=in_channel, out_channel=64, patch_size=patch_size
        )  # general decoder 32x32x32 -> 64x64x1

        self.phycell = phycell
        self.convcell = convcell

    def forward(self, input, first_timestep=False, decoding=False):
        input = self.encoder_E(input)  # general encoder 64x64x1 -> 32x32x32

        if decoding:  # input=None in decoding phase
            input_phys = None
        else:
            input_phys = self.encoder_Ep(input)
        input_conv = self.encoder_Er(input)

        hidden1, output1 = self.phycell(input_phys, first_timestep)
        hidden2, output2 = self.convcell(input_conv, first_timestep)

        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])

        out_phys = torch.sigmoid(
            self.decoder_D(decoded_Dp)
        )  # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr
        output_image = torch.sigmoid(self.decoder_D(concat))
        return out_phys, hidden1, output_image, out_phys, out_conv


def tensordot(
    a: torch.Tensor, b: torch.Tensor, dim: Union[int, Tuple[List[int], List[int]]]
) -> torch.Tensor:
    """
    Optimized tensordot operation similar to numpy.tensordot for PyTorch tensors.

    Args:
        a (torch.Tensor): First input tensor.
        b (torch.Tensor): Second input tensor.
        dim (int or tuple): Dimensions to contract. If int, number of last dims of a and first dims of b.
                           If tuple, lists of dimensions for a and b to contract.

    Returns:
        torch.Tensor: Result of tensor contraction.
    """
    a = a.contiguous()
    b = b.contiguous()

    if isinstance(dim, int):
        sizea = a.shape
        sizeb = b.shape
        sizea0, sizea1 = sizea[:-dim], sizea[-dim:]
        sizeb0, sizeb1 = sizeb[:dim], sizeb[dim:]
        N = torch.prod(torch.tensor(sizea1))
        assert torch.prod(torch.tensor(sizeb0)) == N, (
            "Contraction dimensions must match"
        )
    else:
        adims, bdims = dim
        adims = [adims] if isinstance(adims, int) else adims
        bdims = [bdims] if isinstance(bdims, int) else bdims

        # Compute dimensions to keep and permute
        adims_ = sorted(set(range(a.dim())) - set(adims))
        bdims_ = sorted(set(range(b.dim())) - set(bdims))

        # Permute and reshape
        a = a.permute(*adims_, *adims).contiguous()
        b = b.permute(*bdims, *bdims_).contiguous()

        sizea = a.shape
        sizeb = b.shape
        sizea0, sizea1 = sizea[: -len(adims)], sizea[-len(adims) :]
        sizeb0, sizeb1 = sizeb[: len(bdims)], sizeb[len(bdims) :]
        N = torch.prod(torch.tensor(sizea1))
        assert torch.prod(torch.tensor(sizeb0)) == N, (
            "Contraction dimensions must match"
        )

    # Reshape for matrix multiplication
    a = a.view(-1, N)
    b = b.view(N, -1)

    # Perform matrix multiplication
    c = torch.matmul(a, b)

    # Reshape to final output
    return c.view(*sizea0, *sizeb1)


def apply_axis_left_dot(x: torch.Tensor, mats: List[torch.Tensor]) -> torch.Tensor:
    """
    Apply matrix multiplication along specified axes iteratively.

    Args:
        x (torch.Tensor): Input tensor of shape [..., *mat_shapes].
        mats (List[torch.Tensor]): List of matrices to apply.

    Returns:
        torch.Tensor: Transformed tensor with same shape as input.
    """
    assert x.dim() == len(mats) + 1, "Input dimension must match number of matrices + 1"
    sizex = x.shape
    k = x.dim() - 1

    for i in range(k):
        x = tensordot(mats[k - i - 1], x, dim=[1, k])

    # Permute and reshape to original shape
    x = x.permute(k, *range(k)).contiguous()
    return x.view(*sizex)


def factorial(n: int) -> int:
    """Compute factorial of n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


class MK(nn.Module):
    """
    Base class for moment kernel transformations.

    Args:
        shape (Tuple[int]): Shape of the kernel.
    """

    def __init__(self, shape: List[int]):
        super().__init__()
        self._size = torch.Size(shape)
        self._dim = len(shape)

        assert self._dim > 0, "Shape must be non-empty"

        # Precompute transformation matrices
        for j, l in enumerate(shape):
            M = np.zeros((l, l))
            for i in range(l):
                M[i] = ((np.arange(l) - (l - 1) // 2) ** i) / factorial(i)

            M_tensor = torch.from_numpy(M)  # Convert NumPy array to tensor
            invM = torch.linalg.inv(M_tensor)  # Compute inverse on tensor

            self.register_buffer(f"M_{j}", M_tensor)  # Register M as buffer
            self.register_buffer(f"invM_{j}", invM)

    @property
    def M(self) -> List[torch.Tensor]:
        """List of transformation matrices."""
        return [self._buffers[f"M_{j}"] for j in range(self._dim)]

    @property
    def invM(self) -> List[torch.Tensor]:
        """List of inverse transformation matrices."""
        return [self._buffers[f"invM_{j}"] for j in range(self._dim)]

    def size(self) -> torch.Size:
        """Return the kernel shape."""
        return self._size

    def dim(self) -> int:
        """Return the number of dimensions."""
        return self._dim

    def _packdim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input tensor for processing.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Reshaped tensor.
        """
        assert x.dim() >= self._dim, "Input dimension must be at least kernel dimension"
        if x.dim() == self._dim:
            x = x.unsqueeze(0)
        return x.view(-1, *x.shape[-self._dim :]).contiguous()


class K2M(MK):
    """
    Convert convolution kernel to moment matrix.

    Args:
        shape (Tuple[int]): Kernel shape.

    Example:
        k2m = K2M((5, 5))
        k = torch.randn(5, 5, dtype=torch.float64)
        m = k2m(k)
    """

    def __init__(self, shape: Tuple[int]):
        super().__init__(shape)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Convert kernel to moment matrix.

        Args:
            k (torch.Tensor): Input kernel tensor of shape [..., *self.shape].

        Returns:
            torch.Tensor: Transformed moment matrix.
        """
        sizek = k.shape
        k = self._packdim(k)
        k = apply_axis_left_dot(k, self.M)
        return k.view(*sizek)
