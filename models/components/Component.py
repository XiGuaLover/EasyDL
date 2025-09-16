from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def wrap(input, flow):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(input.device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(input.device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output


class TrajGRUCell(nn.Module):
    def __init__(self, input_channel, num_filter, h2h_kernel, L, act_type=torch.tanh):
        super(TrajGRUCell, self).__init__()
        self._L = L
        self._act_type = act_type
        self._num_filter = num_filter

        # reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(
            in_channels=input_channel,
            out_channels=num_filter * 3,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )

        # inputs to flow
        self.i2f_conv1 = nn.Conv2d(
            in_channels=input_channel,
            out_channels=32,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2),
            dilation=(1, 1),
        )

        # hidden to flow
        self.h2f_conv1 = nn.Conv2d(
            in_channels=num_filter,
            out_channels=32,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2),
            dilation=(1, 1),
        )

        # generate flow
        self.flows_conv = nn.Conv2d(
            in_channels=32,
            out_channels=L * 2,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2),
        )

        self.ret = nn.Conv2d(
            in_channels=num_filter * L,
            out_channels=num_filter * 3,
            kernel_size=(1, 1),
            stride=1,
        )

    def _flow_generator(self, inputs, states):
        i2f_conv1 = self.i2f_conv1(inputs) if inputs is not None else None
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)

        flows = self.flows_conv(f_conv1)
        flows = torch.split(flows, 2, dim=1)
        return flows

    def forward(self, inputs, states):
        # Input to hidden
        i2h = self.i2h(inputs) if inputs is not None else None
        i2h_slice = (
            torch.split(i2h, self._num_filter, dim=1) if i2h is not None else None
        )

        prev_h = states
        flows = self._flow_generator(inputs, prev_h)
        wrapped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            wrapped_data.append(wrap(prev_h, -flow))
        wrapped_data = torch.cat(wrapped_data, dim=1)
        h2h = self.ret(wrapped_data)
        h2h_slice = torch.split(h2h, self._num_filter, dim=1)
        if i2h_slice is not None:
            reset_gate = torch.sigmoid(i2h_slice[0] + h2h_slice[0])
            update_gate = torch.sigmoid(i2h_slice[1] + h2h_slice[1])
            new_mem = self._act_type(i2h_slice[2] + reset_gate * h2h_slice[2])
        else:
            reset_gate = torch.sigmoid(h2h_slice[0])
            update_gate = torch.sigmoid(h2h_slice[1])
            new_mem = self._act_type(reset_gate * h2h_slice[2])
        next_h = update_gate * prev_h + (1 - update_gate) * new_mem
        return next_h


class ConvGRUCell(nn.Module):
    def __init__(
        self,
        inputFeatureDimensions: int,
        hiddenFeatureDimensions: int,
        kernelSize: Tuple[int, int],
        stride: int = 1,
        bias: bool = True,
    ):
        super(ConvGRUCell, self).__init__()

        self.inputFeatureDimensions: int = inputFeatureDimensions
        self.hiddenFeatureDimensions: int = hiddenFeatureDimensions
        padding: Tuple[int, int] = (
            kernelSize[0] // 2,
            kernelSize[1] // 2,
        )  # Same padding to preserve spatial dimensions

        # Convolutional layers for update and reset gates
        self.conv_gates = nn.Conv2d(
            in_channels=inputFeatureDimensions + hiddenFeatureDimensions,
            # For update (z) and reset (r) gates
            out_channels=2 * hiddenFeatureDimensions,
            kernel_size=kernelSize,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Convolutional layer for candidate hidden state
        self.conv_hidden = nn.Conv2d(
            in_channels=inputFeatureDimensions + hiddenFeatureDimensions,
            out_channels=hiddenFeatureDimensions,
            kernel_size=kernelSize,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputTensor: torch.Tensor, hCur: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ConvGRU cell.

        Parameters:
        - inputTensor: Input tensor of shape (batch, input_dim, height, width)
        - hCur: Current hidden state of shape (batch, hidden_dim, height, width)

        Returns:
        - hNext: Next hidden state
        """
        # Concatenate input and current hidden state
        combined = torch.cat([inputTensor, hCur], dim=1)

        # Compute update and reset gates
        zr_out = self.conv_gates(combined)
        z, r = torch.split(zr_out, self.hiddenFeatureDimensions, dim=1)
        updateGate = torch.sigmoid(z)  # Update gate
        resetGate = torch.sigmoid(r)  # Reset gate

        # Compute candidate hidden state
        combined_reset = torch.cat([inputTensor, resetGate * hCur], dim=1)
        hCandidate = torch.tanh(self.conv_hidden(combined_reset))

        # Update hidden state
        hNext = (1 - updateGate) * hCur + updateGate * hCandidate

        return hNext


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        inputFeatureDimensions: int,
        hiddenFeatureDimensions: int,
        stride: int,
        kernelSize: Tuple[int, int],
        bias: bool = True,
    ):
        super(ConvLSTMCell, self).__init__()

        self.inputFeatureDimensions: int = inputFeatureDimensions
        self.hiddenFeatureDimensions: int = hiddenFeatureDimensions
        padding: Tuple[int, int] = (
            kernelSize[0] // 2,
            kernelSize[1] // 2,
        )  # Same padding to preserve spatial dimensions

        # Convolution for all gates at once
        self.conv = nn.Conv2d(
            in_channels=inputFeatureDimensions + hiddenFeatureDimensions,
            # For input, forget, cell, output gates
            out_channels=4 * hiddenFeatureDimensions,
            stride=stride,
            kernel_size=kernelSize,
            padding=padding,
            bias=bias,
        )

    def forward(
        self, inputTensor: torch.Tensor, curState: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ConvLSTM cell.

        Parameters:
        - input_tensor: Input tensor of shape (batch, input_dim, height, width)
        - cur_state: Tuple of (h_cur, c_cur), hidden and cell states

        Returns:
        - h_next, c_next: Next hidden and cell states
        """
        hCur, cCur = curState

        # Concatenate input and hidden state
        combined = torch.cat([inputTensor, hCur], dim=1)
        # Combine current spatial input with temporal context from the hidden state.

        # Convolutional operation
        conv_out = self.conv(combined)
        i, f, g, o = torch.split(conv_out, self.hiddenFeatureDimensions, dim=1)
        # each gate shape: (batch, hidden_dim, height, width)

        # Gate activations
        inputGate = torch.sigmoid(i)
        forgetGate = torch.sigmoid(f)
        cellGate = torch.tanh(g)
        outputGate = torch.sigmoid(o)

        # Update cell and hidden states
        cNext = forgetGate * cCur + inputGate * cellGate
        hNext = outputGate * torch.tanh(cNext)

        return hNext, cNext


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_hidden: int,
        height: int,
        width: int,
        kernelSize: Tuple[int, int],
        stride: int,
        layer_norm: bool,
    ) -> None:
        super(SpatioTemporalLSTMCell, self).__init__()
        self.num_hidden: int = num_hidden
        padding: Tuple[int, int] = (
            kernelSize[0] // 2,
            kernelSize[1] // 2,
        )
        self._forget_bias: float = 1.0

        def create_conv_sequential(
            in_channels: int, out_channels: int, norm_shape: Optional[List[int]] = None
        ) -> nn.Sequential:
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernelSize,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            ]
            if layer_norm and norm_shape:
                layers.append(nn.LayerNorm(norm_shape))
            return nn.Sequential(*layers)

        self.conv_x: nn.Sequential = create_conv_sequential(
            in_channel, num_hidden * 7, [num_hidden * 7, height, width]
        )
        self.conv_h: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, height, width]
        )
        self.conv_m: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 3, [num_hidden * 3, height, width]
        )
        self.conv_o: nn.Sequential = create_conv_sequential(
            num_hidden * 2, num_hidden, [num_hidden, height, width]
        )
        self.conv_last: nn.Conv2d = nn.Conv2d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(
        self, x_t: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor, m_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the Spatio-Temporal LSTM Cell.

        Args:
            x_t: Input tensor
            h_t: Hidden state tensor
            c_t: Cell state tensor
            m_t: Memory state tensor

        Returns:
            Tuple of (new hidden state, new cell state, new memory state)
        """
        # Process inputs through convolutional layers
        x_concat: torch.Tensor = self.conv_x(x_t)
        h_concat: torch.Tensor = self.conv_h(h_t)
        m_concat: torch.Tensor = self.conv_m(m_t)

        # Split convolutional outputs
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1
        )
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        # Compute gates
        i_t: torch.Tensor = torch.sigmoid(i_x + i_h)
        f_t: torch.Tensor = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t: torch.Tensor = torch.tanh(g_x + g_h)
        c_new: torch.Tensor = f_t * c_t + i_t * g_t

        i_t_prime: torch.Tensor = torch.sigmoid(i_x_prime + i_m)
        f_t_prime: torch.Tensor = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime: torch.Tensor = torch.tanh(g_x_prime + g_m)
        m_new: torch.Tensor = f_t_prime * m_t + i_t_prime * g_t_prime

        # Combine memory and compute output
        mem: torch.Tensor = torch.cat((c_new, m_new), 1)
        o_t: torch.Tensor = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new: torch.Tensor = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class SpatioTemporalLSTMCellV2(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_hidden: int,
        height: int,
        width: int,
        kernelSize: Tuple[int, int],
        stride: int,
        layer_norm: bool,
    ) -> None:
        super(SpatioTemporalLSTMCellV2, self).__init__()
        self.num_hidden: int = num_hidden
        padding: Tuple[int, int] = (
            kernelSize[0] // 2,
            kernelSize[1] // 2,
        )
        self._forget_bias: float = 1.0

        def create_conv_sequential(
            in_channels: int, out_channels: int, norm_shape: Optional[List[int]] = None
        ) -> nn.Sequential:
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernelSize,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            ]
            if layer_norm and norm_shape:
                layers.append(nn.LayerNorm(norm_shape))
            return nn.Sequential(*layers)

        self.conv_x: nn.Sequential = create_conv_sequential(
            in_channel, num_hidden * 7, [num_hidden * 7, height, width]
        )
        self.conv_h: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, height, width]
        )
        self.conv_m: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 3, [num_hidden * 3, height, width]
        )
        self.conv_o: nn.Sequential = create_conv_sequential(
            num_hidden * 2, num_hidden, [num_hidden, height, width]
        )
        self.conv_last: nn.Conv2d = nn.Conv2d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(
        self, x_t: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor, m_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the Spatio-Temporal LSTM Cell.

        Args:
            x_t: Input tensor
            h_t: Hidden state tensor
            c_t: Cell state tensor
            m_t: Memory state tensor

        Returns:
            Tuple of (new hidden state, new cell state, new memory state)
        """
        # Process inputs through convolutional layers
        x_concat: torch.Tensor = self.conv_x(x_t)
        h_concat: torch.Tensor = self.conv_h(h_t)
        m_concat: torch.Tensor = self.conv_m(m_t)

        # Split convolutional outputs
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1
        )
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        # Compute gates
        i_t: torch.Tensor = torch.sigmoid(i_x + i_h)
        f_t: torch.Tensor = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t: torch.Tensor = torch.tanh(g_x + g_h)

        delta_c: torch.Tensor = i_t * g_t
        c_new: torch.Tensor = f_t * c_t + delta_c

        i_t_prime: torch.Tensor = torch.sigmoid(i_x_prime + i_m)
        f_t_prime: torch.Tensor = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime: torch.Tensor = torch.tanh(g_x_prime + g_m)

        delta_m: torch.Tensor = i_t_prime * g_t_prime
        m_new: torch.Tensor = f_t_prime * m_t + delta_m

        # Combine memory and compute output
        mem: torch.Tensor = torch.cat((c_new, m_new), 1)
        o_t: torch.Tensor = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new: torch.Tensor = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m


class CausalLSTMCell(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_hidden: int,
        height: int,
        width: int,
        kernelSize: Tuple[int, int],
        stride: int,
        layer_norm: bool,
    ):
        super(CausalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        padding: Tuple[int, int] = (
            kernelSize[0] // 2,
            kernelSize[1] // 2,
        )
        self._forget_bias: float = 1.0

        def create_conv_sequential(
            in_channels: int, out_channels: int, norm_shape: Optional[List[int]] = None
        ) -> nn.Sequential:
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernelSize,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            ]
            if layer_norm and norm_shape:
                layers.append(nn.LayerNorm(norm_shape))
            return nn.Sequential(*layers)

        self.conv_x: nn.Sequential = create_conv_sequential(
            in_channel, num_hidden * 7, [num_hidden * 7, height, width]
        )
        self.conv_h: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, height, width]
        )
        self.conv_c: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 3, [num_hidden * 3, height, width]
        )
        self.conv_m: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 3, [num_hidden * 3, height, width]
        )
        self.conv_o: nn.Sequential = create_conv_sequential(
            num_hidden * 2, num_hidden, [num_hidden, height, width]
        )
        self.conv_c2m: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, height, width]
        )
        self.conv_om: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden, [num_hidden, height, width]
        )
        self.conv_last: nn.Conv2d = nn.Conv2d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(
        self, x_t: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor, m_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_concat: torch.Tensor = self.conv_x(x_t)
        h_concat: torch.Tensor = self.conv_h(h_t)
        c_concat: torch.Tensor = self.conv_c(c_t)
        m_concat: torch.Tensor = self.conv_m(m_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1
        )
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, m_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)

        i_t: torch.Tensor = torch.sigmoid(i_x + i_h + i_c)
        f_t: torch.Tensor = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        g_t: torch.Tensor = torch.tanh(g_x + g_h + g_c)

        c_new: torch.Tensor = f_t * c_t + i_t * g_t

        c2m: torch.Tensor = self.conv_c2m(c_new)
        i_c, g_c, f_c, o_c = torch.split(c2m, self.num_hidden, dim=1)

        i_t_prime: torch.Tensor = torch.sigmoid(i_x_prime + i_m + i_c)
        f_t_prime: torch.Tensor = torch.sigmoid(
            f_x_prime + f_m + f_c + self._forget_bias
        )
        g_t_prime: torch.Tensor = torch.tanh(g_x_prime + g_c)

        m_new: torch.Tensor = f_t_prime * torch.tanh(m_m) + i_t_prime * g_t_prime
        o_m: torch.Tensor = self.conv_om(m_new)

        o_t: torch.Tensor = torch.tanh(o_x + o_h + o_c + o_m)
        mem: torch.Tensor = torch.cat((c_new, m_new), 1)
        h_new: torch.Tensor = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class GradientHighwayUnit(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_hidden: int,
        height: int,
        width: int,
        kernelSize: Tuple[int, int],
        stride: int,
        layer_norm: bool,
        initializer: float = 0.001,
    ):
        super(GradientHighwayUnit, self).__init__()

        self.num_hidden: int = num_hidden
        padding: Tuple[int, int] = (
            kernelSize[0] // 2,
            kernelSize[1] // 2,
        )
        self.layer_norm = layer_norm

        def create_conv_sequential(
            in_channels: int, out_channels: int, norm_shape: Optional[List[int]] = None
        ) -> nn.Sequential:
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernelSize,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            ]
            if layer_norm and norm_shape:
                layers.append(nn.LayerNorm(norm_shape))
            return nn.Sequential(*layers)

        self.z_concat: nn.Sequential = create_conv_sequential(
            in_channel, num_hidden * 2, [num_hidden, height, width]
        )
        self.x_concat: nn.Sequential = create_conv_sequential(
            in_channel, num_hidden * 2, [num_hidden, height, width]
        )

        if initializer != -1:
            self.initializer = initializer
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            nn.init.uniform_(m.weight, -self.initializer, self.initializer)

    def _init_state(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        if z is None:
            z = self._init_state(x)
        z_concat = self.z_concat(z)
        x_concat = self.x_concat(x)

        gates = x_concat + z_concat
        p, u = torch.split(gates, self.num_hidden, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z
        return z_new


class Eidetic3DLSTMCell(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_hidden: int,
        window_length: int,
        height: int,
        width: int,
        kernelSize: Tuple[int, int, int],
        stride: int,
        layer_norm: bool,
    ) -> None:
        super(Eidetic3DLSTMCell, self).__init__()
        self.num_hidden: int = num_hidden
        padding: Tuple[int, int] = (
            kernelSize[0] // 2,
            kernelSize[1] // 2,
            kernelSize[2] // 2,
        )
        self._forget_bias: float = 1.0

        def create_conv_sequential(
            in_channels: int, out_channels: int, norm_shape: Optional[List[int]] = None
        ) -> nn.Sequential:
            layers = [
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernelSize,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            ]
            if layer_norm and norm_shape:
                layers.append(nn.LayerNorm(norm_shape))
            return nn.Sequential(*layers)

        self.conv_x: nn.Sequential = create_conv_sequential(
            in_channel, num_hidden * 7, [num_hidden * 7, window_length, height, width]
        )
        self.conv_h: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, window_length, height, width]
        )
        self.conv_global_memory: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, window_length, height, width]
        )
        self.conv_cell: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden, [num_hidden, window_length, height, width]
        )
        self.conv_global_memory_out: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden, [num_hidden, window_length, height, width]
        )
        self.conv_memory: nn.Conv3d = nn.Conv3d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False
        )

        self._norm_c_t = nn.LayerNorm([num_hidden, window_length, height, width])

    def _attn(
        self, in_query: torch.Tensor, in_keys: torch.Tensor, in_values: torch.Tensor
    ):
        batch, num_channels, _, width, height = in_query.shape
        query = in_query.reshape(batch, -1, num_channels)
        keys = in_keys.reshape(batch, -1, num_channels)
        values = in_values.reshape(batch, -1, num_channels)
        attn = torch.einsum("bxc,byc->bxy", query, keys)
        attn = torch.softmax(attn, dim=2)
        attn = torch.einsum("bxy,byc->bxc", attn, values)
        return attn.reshape(batch, num_channels, -1, width, height)

    def forward(
        self,
        x_t: torch.Tensor,
        h_t: torch.Tensor,
        c_t: torch.Tensor,
        global_memory: torch.Tensor,
        eidetic_cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Hidden state convolutions
        new_hidden = self.conv_h(h_t)
        i_h, g_h, r_h, o_h = torch.split(new_hidden, self.num_hidden, dim=1)

        # Input convolutions
        new_inputs = self.conv_x(x_t)
        i_x, g_x, r_x, o_x, temp_i_x, temp_g_x, temp_f_x = torch.split(
            new_inputs, self.num_hidden, dim=1
        )

        # Input, gate, and reset computations
        i_t = torch.sigmoid(i_x + i_h)
        r_t = torch.sigmoid(r_x + r_h)
        g_t = torch.tanh(g_x + g_h)

        # Cell state update with attention
        new_cell = c_t + self._attn(r_t, eidetic_cell, eidetic_cell)
        new_cell = self._norm_c_t(new_cell) + i_t * g_t

        # Global memory update
        new_global_memory = self.conv_global_memory(global_memory)
        i_m, f_m, g_m, m_m = torch.split(new_global_memory, self.num_hidden, dim=1)

        temp_i_t = torch.sigmoid(temp_i_x + i_m)
        temp_f_t = torch.sigmoid(temp_f_x + f_m + self._forget_bias)
        temp_g_t = torch.tanh(temp_g_x + g_m)
        new_global_memory = temp_f_t * torch.tanh(m_m) + temp_i_t * temp_g_t

        # Output gate computations
        o_c = self.conv_cell(new_cell)
        o_m = self.conv_global_memory_out(new_global_memory)
        output_gate = torch.tanh(o_x + o_h + o_c + o_m)

        # Final memory and output
        memory = torch.cat((new_cell, new_global_memory), dim=1)
        memory = self.conv_memory(memory)
        output = torch.tanh(memory) * torch.sigmoid(output_gate)

        return output, new_cell, global_memory


class MIMBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_hidden: int,
        height: int,
        width: int,
        kernelSize: Tuple[int, int],
        stride: int,
        layer_norm: bool,
    ) -> None:
        super(MIMBlock, self).__init__()
        self.num_hidden: int = num_hidden
        padding: Tuple[int, int] = (
            kernelSize[0] // 2,
            kernelSize[1] // 2,
        )
        self._forget_bias: float = 1.0

        def create_conv_sequential(
            in_channels: int, out_channels: int, norm_shape: Optional[List[int]] = None
        ) -> nn.Sequential:
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernelSize,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            ]
            if layer_norm and norm_shape:
                layers.append(nn.LayerNorm(norm_shape))
            return nn.Sequential(*layers)

        self.conv_t_cc: nn.Sequential = create_conv_sequential(
            in_channel, num_hidden * 3, [num_hidden * 3, height, width]
        )
        self.conv_s_cc: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, height, width]
        )
        self.conv_x_cc: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, height, width]
        )
        self.conv_h_concat: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, height, width]
        )
        self.conv_x_concat: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, height, width]
        )

        self.conv_last: nn.Conv2d = nn.Conv2d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.convlstm_c = None
        self.ct_weight = nn.Parameter(torch.zeros(num_hidden * 2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))

    def MIMS(
        self, x: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_t = torch.zeros_like(x) if h_t is None else h_t
        c_t = torch.zeros_like(x) if c_t is None else c_t

        h_concat: torch.Tensor = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation: torch.Tensor = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new

    def forward(
        self,
        x: torch.Tensor,
        diff_h: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.zeros_like(x) if h is None else h
        c = torch.zeros_like(x) if c is None else c
        m = torch.zeros_like(x) if m is None else m
        diff_h = torch.zeros_like(x) if diff_h is None else diff_h

        t_cc = self.conv_t_cc(h)
        s_cc = self.conv_s_cc(m)
        x_cc = self.conv_x_cc(x)

        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_

        c, self.convlstm_c = self.MIMS(
            diff_h,
            c,
            self.convlstm_c if self.convlstm_c is None else self.convlstm_c.detach(),
        )

        new_c = c + i * g
        cell = torch.cat((new_c, new_m), 1)
        new_h = o * torch.tanh(self.conv_last(cell))

        return new_h, new_c, new_m


class MIMN(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_hidden: int,
        height: int,
        width: int,
        kernelSize: Tuple[int, int],
        stride: int,
        layer_norm: bool,
    ) -> None:
        super(MIMN, self).__init__()
        self.num_hidden: int = num_hidden
        padding: Tuple[int, int] = (
            kernelSize[0] // 2,
            kernelSize[1] // 2,
        )
        self._forget_bias: float = 1.0

        def create_conv_sequential(
            in_channels: int, out_channels: int, norm_shape: Optional[List[int]] = None
        ) -> nn.Sequential:
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernelSize,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            ]
            if layer_norm and norm_shape:
                layers.append(nn.LayerNorm(norm_shape))
            return nn.Sequential(*layers)

        self.conv_h_concat: nn.Sequential = create_conv_sequential(
            in_channel, num_hidden * 4, [num_hidden * 4, height, width]
        )
        self.conv_x_concat: nn.Sequential = create_conv_sequential(
            num_hidden, num_hidden * 4, [num_hidden * 4, height, width]
        )

        self.conv_last: nn.Conv2d = nn.Conv2d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.ct_weight = nn.Parameter(torch.zeros(num_hidden * 2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))

    def forward(
        self, x: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_t = torch.zeros_like(x) if h_t is None else h_t
        c_t = torch.zeros_like(x) if c_t is None else c_t

        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new


class SelfAttentionMemoryModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.layer_qh = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_kh = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_vh = nn.Conv2d(input_dim, hidden_dim, 1)

        self.layer_km = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_vm = nn.Conv2d(input_dim, hidden_dim, 1)

        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(
        self, h: torch.Tensor, m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channel, H, W = h.shape

        K_h = self.layer_kh(h)
        Q_h = self.layer_qh(h)
        V_h = self.layer_vh(h)

        K_h = K_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W).transpose(1, 2)
        V_h = V_h.view(batch_size, self.hidden_dim, H * W)

        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)  # batch_size, H*W, H*W
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))

        K_m = self.layer_km(m)
        V_m = self.layer_vm(m)

        K_m = K_m.view(batch_size, self.hidden_dim, H * W)
        V_m = V_m.view(batch_size, self.hidden_dim, H * W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))

        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)

        ## Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim=1))
        mo, mg, mi = torch.chunk(combined, chunks=3, dim=1)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m


class SelfAttentionConvLstmCell(nn.Module):
    def __init__(
        self,
        inputFeatureDimensions: int,
        hiddenFeatureDimensions: int,
        height: int,
        width: int,
        stride: int,
        kernelSize: Tuple[int, int],
        layer_norm: bool,
        bias: bool = True,
    ):
        super().__init__()
        self.input_channels = inputFeatureDimensions
        self.hidden_dim = hiddenFeatureDimensions
        padding: Tuple[int, int] = (
            kernelSize[0] // 2,
            kernelSize[1] // 2,
        )
        self.attention_layer = SelfAttentionMemoryModule(
            hiddenFeatureDimensions, hiddenFeatureDimensions
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels + self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=kernelSize,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.GroupNorm(4 * self.hidden_dim, 4 * self.hidden_dim),
        )
        if layer_norm:
            self.conv.append(nn.LayerNorm([hiddenFeatureDimensions * 4, height, width]))

    def forward(
        self,
        inputTensor: torch.Tensor,
        curState: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h, c, m = curState
        combined = torch.cat([inputTensor, h], dim=1)
        combined_conv = self.conv(combined)
        i, f, g, o = torch.chunk(combined_conv, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = torch.mul(f, c) + torch.mul(i, g)
        h_next = torch.mul(o, torch.tanh(c_next))

        # Self-Attention
        h_next, m_next = self.attention_layer(h_next, m)

        return h_next, c_next, m_next


class CubicConvLSTMCell(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_hidden: int,
        kernelSizeX: Tuple[int, int] = (3, 3),
        kernelSizeY: Tuple[int, int] = (1, 1),
        kernelSizeZ: Tuple[int, int] = (5, 5),
        stride: int = 1,
        forget_bias: float = 1.0,
        bias: bool = True,
    ) -> None:
        super(CubicConvLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.forget_bias = forget_bias

        paddingX: Tuple[int, int] = (
            kernelSizeX[0] // 2,
            kernelSizeX[1] // 2,
        )
        paddingY: Tuple[int, int] = (
            kernelSizeY[0] // 2,
            kernelSizeY[1] // 2,
        )
        paddingZ: Tuple[int, int] = (
            kernelSizeZ[0] // 2,
            kernelSizeZ[1] // 2,
        )

        # Spatial convolution for gates (y-direction)
        self.conv_y = nn.Conv2d(
            in_channels=in_channel + num_hidden * 2,  # inputs + h_x + h_y
            out_channels=num_hidden * 4,  # i_y, j_y, f_y, o_y
            kernel_size=kernelSizeX,
            stride=stride,
            padding=paddingX,
            bias=bias,
        )

        # Temporal convolution for gates (x-direction)
        self.conv_x = nn.Conv2d(
            in_channels=in_channel + num_hidden * 2,  # inputs + h_x + h_y
            out_channels=num_hidden * 4,  # i_x, j_x, f_x, o_x
            kernel_size=kernelSizeY,
            stride=stride,
            padding=paddingY,
            bias=bias,
        )

        # Output convolution
        self.conv_output = nn.Conv2d(
            in_channels=num_hidden * 2,  # new_h_x + new_h_y
            out_channels=num_hidden,
            kernel_size=kernelSizeZ,
            stride=stride,
            padding=paddingZ,
            bias=bias,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        c_x: torch.Tensor,
        h_x: torch.Tensor,
        c_y: torch.Tensor,
        h_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = inputs.shape

        # Initialize states if None
        def initState(state: torch.Tensor):
            return (
                torch.zeros([batch_size, self.num_hidden, height, width]).to(
                    inputs.device
                )
                if state is None
                else state
            )

        c_x = initState(c_x)
        h_x = torch.zeros_like(c_x) if h_x is None else h_x
        c_y = torch.zeros_like(c_x) if c_y is None else c_y
        h_y = torch.zeros_like(c_x) if h_y is None else h_y

        inputs_h = torch.cat([inputs, h_x, h_y], dim=1)

        # Spatial gates (y-direction)
        gates_y = self.conv_y(inputs_h)
        i_y, j_y, f_y, o_y = torch.chunk(gates_y, 4, dim=1)

        # Compute new cell and hidden states (y-direction)
        new_c_y = c_y * torch.sigmoid(f_y + self.forget_bias) + torch.sigmoid(
            i_y
        ) * torch.tanh(j_y)
        new_h_y = torch.tanh(new_c_y) * torch.sigmoid(o_y)

        # Temporal gates (x-direction)
        gates_x = self.conv_x(inputs_h)
        i_x, j_x, f_x, o_x = torch.chunk(gates_x, 4, dim=1)

        # Compute new cell and hidden states (x-direction)
        new_c_x = c_x * torch.sigmoid(f_x + self.forget_bias) + torch.sigmoid(
            i_x
        ) * torch.tanh(j_x)
        new_h_x = torch.tanh(new_c_x) * torch.sigmoid(o_x)

        # Output
        new_h = self.conv_output(torch.cat([new_h_x, new_h_y], dim=1))

        # Return new hidden state and states
        return (new_h, new_c_x, new_h_x, new_c_y, new_h_y)
