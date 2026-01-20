"""
模型定义
Model Definitions

ConvLSTM-UNet: 用于时序线粒体分割
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell

    输入: (B, C, H, W)
    输出: (h, c) 各为 (B, hidden_dim, H, W)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # 合并输入和隐藏状态，一次卷积计算4个门
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # 分割为4个门
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        o = torch.sigmoid(o)  # 输出门
        g = torch.tanh(g)  # 候选记忆

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    """
    双向ConvLSTM

    输入: (B, T, C, H, W)
    输出: (B, T, hidden_dim*2, H, W) if bidirectional else (B, T, hidden_dim, H, W)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.forward_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        if bidirectional:
            self.backward_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        device = x.device

        # 前向传播
        h_f, c_f = self.forward_cell.init_hidden(B, H, W, device)
        forward_outputs = []
        for t in range(T):
            h_f, c_f = self.forward_cell(x[:, t], (h_f, c_f))
            forward_outputs.append(h_f)

        if self.bidirectional:
            # 反向传播
            h_b, c_b = self.backward_cell.init_hidden(B, H, W, device)
            backward_outputs = []
            for t in range(T - 1, -1, -1):
                h_b, c_b = self.backward_cell(x[:, t], (h_b, c_b))
                backward_outputs.insert(0, h_b)

            # 拼接前向和反向
            outputs = [torch.cat([f, b], dim=1) for f, b in zip(forward_outputs, backward_outputs)]
        else:
            outputs = forward_outputs

        return torch.stack(outputs, dim=1)


class DoubleConv(nn.Module):
    """
    双层卷积块: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvLSTMUNet(nn.Module):
    """
    ConvLSTM-UNet: 用于时序分割

    架构:
    - Encoder: 每帧独立编码（共享权重）
    - ConvLSTM: 在bottleneck处理时序依赖
    - Decoder: 每帧独立解码（带skip connection）

    输入: (B, T, 1, H, W)
    输出: (B, T, num_classes, H, W)
    """

    def __init__(self, in_channels=1, num_classes=4, base_ch=32, lstm_hidden=64, bidirectional=True):
        super().__init__()

        self.num_classes = num_classes

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

        # ConvLSTM at bottleneck
        lstm_input_dim = base_ch * 8
        self.convlstm = ConvLSTM(lstm_input_dim, lstm_hidden, kernel_size=3, bidirectional=bidirectional)
        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden

        # Decoder
        self.up3 = nn.ConvTranspose2d(lstm_output_dim, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        # Output
        self.out = nn.Conv2d(base_ch, num_classes, 1)

    def encode_frame(self, x):
        """编码单帧"""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def decode_frame(self, lstm_feat, e1, e2, e3):
        """解码单帧"""
        d3 = self.up3(lstm_feat)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        return: (B, T, num_classes, H, W)
        """
        B, T, C, H, W = x.shape

        # 编码每帧
        encoder_features = []
        for t in range(T):
            e1, e2, e3, e4 = self.encode_frame(x[:, t])
            encoder_features.append((e1, e2, e3, e4))

        # 提取bottleneck特征序列
        e4_seq = torch.stack([f[3] for f in encoder_features], dim=1)  # (B, T, C, H, W)

        # ConvLSTM处理时序
        lstm_out = self.convlstm(e4_seq)  # (B, T, lstm_out_dim, H, W)

        # 解码每帧
        outputs = []
        for t in range(T):
            e1, e2, e3, _ = encoder_features[t]
            out = self.decode_frame(lstm_out[:, t], e1, e2, e3)
            outputs.append(out)

        return torch.stack(outputs, dim=1)


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """测试模型"""
    print("测试ConvLSTM-UNet...")

    model = ConvLSTMUNet(
        in_channels=1,
        num_classes=4,
        base_ch=32,
        lstm_hidden=64,
        bidirectional=True
    )

    # 测试输入
    x = torch.randn(2, 5, 1, 256, 256)

    with torch.no_grad():
        y = model(x)

    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {y.shape}")
    print(f"  参数量: {count_parameters(model) / 1e6:.2f}M")

    assert y.shape == (2, 5, 4, 256, 256), "输出形状不正确"
    print("  ✓ 测试通过")


if __name__ == "__main__":
    test_model()