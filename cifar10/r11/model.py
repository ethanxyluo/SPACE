import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import layer


#### Surrogate function ####
class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors[0]
        grad_x = grad_output.clone()
        grad_x[inputs <= 0.0] = 0
        return grad_x


#### Neurons ####
class BaseNode(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=relu.apply, monitor=False):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        if self.v_reset is None:
            self.v = 0
        else:
            self.v = self.v_reset
        self.surrogate_function = surrogate_function
        # Accumulated voltage (Assuming NO fire for this neuron)
        self.v_acc = 0
        # Accumulated voltage with leaky (Assuming NO fire for this neuron)
        self.v_acc_l = 0
        if monitor:
            self.monitor = {'v': [], 's': []}
        else:
            self.monitor = False

        self.new_grad = None

    def spiking(self):
        spike = self.v - self.v_threshold
        self.v.masked_fill_(spike > 0, self.v_reset)
        spike = self.surrogate_function(spike)

        return spike

    def forward(self, dv: torch.Tensor):
        raise NotImplementedError

    def reset(self):
        if self.v_reset is None:
            self.v = 0
        else:
            self.v = self.v_reset
        if self.monitor:
            self.monitor = {'v': [], 's': []}
        self.v_acc = 0
        self.v_acc_l = 0


class LIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function=relu.apply, fire=True):
        super().__init__(v_threshold, v_reset, surrogate_function)
        self.tau = tau
        self.fire = fire  # If no fire, the voltage threshold of neuron is infinity
        self.new_grad = None

    def forward(self, dv: torch.Tensor):
        self.v += dv
        if self.fire:
            spike = self.spiking()
            self.v_acc += spike
            self.v_acc_l = self.v - \
                ((self.v - self.v_reset) / self.tau) + spike

        self.v = self.v - ((self.v - self.v_reset) / self.tau).detach()

        if self.fire:
            if self.training:
                spike.register_hook(
                    lambda grad: torch.mul(grad, self.new_grad))
            return spike

        return self.v


class IFNode(BaseNode):
    def __init__(self, v_threshold=0.75, v_reset=0.0, surrogate_function=relu.apply):
        super().__init__(v_threshold, v_reset, surrogate_function)

    def forward(self, dv: torch.Tensor):
        self.v += dv
        return self.spiking()


#### Network ####
class ResNet11(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_epoch = 0

        self.cnn11 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.lif11 = nn.Sequential(
            LIFNode(),
            layer.Dropout(0.25)
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.if1 = IFNode()

        self.cnn21 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.lif21 = nn.Sequential(
            LIFNode(),
            layer.Dropout(0.25)
        )
        self.cnn22 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),)
        self.lif2 = nn.Sequential(
            LIFNode(),
            layer.Dropout(0.25)
        )

        self.cnn31 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.lif31 = nn.Sequential(
            LIFNode(),
            layer.Dropout(0.25)
        )
        self.cnn32 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),)
        self.lif3 = nn.Sequential(
            LIFNode(),
            layer.Dropout(0.25)
        )

        self.cnn41 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.lif41 = nn.Sequential(
            LIFNode(),
            layer.Dropout(0.25)
        )
        self.cnn42 = nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),)
        self.lif4 = nn.Sequential(
            LIFNode(),
            layer.Dropout(0.25)
        )

        self.cnn51 = nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.lif51 = nn.Sequential(
            LIFNode(),
            layer.Dropout(0.25)
        )
        self.cnn52 = nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.shortcut4 = nn.Sequential(nn.AvgPool2d(
            kernel_size=(1, 1), stride=(2, 2), padding=(0, 0)))
        self.lif5 = nn.Sequential(
            LIFNode(),
            layer.Dropout(0.25)
        )

        self.fc0 = nn.Linear(512 * 4 * 4, 1024, bias=False)
        self.lif6 = nn.Sequential(
            LIFNode(),
            layer.Dropout(0.25)
        )
        self.fc1 = nn.Linear(1024, 10, bias=False)
        self.lif_out = LIFNode(fire=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(1.0 / n)
                m.weight.data.normal_(0, variance1)

            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]
                variance2 = math.sqrt(1.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)

    def forward(self, x):
        x = self.if1(self.avgpool1(self.lif11(self.cnn11(x))))
        x = self.lif2(self.cnn22(self.lif21(
            self.cnn21(x))) + self.shortcut1(x))
        x = self.lif3(self.cnn32(self.lif31(
            self.cnn31(x))) + self.shortcut2(x))
        x = self.lif4(self.cnn42(self.lif41(
            self.cnn41(x))) + self.shortcut3(x))
        x = self.lif5(self.cnn52(self.lif51(
            self.cnn51(x))) + self.shortcut4(x))

        out = x.view(x.size(0), -1)
        feature_map = x

        out = self.lif_out(self.fc1(self.lif6(self.fc0(out))))

        return out, feature_map

    def reset_(self):
        for item in self.modules():
            if hasattr(item, 'reset'):
                item.reset()


def net_forward(net, img, T, phase='train'):
    fm_list = []
    for _ in range(T - 1):
        # Poisson encoding
        rand_num = torch.rand_like(img).cuda()
        poisson_input = (torch.abs(img) > rand_num).float()
        poisson_input = torch.mul(poisson_input, torch.sign(img))

        fm_list.append(net(poisson_input)[1])

    output, fm_final = net(poisson_input)
    fm_list.append(fm_final)
    fm = sum(fm_list)

    if phase == 'train':
        for m in net.modules():
            if isinstance(m, LIFNode) and m.fire:
                m.v_acc += (m.v_acc < 1e-3).float()
                m.new_grad = (m.v_acc_l > 1e-3).float() + math.log(1 -
                                                                   1 / m.tau) * torch.div(m.v_acc_l, m.v_acc)

    return output, fm
