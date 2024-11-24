#===============树突神经元相关内容======================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from spikingjelly.clock_driven import surrogate


@torch.jit.script
def gaussian(
    x: torch.Tensor, mean: float, var: float, eps: float=1e-5
) -> torch.Tensor:
    x = x - mean
    var = var + eps
    return torch.exp(-0.5 * (x**2) / var) / (2 * torch.pi * var) ** 0.5


@torch.jit.script
def mexican_hat(
    x: torch.Tensor, mean: float = 0., var: float = 0.75, eps:float=1e-5
):
    x = x - mean
    var = var + eps
    return (1. - (x**2) / var) * gaussian(x, 0., var, eps=0.) / var


def dendritic_spiking_neuron_forward(
    B: int,
    x_seq: torch.Tensor, # shape = [T, N, C, *]
    alpha: torch.Tensor, # shape = [1]
    k: torch.Tensor, # shape = [1, B, 1] or [C//B, B, L]
    f_dact: Callable,
    beta: torch.Tensor, # shape = [1]
    v_th: torch.Tensor, # shape = [1]
    v_reset: torch.Tensor, # shape = [1]
    sg: Callable = surrogate.ATan(),
):
    """Describe the computational logic of a dendritic spiking neuron layer.
    """
    out_shape = list(x_seq.shape)
    x_seq = x_seq.unsqueeze(-1).flatten(3)  # shape = [T, N, C, L]
    T, N, C, L = x_seq.shape
    out_shape[2] = C // B # [T, N, C//B, *]

    x_seq = x_seq.view(T, N, C // B, B, L)

    v_d = torch.zeros_like(x_seq[0])
    h = torch.zeros([N, C // B, L], device=x_seq.device, dtype=x_seq.dtype)
    spike_seq = torch.empty(
        [T, N, C // B, L], device=x_seq.device, dtype=x_seq.dtype
    )
    for t in range(T):
        # dendritic compartment
        v_d = alpha * v_d + x_seq[t]
        # dendrite-to-soma forwarding
        y = k * f_dact(v_d) # shape = [N, C // B, B, L]
        y = y.sum(3) # shape = [N, C // B, L]
        # soma: LIF
        h = beta * h + y
        spike_seq[t] = sg(h - v_th)
        spike = spike_seq[t]
        h = (1. - spike) * h + spike * v_reset

    return spike_seq.view(out_shape)


class DendSN(nn.Module):
    def __init__(
        self, f_dact: str, n_compartment: int, alpha: float=0.5,
        fs_learnable: bool=True, fs_normalize: bool=True, beta: float=0.5,
        v_th: float=1., v_reset: float=0., decay_somatic_input: bool=False,
        surrogate_function: Callable=surrogate.Sigmoid()
    ):
        """Dendritic spiking neuron model.

        Args:
            f_dact (str): dendritic activation function. Can be "mexican_hat"
                or "leaky_relu".   非线性函数
            n_compartment (int): number of dendritic compartments for each neuron.
                Denoted as B in the paper.  隔间个数
            alpha (float, optional): the temporal decay factor of dendritic
                compartments. Defaults to 0.5.   衰减因子
            fs_learnable (bool, optional): whether the dendritic strength vector
                `k` can be learned through BP. Defaults to True.
            fs_normalize (bool, optional): whether to normalize the dendritic
                strength vector `k` using `softmax`. Defaults to True.
            beta (float, optional): the temporal decay factor of soma.   #soma衰减系数
                LIF soma => beta should be in [0, 1)
                IF soma => beta equals 1
                Defaults to 0.5.
            v_th (float, optional): the somatic firing threshold. Defaults to 1.
            v_reset (float, optional): the somatic reset potential. Defaults to 0.
            decay_somatic_input (bool, optional): whether to decay the dend2soma
                input using the factor `1-beta`. Notice that if IF soma is adopted,
                this flag will be hard-coded to False. Defaults to False.
            surrogate_function (Callable, optional): the firing function with
                surrogate gradient. Defaults to
                spikingjelly.activation_based.surrogate.Sigmoid().

        Raises:
            NotImplementedError: if `f_dact` is not currently supported. See the
                `supported_f_dact` property for details.
        """
        super().__init__()
        if f_dact not in self.supported_f_dact:
            raise NotImplementedError(
                f"`f_dact` should be in {self.supported_f_dact}"
            )
        self._f_dact = f_dact

        # dendritic parameters
        self._fs_learnable = fs_learnable
        self.fs_normalize = fs_normalize
        # k.shape = [C//B, B, L],
        #   or any shape with length 3 which can be broadcasted to [C//B, B, L],
        #   e.g. [1, B, 1]
        self.k = nn.Parameter(
            torch.ones([1, n_compartment, 1]), requires_grad=fs_learnable
        )
        self.n_compartment = n_compartment
        self._alpha = alpha

        # somatic parameters
        self.decay_somatic_input = (
            decay_somatic_input if beta < 1. and beta >= 0. else False
        )
        self.beta = beta
        self.v_th = v_th
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function

        # the factor for each time step
        self.alpha_matrix = None
        self.T = None

    @property
    def supported_f_dact(self):
        return ["mexican_hat", "leaky_relu",]

    # read-only
    @property
    def f_dact(self):
        return self._f_dact

    @property
    def fs_learnable(self):
        return self._fs_learnable

    @fs_learnable.setter
    def fs_learnable(self, val: bool):
        self.k.requires_grad = val
        self._fs_learnable = val

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val: float):
        self._alpha = val
        if self.alpha_matrix is not None:
            self.alpha_matrix = self.get_alpha_matrix(
                val, self.alpha_matrix.shape[0],
                self.alpha_matrix.device, self.alpha_matrix.dtype
            )

    @property
    def forward_strength(self):
        if self.fs_normalize:
            return torch.softmax(self.k, dim=1)
        else:
            return self.k

    @staticmethod
    @torch.jit.script
    def dendrite_mh_forward(
        alpha_matrix: torch.Tensor, x_seq: torch.Tensor, fs: torch.Tensor,
        T:int,
    ):
        # alpha_matrix.shape = [T, T]
        wa = alpha_matrix.view(T, T, 1, 1, 1, 1)
        # x_seq.shape = [T, N, C // B, B, L]
        pre_act_seq = wa * x_seq # shape = [T, T, N, C // B, B, L]
        pre_act_seq = pre_act_seq.sum(1) # shape = [T, N, C // B, B, L]
        # fs.shape = [C//B, B, L] or [1, B, 1]
        y_seq = fs * 3. * mexican_hat(pre_act_seq) # fda = 3 * mexican_hat
        return y_seq.sum(3), pre_act_seq # shape = [T, N, C // B, L]

    @staticmethod
    @torch.jit.script
    def dendritic_lrelu_forward(
        alpha_matrix: torch.Tensor, x_seq: torch.Tensor, fs: torch.Tensor,
        T:int,
    ):
        wa = alpha_matrix.view(T, T, 1, 1, 1, 1)
        pre_act_seq = wa * x_seq
        pre_act_seq = pre_act_seq.sum(1)
        y_seq = fs * F.leaky_relu(pre_act_seq)
        return y_seq.sum(3), pre_act_seq

    @staticmethod
    def get_alpha_matrix(alpha: float, T: int, device: str, dtype: str):
        # alpha_matrix[i, j]: the coefficient of x[t=j] for v[t=i]
        alpha_matrix = torch.zeros(
            [T, T], device=device, dtype=dtype
        )
        for i in range(T):
            for j in range(0, i+1):
                alpha_matrix[i, j] = alpha ** (i - j)
        return alpha_matrix


    def forward(self, x_seq: torch.Tensor):
        x_seqbefore = x_seq

        out_shape = list(x_seq.shape)  #输出：[2,5,9,6,9]  T,N,C,H,W
        # unsqueeze x_seq so that flatten will not be out of range in FC network
        x_seq = x_seq.unsqueeze(-1).flatten(3)  # shape = [T, N, C, L]  [2,5,9,54]
        T, N, C, L = x_seq.shape
        B = self.n_compartment  #树突隔间的个数
        out_shape[2] = C // B # out_shape = [T, N, C // B, *]  输出shape会将通道数减少到1/B

        # compute the alpha matrix
        if self.T is None or self.T != T:
            self.T = T
            self.alpha_matrix = self.get_alpha_matrix(  #得到并行化的衰减矩阵？
                self.alpha, T, x_seq.device, x_seq.dtype
            )

        # # dendrite
        x_seq = x_seq.view(T, N, C // B, B, L)
        fs = self.forward_strength  #1
        # self.x_seq = x_seq  #[2,5,9,3,54]，将27的c拆到了3个隔间
        if self.f_dact == "mexican_hat":
            y_seq, comp = self.dendrite_mh_forward(  #[2,5,9,54]和[2,5,9,3,54] 对输入结果进行非线性转化   这个结果只要不带self就可以
                self.alpha_matrix, x_seq, fs, T
            )
        elif self.f_dact == "leaky_relu":
            y_seq, comp = self.dendritic_lrelu_forward(
                self.alpha_matrix, x_seq, fs, T
            )
        else:
            y_seq = x_seq.sum(3)

        #
        # # soma: LIF
        # # assume that v_rest = 0
        h = torch.zeros(
            [N, C // B, L], device=x_seq.device, dtype=x_seq.dtype
        )
        spike_seq = torch.empty(
            [T, N, C // B, L], device=x_seq.device, dtype=x_seq.dtype
        )
        if self.decay_somatic_input: #false
            # beta = 1 - 1 / tau_s
            # tau_s = 1 / (1 - beta)
            # y_seq = y_seq / tau_s
            y_seq = y_seq * (1. - self.beta)
        for t in range(T):  #这里就是正常的
            # soma: LIF
            y = y_seq[t]
            # self.h = self.h.detach()
            h = self.beta * h + y  #self.beta是衰减因子，self.h是残余膜电势，y是新输入，
            spike_seq[t] = self.surrogate_function(h - self.v_th)  #输出脉冲
            spike = spike_seq[t]
            h = (1. - spike) * h + spike * self.v_reset


        return spike_seq.view(out_shape)



    # def forward(self, x_seq: torch.Tensor):
    #     x_seqbefore = x_seq
    #
    #     out_shape = list(x_seq.shape)  #输出：[2,5,9,6,9]  T,N,C,H,W
    #     # unsqueeze x_seq so that flatten will not be out of range in FC network
    #     x_seq = x_seq.unsqueeze(-1).flatten(3)  # shape = [T, N, C, L]  [2,5,9,54]
    #     T, N, C, L = x_seq.shape
    #     B = self.n_compartment  #树突隔间的个数
    #     out_shape[2] = C // B # out_shape = [T, N, C // B, *]  输出shape会将通道数减少到1/B
    #
    #     # compute the alpha matrix
    #     if self.T is None or self.T != T:
    #         self.T = T
    #         self.alpha_matrix = self.get_alpha_matrix(  #得到并行化的衰减矩阵？
    #             self.alpha, T, x_seq.device, x_seq.dtype
    #         )
    #
    #     # # dendrite
    #     x_seq = x_seq.view(T, N, C // B, B, L)
    #     fs = self.forward_strength  #1
    #     # self.x_seq = x_seq  #[2,5,9,3,54]，将27的c拆到了3个隔间
    #     if self.f_dact == "mexican_hat":
    #         y_seq, comp = self.dendrite_mh_forward(  #[2,5,9,54]和[2,5,9,3,54] 对输入结果进行非线性转化   这个结果只要不带self就可以
    #             self.alpha_matrix, x_seq, fs, T
    #         )
    #     elif self.f_dact == "leaky_relu":
    #         y_seq, comp = self.dendritic_lrelu_forward(
    #             self.alpha_matrix, x_seq, fs, T
    #         )
    #     else:
    #         y_seq = x_seq.sum(3)
    #
    #     #
    #     # # soma: LIF
    #     # # assume that v_rest = 0
    #     self.h = torch.zeros(
    #         [N, C // B, L], device=x_seq.device, dtype=x_seq.dtype
    #     )
    #     spike_seq = torch.empty(
    #         [T, N, C // B, L], device=x_seq.device, dtype=x_seq.dtype
    #     )
    #     if self.decay_somatic_input: #false
    #         # beta = 1 - 1 / tau_s
    #         # tau_s = 1 / (1 - beta)
    #         # y_seq = y_seq / tau_s
    #         y_seq = y_seq * (1. - self.beta)
    #     for t in range(T):  #这里就是正常的
    #         # soma: LIF
    #         y = y_seq[t]
    #         # self.h = self.h.detach()
    #         self.h = self.beta * self.h + y  #self.beta是衰减因子，self.h是残余膜电势，y是新输入，
    #         spike_seq[t] = self.surrogate_function(self.h - self.v_th)  #输出脉冲
    #         spike = spike_seq[t]
    #         self.h = (1. - spike) * self.h + spike * self.v_reset
    #
    #
    #     # return spike_seq.view(out_shape)
    #     return x_seqbefore

#===============树突神经元相关内容结束======================
