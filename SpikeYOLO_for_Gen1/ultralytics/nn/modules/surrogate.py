import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@torch.jit.script
def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)


'''
Rectangle Backward
'''
class rect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return heaviside(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < ctx.lens
        return grad_input * temp.float(), None

class Rect(nn.Module):
    def __init__(self, lens=0.5, spike=rect):
        super().__init__()
        self.lens = lens
        self.spike = spike
    
    def forward(self, inputs):
        return self.spike.apply(inputs, self.lens)


'''
Sigmoid Backward
''' 
@torch.jit.script
def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None


class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)
    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class Sigmoid(nn.Module):
    def __init__(self, alpha=2, spike=sigmoid):
        super().__init__()
        self.alpha = alpha
        self.spike = spike
    
    def forward(self, inputs):
        return self.spike.apply(inputs, self.alpha)

'''
Atan Backward
'''

@torch.jit.script
def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None


class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class Atan(nn.Module):
    def __init__(self, alpha=2, spike=atan):
        super().__init__()
        self.alpha = alpha
        self.spike = spike
    
    def forward(self, inputs):
        return self.spike.apply(inputs, self.alpha)

'''
Erf Backward
'''

@torch.jit.script
def erf_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output * (- (x * alpha).pow_(2)).exp_() * (alpha / math.sqrt(math.pi)), None


class erf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return erf_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class Erf(nn.Module):
    def __init__(self, alpha=2, spike=erf):
        super().__init__()
        self.alpha = alpha
        self.spike = spike
    
    def forward(self, inputs):
        return self.spike.apply(inputs, self.alpha)