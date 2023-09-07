# coding=utf-8
# Copyright (c) 2023 ADEPT AI LABS INC.

import torch


###### BIAS SQRELU FUSION/ NO AUTOGRAD ################


@torch.jit.script
def bias_sqrelu(bias, y):
    x = bias + y
    relud_x = torch.relu(x)
    return relud_x * relud_x


@torch.jit.script
def bias_sqrelu_back(g, bias, y):
    x = bias + y
    return g * 2 * torch.relu(x)


class SqReLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_sqrelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_sqrelu_back(grad_output, bias, input)
        return tmp, tmp


bias_sqrelu_impl = SqReLUFunction.apply
