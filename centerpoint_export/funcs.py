# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import collections
from det3d.models.backbones.scn import SpMiddleResNetFHD
from spconv.pytorch import SparseSequential
from spconv.pytorch import conv
from det3d.models.backbones.scn import SparseBasicBlock
import cumm.tensorview as tv
import numpy as np

def make_new_repr(old_repr):
    def new_repr(self):
        s = old_repr(self)
        if self.act_type is not None:
            p = s.rfind(")")
            s = s[:p] + f', act={self.act_type}' + s[p:]
        return s
    return new_repr

# setup repr function, add activation
conv.SparseConvolution.__repr__ = make_new_repr(conv.SparseConvolution.__repr__)

def fuse_bn_weights(conv_w_OKI, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    NDim = conv_w_OKI.ndim - 2
    permute = [0, NDim+1] + [i+1 for i in range(NDim)]
    conv_w_OIK = conv_w_OKI.permute(*permute)
    # OIDHW
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w_OIK = conv_w_OIK * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w_OIK.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    permute = [0,] + [i+2 for i in range(NDim)] + [1,]
    conv_w_OKI = conv_w_OIK.permute(*permute).contiguous()
    return torch.nn.Parameter(conv_w_OKI), torch.nn.Parameter(conv_b)

def fuse_bn(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    conv.weight, conv.bias = fuse_bn_weights(conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

def fuse_scn_backbone_checkpoint(model, file):
    device   = next(model.parameters()).device
    ckpt = torch.load(file, map_location=device)["state_dict"]
    new_ckpt = collections.OrderedDict()
    for key, val in ckpt.items():
        if key.startswith("backbone."):
            newkey = key[key.find(".")+1:]
            new_ckpt[newkey] = val
    
    for name, module in model.named_modules(): 
        if name not in new_ckpt:
            print(name)
            #delattr(model, name)

def load_scn_backbone_checkpoint(model, file, use_quant= True):

    device   = next(model.parameters()).device
    ckpt     = torch.load(file, map_location=device)["state_dict"]
    new_ckpt = collections.OrderedDict()
    for key, val in ckpt.items():
        if key.startswith("backbone."):
            newkey = key[key.find(".")+1:]
            if use_quant == False:
                if val.ndim == 5:
                    val = val.permute(4, 0, 1, 2, 3)

            new_ckpt[newkey] = val

    model.load_state_dict(new_ckpt,strict =True)
    return model

def load_scn_backbone_checkpoint_KITTI(model, file):
    device   = next(model.parameters()).device
    ckpt     = torch.load(file, map_location=device)["model_state"]
    new_ckpt = collections.OrderedDict()
    for key, val in ckpt.items():
        if key.startswith("backbone_3d."):
            newkey = key[key.find(".")+1:]
            if(newkey.startswith("conv2.0.0")):
                newkey = "conv2.0" + newkey.split("conv2.0.0")[-1]
            elif(newkey.startswith("conv2.0.1")):
                newkey = "conv2.1" + newkey.split("conv2.0.1")[-1]
            elif(newkey.startswith("conv2.1")):
                newkey = "conv2.3" + newkey.split("conv2.1")[-1]
            elif(newkey.startswith("conv2.2")):
                newkey = "conv2.4" + newkey.split("conv2.2")[-1]
            elif(newkey.startswith("conv3.0.0")):
                newkey = "conv3.0" + newkey.split("conv3.0.0")[-1]
            elif(newkey.startswith("conv3.0.1")):
                newkey = "conv3.1" + newkey.split("conv3.0.1")[-1]
            elif(newkey.startswith("conv3.1")):
                newkey = "conv3.3" + newkey.split("conv3.1")[-1]
            elif(newkey.startswith("conv3.2")):
                newkey = "conv3.4" + newkey.split("conv3.2")[-1]
            elif(newkey.startswith("conv4.0.0")):
                newkey = "conv4.0" + newkey.split("conv4.0.0")[-1]
            elif(newkey.startswith("conv4.0.1")):
                newkey = "conv4.1" + newkey.split("conv4.0.1")[-1]
            elif(newkey.startswith("conv4.1")):
                newkey = "conv4.3" + newkey.split("conv4.1")[-1]
            elif(newkey.startswith("conv4.2")):
                newkey = "conv4.4" + newkey.split("conv4.2")[-1]
            elif(newkey.startswith("conv_out")):
                newkey = "extra_conv" + newkey.split("conv_out")[-1]
            else:
                print("backbone3d key is matching:", newkey)

            new_ckpt[newkey] = val
    model.load_state_dict(new_ckpt)
    return model


# def new_sparse_basic_block_forward(self, is_fuse_relu=True):
#     def sparse_basic_block_forward(x):
#         identity = x
#         out = self.conv1(x)
#         if is_fuse_relu == False:
#             out = out.replace_feature(self.relu(out.features))#####note train only

#         out = self.conv2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         if hasattr(self, 'quant_add'):
#             out = out.replace_feature(self.quant_add(out.features, identity.features))
#         else:
#             out = out.replace_feature(out.features + identity.features)            
#         out = out.replace_feature(self.relu(out.features))
#         return out
#     return sparse_basic_block_forward

def new_sparse_basic_block_forward(self, is_fuse_relu=True):
    def sparse_basic_block_forward(x):
        identity = x
        out = self.conv1(x)
        if is_fuse_relu == False:
            out = out.replace_feature(self.relu(out.features))#####note train only

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # if hasattr(self, 'quant_add'):
        #     out = out.replace_feature(self.quant_add(out.features, identity.features))
        # else:
        #     out = out.replace_feature(out.features + identity.features)  
        
        out = out.replace_feature(out.features + identity.features)            
        out = out.replace_feature(self.relu(out.features))
        return out
    return sparse_basic_block_forward


def fuse_sparse_basic_block(self, is_fuse_bn = False, is_fuse_relu=True):
    self.forward = new_sparse_basic_block_forward(self, is_fuse_relu=is_fuse_relu)
    if is_fuse_relu == True:
        self.conv1.act_type = tv.gemm.Activation.ReLU 

    if is_fuse_bn == True:
        fuse_bn(self.conv1, self.bn1)
        fuse_bn(self.conv2, self.bn2)
        delattr(self, "bn1")
        delattr(self, "bn2")

def layer_fusion_bn(model : SpMiddleResNetFHD):
    # fuse conv except for conv_input During training (To get a better mAP)
    for conv_name in ["conv2", "conv3", "conv4", "extra_conv"]:
        conv_instance = getattr(model, conv_name)
        c, b, r = [conv_instance[i] for i in range(3)]
        fuse_bn(c, b)
        if len(conv_instance) == 3:
            new_conv = SparseSequential(
                c,r
            )
        else:
            new_conv = SparseSequential(
                *([c,r] + [conv_instance[i] for i in range(3, len(conv_instance))])
            )
            
        setattr(model, conv_name, new_conv)

    ## fuse all SparseBasicBlock
    for name, block in model.named_modules():
        if isinstance(block, SparseBasicBlock):
            fuse_sparse_basic_block(block, is_fuse_bn = True, is_fuse_relu =False)
    return model

def layer_fusion_relu(model : SpMiddleResNetFHD):
    # fuse all conv
    for conv_name in ["conv_input", "conv2", "conv3", "conv4", "extra_conv"]:
        conv_instance = getattr(model, conv_name) #
        if(conv_name == "conv_input"):
            c, b, r = [conv_instance[i] for i in range(3)]
            fuse_bn(c, b)
            c.act_type = tv.gemm.Activation.ReLU
            new_conv = c
            setattr(model, conv_name, new_conv)
        else:
            c, r = [conv_instance[i] for i in range(2)]
            c.act_type = tv.gemm.Activation.ReLU
            if len(conv_instance) == 2:
                new_conv = c
            else:
                new_conv = SparseSequential(
                    *([c] + [conv_instance[i] for i in range(2, len(conv_instance))])
                )
            setattr(model, conv_name, new_conv)

    # fuse all SparseBasicBlock
    for name, block in model.named_modules():
        if isinstance(block, SparseBasicBlock):
            fuse_sparse_basic_block(block, is_fuse_bn= False, is_fuse_relu= True)
    return model


# export for orignal model
def layer_fusion_bn_relu(model : SpMiddleResNetFHD):
    # fuse all conv
    for conv_name in ["conv_input", "conv2", "conv3", "conv4", "extra_conv"]:
        conv_instance = getattr(model, conv_name)
        c, b, r = [conv_instance[i] for i in range(3)]
        fuse_bn(c, b)
        c.act_type = tv.gemm.Activation.ReLU
        if len(conv_instance) == 3:
            new_conv = c
        else:
            new_conv = SparseSequential(
                *([c] + [conv_instance[i] for i in range(3, len(conv_instance))])
            )
        setattr(model, conv_name, new_conv)

    # fuse all SparseBasicBlock
    for name, block in model.named_modules():
        if isinstance(block, SparseBasicBlock):
            fuse_sparse_basic_block(block, is_fuse_relu= True, is_fuse_bn= True)
    return model


# This function stores a file that can be very easily loaded and used by c++
def save_tensor(tensor, file):

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().data.numpy()
    elif not isinstance(tensor, np.ndarray):
        tensor = np.array(tensor)

    dtype_map = {"float32" : 0, "float16" : 1, "int32" : 2, "int64" : 3}
    if str(tensor.dtype) not in dtype_map:
        raise RuntimeError(f"Unsupport dtype {tensor.dtype}")

    magic_number = 0x33ff1101
    with open(file, "wb") as f:
        head = np.array([magic_number, tensor.ndim, dtype_map[str(tensor.dtype)]], dtype=np.int32).tobytes()
        f.write(head)

        dims = np.array(tensor.shape, dtype=np.int32).tobytes()
        f.write(dims)
        
        data = tensor.tobytes()
        f.write(data)

# This function stores a file that can be very easily loaded and used by c++
def load_tensor(file):

    dtype_for_integer_mapping = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.int64}
    dtype_size_mapping        = {np.float32 : 4, np.float16 : 2, np.int32 : 4, np.int64 : 8}

    with open(file, "rb") as f:
        magic_number, ndim, dtype_integer = np.frombuffer(f.read(12), dtype=np.int32)
        if dtype_integer not in dtype_for_integer_mapping:
            raise RuntimeError(f"Can not find match dtype for index {dtype_integer}")

        dtype            = dtype_for_integer_mapping[dtype_integer]
        magic_number_std = 0x33ff1101
        assert magic_number == magic_number_std, f"this file is not tensor file"
        dims   = np.frombuffer(f.read(ndim * 4), dtype=np.int32)
        volumn = np.cumprod(dims)[-1]
        data   = np.frombuffer(f.read(volumn * dtype_size_mapping[dtype]), dtype=dtype).reshape(*dims)
        return data