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

import sys; sys.path.insert(0, ".")
from det3d.models.backbones.scn import SpMiddleResNetFHD
import torch
import pickle
import argparse
 
# custom functional package
import funcs
import exptool
# from tools.sparseconv_quantization import initialize, disable_quantization, quant_sparseconv_module, quant_add_module
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export scn to onnx file")
    parser.add_argument("--in-channel", type=int, default=4, help="SCN num of input channels")
    parser.add_argument("--ckpt", type=str, default="(这里填.pth模型的路径,比如centerpoint.pth)", help="SCN Checkpoint (scn backbone checkpoint)")
    parser.add_argument("--input", type=str, default=None, help="input pickle data, random if there have no input")
    parser.add_argument("--save-onnx", type=str, default="(这里填转换后的.scn.onnx模型的路径+文件名，比如centerpoint_pre.csn.onnx)", help="output onnx")
    parser.add_argument("--save-tensor", type=str, default=None, help="Save input/output tensor to file. The purpose of this operation is to verify the inference result of c++")
   
    args = parser.parse_args()
    # FP16 build 稀疏模型
    model = SpMiddleResNetFHD(args.in_channel).cuda().eval().half()
    
    print("🔥export original model🔥") 
    if args.ckpt:
        model = funcs.load_scn_backbone_checkpoint_KITTI(model, args.ckpt)
         # 进行层融合
    model = funcs.layer_fusion_bn_relu(model)         
    print("Fusion model:")
    print(model)

    if args.input:
        with open(args.input, "rb") as f:
            voxels, coors, spatial_shape, batch_size = pickle.load(f)
            voxels = torch.tensor(voxels).half().cuda()
            coors  = torch.tensor(coors).int().cuda()
    else:
        voxels = torch.zeros(1, args.in_channel).half().cuda()
        coors  = torch.zeros(1, 4).int().cuda()
        batch_size    = 1
        # spatial_shape计算公式举例：（需要根据自己训练模型时的参数去计算）
        # POINT_CLOUD_RANGE: [0, -50, -10, 150, 80, 10]
        # VOXEL_SIZE: [0.2, 0.2, 0.4]
        # spatial_shape = [(150-0)/0.2, (80-(-50))/0.2, (10-(-10))/0.4] = [750,650,50]     
        # spatial_shape = [750,650,50]
        # spatial_shape = [(70.4)/0.05, (40-(-40))/0.05, (1-(-3))/0.1] = [1408,,1600,40]    
        spatial_shape = [1408,1600,40]
    exptool.export_onnx(model, voxels, coors, batch_size, spatial_shape, args.save_onnx, args.save_tensor)


# if __name__ == "__main__":
#     initialize()
#     parser = argparse.ArgumentParser(description="Export scn to onnx file")
#     parser.add_argument("--in-channel", type=int, default=5, help="SCN num of input channels")
#     parser.add_argument("--ckpt", type=str, default=None, help="SCN Checkpoint (scn backbone checkpoint)")
#     parser.add_argument("--input", type=str, default=None, help="input pickle data, random if there have no input")
#     parser.add_argument("--save-onnx", type=str, default=None, help="output onnx")
#     parser.add_argument("--save-tensor", type=str, default=None, help="Save input/output tensor to file. The purpose of this operation is to verify the inference result of c++")
#     parser.add_argument("--use-quantization", action="store_true", help="use quantization")
#     args = parser.parse_args()

#     model = SpMiddleResNetFHD(args.in_channel).cuda().eval().half()
#     if args.use_quantization:
#         # QAT training using the fusionBn mode, so loading the checkpoint needs to be followed by the fusion operation
#         quant_sparseconv_module(model)
#         quant_add_module(model)
#         model = funcs.layer_fusion_bn(model)  
#         if args.ckpt:
#             model = funcs.load_scn_backbone_checkpoint(model, args.ckpt)
#         model = funcs.layer_fusion_relu(model)  
#         disable_quantization(model).apply()

#         # Set layer attributes
#         print("🔥export QAT/PTQ model🔥") 
#         for name, module in model.named_modules():
#             module.precision = "int8"
#             module.output_precision = "int8"
#         model.conv_input.precision = "fp16"
#         model.extra_conv.output_precision = "fp16"
#         model.conv1[0].conv1.precision = "fp16"   
#         model.conv1[0].conv2.precision = "fp16"
#         model.conv1[0].quant_add.precision = "fp16"
#         model.conv_input.output_precision = "fp16"  
#         model.conv1[0].conv1.output_precision = "fp16"   
#         model.conv1[0].conv2.output_precision = "fp16"
#         model.conv1[0].quant_add.output_precision = "int8"    
#     else:
#         print("🔥export original model🔥") 
#         if args.ckpt:
#             model = funcs.load_scn_backbone_checkpoint(model, args.ckpt, use_quant =False)
#         model = funcs.layer_fusion_bn_relu(model)
          
#     print("Fusion model:")
#     print(model)

#     if args.input:
#         with open(args.input, "rb") as f:
#             voxels, coors, spatial_shape, batch_size = pickle.load(f)
#             voxels = torch.tensor(voxels).half().cuda()
#             coors  = torch.tensor(coors).int().cuda()
#     else:
#         voxels = torch.zeros(1, args.in_channel).half().cuda()
#         coors  = torch.zeros(1, 4).int().cuda()
#         batch_size    = 1
#         spatial_shape = [1440, 1440, 40]

#     exptool.export_onnx(model, voxels, coors, batch_size, spatial_shape, args.save_onnx, args.save_tensor)