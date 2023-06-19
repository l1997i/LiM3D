import sys
import time
from typing import Optional

import numpy as np
import torch
from spconv.constants import FILTER_HWIO
from spconv.core import ConvAlgo
from spconv.cppconstants import CPU_ONLY_BUILD
from spconv.debug_utils import spconv_save_debug_data
from spconv.pytorch import functional as Fsp
from spconv.pytorch import ops
from spconv.pytorch.conv import SparseConv3d, SparseConvolution
from spconv.pytorch.core import (ImplicitGemmIndiceData, IndiceData,
                                 SparseConvTensor, expand_nd)
from spconv.pytorch.modules import SparseModule
from spconv.utils import nullcontext
from torch import nn


class GroupSparseConv3d(SparseConvolution, SparseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 n_groups=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        SparseModule.__init__(self, name=name)
        self.ndim = 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = expand_nd(self.ndim, kernel_size)
        self.stride = expand_nd(self.ndim, stride)
        kv = int(np.prod(self.kernel_size))
        self.dilation = expand_nd(self.ndim, dilation)
        self.padding = expand_nd(self.ndim, padding)

        self.conv1x1 = kv == 1
        self.transposed = False
        self.inverse = False
        self.output_padding = expand_nd(self.ndim, 0)
        self.groups = groups
        self.subm = True
        self.indice_key = indice_key
        self.bias = bias
        if algo is None:
            if kv <= 32 and not CPU_ONLY_BUILD:
                if kv < 8:
                    algo = ConvAlgo.MaskImplicitGemm
                else:
                    algo = ConvAlgo.MaskImplicitGemm
            else:
                algo = ConvAlgo.Native
        if kv > 32:
            assert algo == ConvAlgo.Native, "implicit gemm don't support kv >= 32 for now"
        if CPU_ONLY_BUILD:
            assert algo == ConvAlgo.Native, "cpu only build only support native algorithm"
        self.algo = algo
        self.fp32_accum = fp32_accum
        self.n_groups = n_groups

        # num of input/output channels per sub-convolution
        self.in_ch_group = in_channels // self.n_groups
        self.out_ch_group = out_channels // self.n_groups
        self.weights = []

        # self._weight_initialization_tensor()
                
    def _weight_initialization_tensor(self, device):
        bias = self.bias

        for idx in range(self.n_groups):

            if self.algo == ConvAlgo.Native:
                if FILTER_HWIO:
                    # RSCK
                    self.weights.append(torch.randn(*self.kernel_size, self.in_ch_group, self.out_ch_group).to("cuda"))
                else:
                    # RSKC
                    self.weights.append(torch.randn(*self.kernel_size, self.out_ch_group, self.in_ch_group).to("cuda"))
            else:
                # KRSC
                self.weights.append(torch.randn(self.out_ch_group, *self.kernel_size, self.in_ch_group).to("cuda"))

            if bias:
                self.weights.append(torch.randn(
                    self.out_ch_group).to("cuda"))

    def _chunk_sptensor_to_dense_list(self, x):
        n_channel = x.features.shape[1]
        ret = []

        for i in range(n_channel):
            x_i_feature = x.features[:, i].unsqueeze(
                1).to("cuda")  # [N] --> [N, 1]
            ret.append(x_i_feature)
        return ret

    def forward(self, input: SparseConvTensor):
        assert isinstance(input, SparseConvTensor)
        assert input.features.shape[
            1] == self.in_channels, "channel size mismatch"

        features = input.features
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        out_spatial_shape = spatial_shape

        out_tensor = input.shadow_copy()
        indice_dict = input.indice_dict.copy()

        algo = self.algo
        if self.indice_key is not None:
            datas = input.find_indice_pair(self.indice_key)
            if datas is not None:
                msg = "due to limitation of pytorch, you must provide same algo to layers share same indice key."
                assert algo == datas.algo, msg
                # algo = datas.algo
        profile_ctx = nullcontext()

        with profile_ctx:
            if algo == ConvAlgo.Native:
                datas = input.find_indice_pair(self.indice_key)
                if datas is not None:
                    assert isinstance(datas, IndiceData)

                if self.indice_key is not None and datas is not None:
                    outids = datas.out_indices
                    indice_pairs = datas.indice_pairs
                    indice_pair_num = datas.indice_pair_num
                    assert self.subm, "only support reuse subm indices"
                    self._check_subm_reuse_valid(input, spatial_shape,
                                                 datas)
                else:
                    if input.benchmark:
                        torch.cuda.synchronize()
                        t = time.time()
                    try:

                        outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                            indices, batch_size, spatial_shape, algo,
                            self.kernel_size, self.stride, self.padding,
                            self.dilation, self.output_padding, self.subm,
                            self.transposed)

                    except Exception as e:
                        msg = "[Exception|native_pair]"
                        msg += f"indices={indices.shape},bs={batch_size},ss={spatial_shape},"
                        msg += f"algo={algo},ksize={self.kernel_size},stride={self.stride},"
                        msg += f"padding={self.padding},dilation={self.dilation},subm={self.subm},"
                        msg += f"transpose={self.transposed}"
                        print(msg, file=sys.stderr)
                        spconv_save_debug_data(indices)
                        raise e
                    if input.benchmark:
                        torch.cuda.synchronize()
                        interval = time.time() - t
                        out_tensor.benchmark_record[
                            self.name]["indice_gen_time"].append(interval)

                    indice_data = IndiceData(outids,
                                             indices,
                                             indice_pairs,
                                             indice_pair_num,
                                             spatial_shape,
                                             out_spatial_shape,
                                             is_subm=self.subm,
                                             algo=algo,
                                             ksize=self.kernel_size,
                                             stride=self.stride,
                                             padding=self.padding,
                                             dilation=self.dilation)

                    if self.indice_key is not None:
                        msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                        assert self.indice_key not in indice_dict, msg
                        indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()

                indice_pairs_calc = indice_pairs
                if indice_pairs.device != features.device:
                    indice_pairs_calc = indice_pairs.to(features.device)

                # iterate through each sub-convolution
                features_list = self._chunk_sptensor_to_dense_list(input)
                weights_device = input.features.device
                self._weight_initialization_tensor(weights_device)
                
                # to store the results of each channel
                out = Fsp.indice_subm_conv(features_list[0], self.weights[0], 
                                             indice_pairs_calc, indice_pair_num, outids.shape[0], 
                                             algo, input._timer)
                conv_out = [Fsp.indice_subm_conv(features_list[idx], self.weights[idx], 
                                             indice_pairs_calc, indice_pair_num, outids.shape[0], 
                                             algo, input._timer) for idx in range(1, self.n_groups)]
                for idx in range(self.n_groups-1):
                    out = torch.cat((out, conv_out[idx]), dim=1)
                out_features = out

            else:
                datas = input.find_indice_pair(self.indice_key)
                if datas is not None:
                    assert isinstance(datas, ImplicitGemmIndiceData)
                else:
                    if self.indice_key is not None and datas is not None:
                        outids = datas.out_indices
                        pair_fwd = datas.pair_fwd
                        pair_bwd = datas.pair_bwd
                        pair_mask_fwd_splits = datas.pair_mask_fwd_splits
                        pair_mask_bwd_splits = datas.pair_mask_bwd_splits
                        mask_argsort_fwd_splits = datas.mask_argsort_fwd_splits
                        mask_argsort_bwd_splits = datas.mask_argsort_bwd_splits
                        masks = datas.masks
                        assert self.subm, "only support reuse subm indices"
                        self._check_subm_reuse_valid(input, spatial_shape,
                                                     datas)
                    else:

                        with input._timer.namespace("gen_pairs"):
                            # we need to gen bwd indices for regular conv
                            # because it may be inversed.
                            try:
                                res = ops.get_indice_pairs_implicit_gemm(
                                    indices,
                                    batch_size,
                                    spatial_shape,
                                    algo,
                                    ksize=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    dilation=self.dilation,
                                    out_padding=self.output_padding,
                                    subm=self.subm,
                                    transpose=self.transposed,
                                    is_train=(not self.subm) or self.training,
                                    alloc=input.thrust_allocator,
                                    timer=input._timer)

                            except Exception as e:
                                msg = "[Exception|implicit_gemm_pair]"
                                msg += f"indices={indices.shape},bs={batch_size},ss={spatial_shape},"
                                msg += f"algo={algo},ksize={self.kernel_size},stride={self.stride},"
                                msg += f"padding={self.padding},dilation={self.dilation},subm={self.subm},"
                                msg += f"transpose={self.transposed}"
                                print(msg, file=sys.stderr)
                                spconv_save_debug_data(indices)
                                raise e

                        outids = res[0]
                        pair_fwd = res[2]
                        pair_bwd = res[3]
                        pair_mask_fwd_splits = res[4]
                        pair_mask_bwd_splits = res[5]
                        mask_argsort_fwd_splits = res[6]
                        mask_argsort_bwd_splits = res[7]
                        masks = res[8]
                        if self.indice_key is not None:
                            indice_data = ImplicitGemmIndiceData(
                                outids,
                                indices,
                                pair_fwd,
                                pair_bwd,
                                pair_mask_fwd_splits=pair_mask_fwd_splits,
                                pair_mask_bwd_splits=pair_mask_bwd_splits,
                                mask_argsort_fwd_splits=mask_argsort_fwd_splits,
                                mask_argsort_bwd_splits=mask_argsort_bwd_splits,
                                masks=masks,
                                is_subm=self.subm,
                                spatial_shape=spatial_shape,
                                out_spatial_shape=out_spatial_shape,
                                algo=algo,
                                ksize=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)
                            msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                            assert self.indice_key not in indice_dict, msg
                            indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()

                # iterate through each sub-convolution
                features_list = self._chunk_sptensor_to_dense_list(input)
                weights_device = features_list.device
                self._weight_initialization_tensor(weights_device)
                
                # to store the results of each channel
                out = Fsp.indice_subm_conv(features_list[0], self.weights[0], 
                                             indice_pairs_calc, indice_pair_num, outids.shape[0], 
                                             algo, input._timer)
                conv_out = [Fsp.indice_subm_conv(features_list[idx], self.weights[idx], 
                                             indice_pairs_calc, indice_pair_num, outids.shape[0], 
                                             algo, input._timer) for idx in range(1, self.n_groups)]
                for idx in range(self.n_groups-1):
                    out = torch.cat((out, conv_out[idx]), dim=1)
                out_features = out
                
        if self.bias is not None:
            out_features += self.bias
        if input.benchmark:
            torch.cuda.synchronize()
            interval = time.time() - t
            out_tensor.benchmark_record[self.name]["time"].append(interval)
            out_tensor.benchmark_record[self.name]["num_points"].append(
                features.shape[0])
            out_tensor.benchmark_record[self.name]["num_out_points"].append(
                out_features.shape[0])
        out_tensor = out_tensor.replace_feature(out_features)
        out_tensor.indices = outids
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = out_spatial_shape
        torch.cuda.empty_cache()
        return out_tensor


class SpDepthWSepaConv3d(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer, kernel_size, stride, padding, n_groups, device, indice_key=None, bias=True, algo=ConvAlgo.Native):
        super(SpDepthWSepaConv3d, self).__init__()

        self.depthwise = GroupSparseConv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, stride=stride,
                                           padding=padding, n_groups=n_groups, bias=bias, indice_key=indice_key, algo=algo)#.to(device)
        self.pointwise = SparseConv3d(nin * kernels_per_layer, nout, kernel_size=1, bias=bias, indice_key=indice_key, algo=algo)#.to(device)

    # @profile_every(1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        torch.cuda.empty_cache()
        return out


class DepthWSepaConv3d(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer, kernel_size, stride, padding, n_groups, device, indice_key=None, bias=True):
        super(DepthWSepaConv3d, self).__init__()

        self.depthwise = torch.nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, stride=stride,
                                         padding=padding, groups=n_groups, bias=bias)#.to(device)
        self.pointwise = torch.nn.Conv3d(
            nin * kernels_per_layer, nout, kernel_size=1, bias=bias)#.to(device)

    def forward(self, x):
        out = self.depthwise(x.dense())
        out = self.pointwise(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convolution - Baseline
    bl_conv3d = nn.Conv3d(20, 40, kernel_size=(1, 3, 3), stride=1,
                          padding=(0, 1, 1), bias=False)#.to(device)

    # Group Convolution: https://arxiv.org/pdf/1605.06489.pdf
    g_conv3d = nn.Conv3d(20, 40, kernel_size=(1, 3, 3), stride=1,
                         padding=(0, 1, 1), groups=20, bias=False)#.to(device)

    # Sparse Convolution: https://arxiv.org/pdf/1612.07837.pdf
    s_conv3d = SparseConv3d(20, 40, kernel_size=(1, 3, 3), stride=1,
                            padding=(0, 1, 1), bias=False)#.to(device)

    # Ours 0: Group Sparse Convolution
    gs_conv_3d = GroupSparseConv3d(20, 40, kernel_size=(1, 3, 3), stride=1,
                                   padding=(0, 1, 1), n_groups=20, bias=False, indice_key="gs_conv_3d", algo=ConvAlgo.Native)

    # Ours 1: Sparse Depthwise Separable Convolution
    sds_conv_3d = SpDepthWSepaConv3d(nin=20, nout=40, kernels_per_layer=2, kernel_size=(
        1, 3, 3), stride=1, padding=(0, 1, 1), n_groups=20, bias=False, device=device, algo=ConvAlgo.Native)

    # Point cloud (voxel features) sample
    # logits = torch.load("save_tensors/logits.pt")
    # sample_size = 40*480*360*32

    # sample: random small
    logits_dense = torch.rand((2, 4, 6, 8, 20))#.to(device)     # NCDHW
    logits = SparseConvTensor.from_dense(logits_dense)
    sample_size = 4*6*8*40
    logits.indices

    # the output of each convolution
    out_bl = bl_conv3d(logits_dense.permute(0, 4, 1, 2, 3))    # NDHWC
    out_g = g_conv3d(logits.dense())
    out_s = s_conv3d(logits).dense()
    out_gs = gs_conv_3d(logits)

    # Test 0: check the correctness of the sparse convolution
    # False, due to the different weights and biases
    diff_s_g = torch.sum((out_s - out_g)**2) / sample_size
    print("T0 - Test Sparse Conv 3D: diff_s_g = ", diff_s_g.item())  # 0.0

    # Test 1: check the correctness of the group convolution
    # False, due to the different weights and biases
    diff_gs_g = torch.sum((out_gs.dense() - out_g)**2) / sample_size
    print("T1 - Test Group Conv 3D: diff_gs_g = ", diff_gs_g.item())  # 0.0

    # Test 2: check the shape of the Sparse Depthwise Separable Convolution
    out_my_dwsConv2d = sds_conv_3d(logits)
    print("T2 - Test the shape of SpDepthWSepaConv3d:",
          out_my_dwsConv2d.dense().shape)  # [1, 40, 480, 360, 32]
