#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Callable, Dict, Optional

import torch
from torch import nn

from internlm.model.ops.linear import (
    ColumnParallelLinearTorch,
    ISPLinear,
    MegatronColumnParallelLinearTorch,
    MegatronRowParallelLinearTorch,
    RowParallelLinearTorch,
)
from internlm.model.utils import Silu


class BaseFeedForward(nn.Module):
    """
    Base FeedForward in flash implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
        column_cls (Optional[Callable]): The column parallel class for w1 and w3. None by default.
        row_cls (Optional[Callable]): The row parallel class for w2. None by default.
        mlp_layer_fusion (Optional[Bool]):  Some linears without bias in FFN can be fused to reduce the comm cost of SP.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        mlp_layer_fusion: Optional[bool] = False,
        sequence_parallel: Optional[bool] = False,
        column_cls: Optional[Callable] = None,
        row_cls: Optional[Callable] = None,
    ):
        super().__init__()
        self.mlp_layer_fusion = mlp_layer_fusion
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)
        mlp_args = {
            "process_group": process_group,
            "bias": bias,
            "sequence_parallel": sequence_parallel,
            "device": device,
            "dtype": dtype,
            "multiple_of": 1,  # TODO: check Column/RowParallelLinearTorch.
        }
        if not self.mlp_layer_fusion:
            # gate_proj
            self.w1 = column_cls(in_features, hidden_features, **mlp_args)
            # down_proj
            self.w2 = row_cls(hidden_features, out_features, **mlp_args)
            # up_proj
            self.w3 = column_cls(in_features, hidden_features, **mlp_args)
        else:
            assert bias is False, "Fuesd FeedForward only support bias is False."
            # fused gate/up projection
            self.fused_w1_w3 = column_cls(in_features, hidden_features * 2, **mlp_args)
            # down_proj
            self.w2 = row_cls(hidden_features, out_features, **mlp_args)

            # TODO: Internal methods could change without a deprecation warning.
            self._register_load_state_dict_pre_hook(BaseFeedForward._mlp_pre_load_convert, with_module=True)
            self._register_state_dict_hook(BaseFeedForward._mlp_save_convert)

    def forward(self, x):
        if not self.mlp_layer_fusion:
            w1_o = self.w1(x)
            w3_o = self.w3(x)
        else:
            fussed_out = self.fused_w1_w3(x)
            w1_o, w3_o = BaseFeedForward.split_fused_mlp_output(fussed_out)
        out = self.w2(Silu(w1_o, w3_o))
        return out

    @staticmethod
    def split_fused_mlp_weight(w1_w3):
        w1, w3 = torch.split(w1_w3, w1_w3.shape[0] // 2, dim=0)
        return w1, w3

    @staticmethod
    def split_fused_mlp_output(w1_w3_out):
        w1_o, w3_o = torch.split(w1_w3_out, w1_w3_out.shape[-1] // 2, dim=-1)
        return w1_o, w3_o

    def _mlp_pre_load_convert(self, state_dict, *args, **kwargs) -> None:  # pylint: disable=W0613
        if self.mlp_layer_fusion and "fused_w1_w3.weight" not in state_dict:
            w1, w3 = state_dict.pop("w1.weight"), state_dict.pop("w3.weight")
            state_dict["fused_w1_w3.weight"] = torch.cat([w1, w3], dim=0)
        if not self.mlp_layer_fusion and ("w1.weight" not in state_dict or "w3.weight" not in state_dict):
            state_dict["w1.weight"], state_dict["w3.weight"] = self.split_fused_mlp_weight(
                state_dict.pop("fused_w1_w3.weight")
            )

    def _mlp_save_convert(self, state_dict, *args, **kwargs) -> Dict:  # pylint: disable=W0613
        if self.mlp_layer_fusion:
            state_dict["w1.weight"], state_dict["w3.weight"] = self.split_fused_mlp_weight(
                w1_w3=state_dict.pop("fused_w1_w3.weight")
            )

        return state_dict


class FeedForward(BaseFeedForward):
    """
    FeedForward in flash implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        mlp_layer_fusion: Optional[bool] = False,
        sequence_parallel: Optional[bool] = False,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            mlp_layer_fusion,
            sequence_parallel,
            ColumnParallelLinearTorch,
            RowParallelLinearTorch,
        )


class MegatronFeedForward(BaseFeedForward):
    """
    FeedForward in megatron implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        mlp_layer_fusion: Optional[bool] = False,
        sequence_parallel: Optional[bool] = False,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            mlp_layer_fusion,
            sequence_parallel,
            MegatronColumnParallelLinearTorch,
            MegatronRowParallelLinearTorch,
        )


class ISPFeedForward(BaseFeedForward):
    """
    FeedForward in ISP.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        mlp_layer_fusion: Optional[bool] = False,
        sequence_parallel: Optional[bool] = False,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            mlp_layer_fusion,
            sequence_parallel,
            ISPLinear,
            ISPLinear,
        )


def get_mlp_cls(tp_mode: str):
    if tp_mode in ["mtp", "fsp"]:
        mlp_cls = FeedForward
    elif tp_mode == "msp":
        mlp_cls = MegatronFeedForward
    else:
        mlp_cls = ISPFeedForward
    return mlp_cls
