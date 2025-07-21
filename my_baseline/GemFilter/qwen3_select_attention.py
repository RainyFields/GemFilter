# transformers.__version__ == '4.43.3'
import math
import typing
import torch
import pdb
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, TypedDict
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,eager_attention_forward

from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .gem_filter_utils import find_context


import sys
if sys.version_info >= (3, 11):
    Unpack = typing.Unpack
else:
    Unpack = typing_extensions.Unpack
sys.path.append('/work/lei/GemFilter')
from my_utils.utils import scaled_dot_product_gqa_attention_weights


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cumulative_seqlens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cumulative_seqlens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cumulative_seqlens_q: Optional[torch.LongTensor]
    cumulative_seqlens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


import torch
import time

class Qwen3SelectAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # print("configuration,", config._attn_implementation)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.reset()
        self.topk = getattr(config, "topk", None)  # Default to 1024 if not set

        self.select_layer_idx = getattr(config, "select_layer_idx", None)
        self.select_mode = False
        self.attn_output = None  # Initialize attn_output to None
        # print(f"configuration within Qwen3SelectAttention: {config}")

       

    def reset(self):
        self.indecies = None
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:


        # Initialize timing
        start_event = torch.cuda.Event(enable_timing=True)
        mid1_event = torch.cuda.Event(enable_timing=True)
        mid2_event = torch.cuda.Event(enable_timing=True)
        mid3_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

       
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        mid1_event.record()

        # print("shape of the query, key, value states:", query_states.shape, key_states.shape, value_states.shape)
        cos, sin = position_embeddings
        
        cos = cos.to(hidden_states.device)
        sin = sin.to(hidden_states.device)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
       
        mid2_event.record()

        if self.select_mode:
            self.reset()
            find_context(self, query_states, key_states)

            # print("after find_context, shape of the query, key, value states:", query_states.shape, key_states.shape, value_states.shape)
        
        if not self.select_mode and past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # print("if not in select_mode, shape of the query, key, value states:", query_states.shape, key_states.shape, value_states.shape)

        attention_interface: Callable = eager_attention_forward
        if kwargs.get("output_attentions", True):
            # print("[INFO] Using eager attention interface for output_attentions=True")
            # self._attn_implementation = "eager"  # Force eager attention if flash attention is not used => why the results are different with flash attention?
            # attention_interface = eager_attention_forward
            self.config._attn_implementation == "flash_attention_2"
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        else:
            # print("what is the attention interface?", self.config._attn_implementation)
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # print(f"[INFO] Attention interface used: {attention_interface.__name__}")

        # print("shape of the query, key, value states:", query_states.shape, key_states.shape, value_states.shape)

        # print("kwargs for attention interface:", kwargs)
        # pdb.set_trace() # flash_attention_forward return None for attn_weights
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        
        self.attn_output = attn_output

        
            # print("Attention Weights Shape:", attention_weights.shape)
            # print("Attention Weights:", attention_weights)

        # print("manual inspection of the attention output:", attn_weights.shape)
        mid3_event.record()
        # print("end of attention interface")

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        end_event.record()

        # Synchronize for accurate timings
        torch.cuda.synchronize()

        # Report timings
        # print(f"[Timing] Q/K/V projection & norm: {start_event.elapsed_time(mid1_event):.3f} ms")
        # print(f"[Timing] Rotary embedding: {mid1_event.elapsed_time(mid2_event):.3f} ms")
        # print(f"[Timing] Attention forward pass: {mid2_event.elapsed_time(mid3_event):.3f} ms")
        # print(f"[Timing] Output projection + reshape: {mid3_event.elapsed_time(end_event):.3f} ms")
        # print(f"[Timing] Total forward pass: {start_event.elapsed_time(end_event):.3f} ms")

        

        return attn_output, attn_weights
