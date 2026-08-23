# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

from . import sm90_quant
from .params import DeepseekV4AttentionType
from .rope import deepseek_v4_rotary_embedding

if TYPE_CHECKING:
    from .metadata import DeepseekV4TrtllmAttentionMetadata


class KVCacheDtype(IntEnum):
    """KV cache dtype/layout preset (values match C++ cache_scale_type parameter).

    The store dtype and scale layout are implied by this value:
      - NONE:              keeps the input dtype (bf16/fp32, decided by the
                           caller's tensor element size).
      - FP8_PERTENSOR:     1 byte per value (FP8 E4M3) with implicit scale=1.
      - FP8_BLOCKWISE:     1 byte per value + 1 fp32 scale per 128 values.
      - MXFP4_BLOCKWISE:   packed FP4 (½ byte per value) + 1 UE8M0 byte per
                           32 values.

    Storage size in bytes per logical element is therefore::

        size_per_value = {
            NONE: elem_bytes,  # caller-side
            FP8_PERTENSOR: 1,
            FP8_BLOCKWISE: 1 + 4 / 128,  # data + fp32 scale
            MXFP4_BLOCKWISE: 0.5 + 1 / 32,  # nibble + ue8m0 byte
        }[kv_cache_dtype]
    """

    NONE = 0
    FP8_PERTENSOR = 1  # FP8 E4M3 with implicit scale=1
    FP8_BLOCKWISE = 2  # FP8 E4M3 with per-128 fp32 scales
    MXFP4_BLOCKWISE = 3  # packed FP4 E2M1 with per-32 UE8M0 scales


_KV_CACHE_DTYPE_MAP = {
    "default": KVCacheDtype.NONE,
    "bf16": KVCacheDtype.NONE,
    "fp8_pertensor": KVCacheDtype.FP8_PERTENSOR,
    "fp8_blockwise": KVCacheDtype.FP8_BLOCKWISE,
    "mxfp4": KVCacheDtype.MXFP4_BLOCKWISE,
}


def resolve_kv_cache_dtype(kv_cache_dtype: Union[str, KVCacheDtype]) -> KVCacheDtype:
    if isinstance(kv_cache_dtype, str):
        return _KV_CACHE_DTYPE_MAP[kv_cache_dtype]
    return kv_cache_dtype


class Compressor(nn.Module):
    """KV compressor using Triton kernels with paged memory management.

    Args:
        mla_params: MLA parameters containing hidden_size and head dimensions
        layer_idx: Layer index for cache management
        compress_ratio: Compression ratio (e.g., 4 compresses 4 tokens into 1)
        norm_eps: RMSNorm epsilon
        skip_create_weights_in_init: Whether to skip weight initialization
        pos_embd_params: Positional embedding parameters for RoPE
        dtype: Data type for computation
        kv_cache_dtype: Cache preset string or KVCacheDtype.
        rotate_activation: Whether to apply Hadamard transform in postprocessing (False to skip)
        simulate_source_act_quant: Reproduce the DeepSeek-V4 checkpoint's own
            FP32 gate/KV projection and its post-pool activation quantisation
            simulation. Only the DeepSeek-V4 SM90 path asks for this.
    """

    def __init__(
        self,
        mla_params: MLAParams,
        layer_idx: int,
        compress_ratio: int,
        norm_eps: float,
        skip_create_weights_in_init: bool,
        pos_embd_params: PositionalEmbeddingParams,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_dtype: Union[str, KVCacheDtype] = KVCacheDtype.NONE,
        is_indexer: bool = False,
        rotate_activation: bool = False,
        simulate_source_act_quant: bool = False,
    ):
        super().__init__()
        # Dimensions
        self.dim = mla_params.hidden_size
        self.head_dim = mla_params.qk_rope_head_dim + mla_params.qk_nope_head_dim
        self.rope_head_dim = mla_params.qk_rope_head_dim
        self.nope_head_dim = mla_params.qk_nope_head_dim

        # Compression config
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.state_dim = 2 * self.head_dim if self.overlap else self.head_dim

        # Cache config
        self.layer_idx = layer_idx
        self.kv_cache_dtype: KVCacheDtype = resolve_kv_cache_dtype(kv_cache_dtype)
        self.is_indexer = is_indexer
        self.rotate_activation = rotate_activation

        # Modules
        self.wkv_gate = Linear(
            self.dim,
            self.state_dim * 2,
            bias=False,
            dtype=dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.norm = RMSNorm(hidden_size=self.head_dim, eps=norm_eps, dtype=dtype)
        self.rotary_emb = deepseek_v4_rotary_embedding(
            pos_embd_params.rope,
            head_dim=self.rope_head_dim,
            is_neox=pos_embd_params.is_neox,
        )

        # Learnable absolute positional encoding for compression
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.state_dim, dtype=torch.float32))

        # DeepSeek-V4's own `Compressor.forward` ends with an activation
        # quantize/dequantize round trip -- blockwise-64 FP8 stored as BF16 for
        # the main compressor, E2M1 for the indexer's -- and projects in FP32.
        # The fused native postprocess models neither, and its E2M1 packing is
        # `__CUDA_ARCH__ >= 1000` only (`packE2M1x2` returns 0 below it), so
        # SM90 supplies both in Triton. Off by default: this is checkpoint
        # semantics, requested by the DeepSeek-V4 call sites that know they are
        # serving that checkpoint, not a property of every Compressor.
        self.simulate_source_act_quant = simulate_source_act_quant

    def _source_postprocess(
        self, kv_comp: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """RMSNorm -> RoPE -> Hadamard, rounding to the input dtype at each step.

        `compressor_postprocess_scatter` runs this whole chain in FP32 and
        rounds once, at the store. The source rounds three times ---
        ``kv = self.norm(kv.to(dtype))``, then an in-place
        ``apply_rotary_emb`` on that BF16 tensor, then ``rotate_activation``,
        which asserts BF16 --- and the difference is not academic: the
        indexer's next step is an E2M1 quantiser whose levels are 25-50% apart,
        so a ~4e-3 BF16 discrepancy pushes roughly half a percent of values one
        level away from the source, which is enough to break exact top-k at
        the checkpoint's 512 width. Measured on a 2304-token prefill: 377 of
        73728 values differed, and 152 of 253 deciding query rows selected a
        different slot set.

        Measured on the real checkpoint, the main compressor is not exempt
        either: its blockwise-64 FP8 levels are ~6% apart, and layer 3's cached
        rows scored rel_max_abs 3.707e-02 against the source through the fused
        kernel and are *bit-exact* through this chain. Both compressors use it
        on SM90; SM100/103 keeps the fused path in both cases.

        The norm is `sm90_quant.source_rms_norm`, not `self.norm`: TensorRT-LLM's
        RMSNorm rounds the normalised value to BF16 before multiplying by a BF16
        weight, while the checkpoint multiplies in FP32 and rounds once. Using
        `self.norm` here scored 7.4e-02 on the same rows -- worse than the fused
        kernel it was meant to replace.
        """
        kv = sm90_quant.source_rms_norm(kv_comp, self.norm.weight, self.norm.variance_epsilon)
        num_tokens = kv.shape[0]
        # Padded rows carry whatever the metadata buffer last held; they are
        # never scattered, but the RoPE table lookup still has to stay in
        # range. Clamped out of place: `position_ids` is a metadata buffer the
        # rest of the forward also reads.
        table = self.rotary_emb.rotary_cos_sin
        positions = position_ids.clamp(0, table.shape[0] - 1)
        torch.ops.trtllm.mla_rope_inplace(
            kv.view(num_tokens, 1, self.head_dim),
            positions,
            table,
            1,
            self.nope_head_dim,
            self.rope_head_dim,
            False,
            self.rotary_emb.is_neox,
        )
        return sm90_quant.hadamard_rotate(kv) if self.rotate_activation else kv

    def forward(
        self,
        x: torch.Tensor,
        metadata: "DeepseekV4TrtllmAttentionMetadata",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for paged KV compression.

        Args:
            x: Input tensor [num_tokens, dim]
            metadata: Attention metadata with cache info

        Returns:
            (kv_data, scale) tuple:
            - default / fp8_pertensor main compressor: (kv_comp, None)
            - default indexer:                         (kv_out, None)  bf16
            - fp8_blockwise indexer:                   (fp8_output, fp32 scale)
            - mxfp4 indexer:                           (fp4_output, ue8m0 scale)
            - no compressed tokens:                    (None, None)
        """
        # Extract metadata
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        bsz = num_contexts + num_generations

        # Determine attention types based on whether this is an indexer compressor
        if self.is_indexer:
            compress_type = DeepseekV4AttentionType.INDEXER_COMPRESS
            kv_type = DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV
            score_type = DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE
        else:
            compress_type = DeepseekV4AttentionType.COMPRESS
            kv_type = DeepseekV4AttentionType.COMPRESSOR_KV
            score_type = DeepseekV4AttentionType.COMPRESSOR_SCORE

        # Get cache buffers
        kv_cache = metadata.kv_cache_manager.get_buffers(self.layer_idx, compress_type)
        paged_kv_state = metadata.kv_cache_manager.get_buffers(self.layer_idx, kv_type)
        paged_score_state = metadata.kv_cache_manager.get_buffers(self.layer_idx, score_type)

        # Get block tables
        local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        if self.is_indexer:
            block_table = metadata.indexer_k_cache_block_offsets
        else:
            block_table = metadata.compress_block_tables[self.compress_ratio]
        block_table_kv_state = metadata.sliding_block_tables[local_layer_idx, kv_type.value]
        block_table_score_state = metadata.sliding_block_tables[local_layer_idx, score_type.value]

        # Get tokens_per_block from cache manager
        # state_tokens_per_block: for compressor kv/score state caches (used in compress kernels)
        # compress_tokens_per_block: for compressed KV cache (used in scatter)
        state_tokens_per_block = metadata.kv_cache_manager.tokens_per_block
        compress_tokens_per_block = metadata.kv_cache_manager.compressed_block_sizes[self.layer_idx]

        # Get compression metadata
        cu_new_comp_kv = metadata.cu_new_comp_kv_cuda[self.compress_ratio]
        kv_lens = metadata.kv_lens_cuda_runtime
        total_num_comp_tokens = metadata.num_total_compressed_tokens[self.compress_ratio]
        num_comp_tokens = metadata.new_comp_kv_lens_cuda[self.compress_ratio][:bsz]
        max_ctx_comp_kv_lens = metadata.max_ctx_compressed_tokens[self.compress_ratio]

        # Project input to KV and score in the checkpoint dtype. The compressor
        # kernels accept bf16 or fp32 kv_score and convert values to fp32
        # internally for state updates and online-softmax accumulation.
        if self.simulate_source_act_quant:
            # `Compressor.forward` in inference/model.py upcasts before this
            # projection --- "# compression need fp32", with wkv/wgate declared
            # as FP32 Linears over BF16 checkpoint weights. A BF16 GEMM here
            # leaves a ~2e-2 relative gap that the compressor's own FP8
            # quantisation then amplifies to a whole FP8 step on any element
            # sitting near a decision boundary. The kernels already accept an
            # FP32 kv_score (KV_SCORE_ELEM_BYTES=4 is instantiated) and
            # accumulate in FP32 either way, so this only widens the input.
            #
            # Issued as the source's *two* projections rather than this
            # module's fused one, and that is not cosmetic: the source runs
            # `self.wkv(x)` and `self.wgate(x)` as separate [state_dim, dim]
            # GEMMs, and asking cuBLAS for one [2 * state_dim, dim] GEMM
            # instead changes the accumulation it picks. Measured on the real
            # checkpoint at layer 40, ratio 4, 257-token prefill: the fused
            # GEMM moves 17 of 32768 pooled values by one BF16 ULP, and the
            # blockwise-64 FP8 quantiser two steps later turns two of them
            # into a whole FP8 level -- rel_max_abs 5.198e-02 against the
            # registered 0.03. Driving the *same* TensorRT-LLM reduction from
            # the split projection is bit-exact against the source at every
            # layer measured (2, 20, 40, 41, 42); driving it from the fused
            # projection is not. Shallow layers happened to pass only because
            # none of their perturbed values sat on a quantiser boundary.
            xf = x.float()
            weight = self.wkv_gate.weight.float()
            kv_score = torch.cat(
                [
                    F.linear(xf, weight[: self.state_dim]),
                    F.linear(xf, weight[self.state_dim :]),
                ],
                dim=-1,
            )
        else:
            kv_score = F.linear(x.to(self.wkv_gate.weight.dtype), self.wkv_gate.weight)

        # Allocate output buffer
        kv_comp = torch.empty(total_num_comp_tokens, self.head_dim, device=x.device, dtype=x.dtype)

        # Run compression kernels
        if num_contexts > 0:
            torch.ops.trtllm.compressor_prefill_reduction(
                kv_score[:num_ctx_tokens],
                self.ape,
                paged_kv_state,
                paged_score_state,
                block_table_kv_state[:num_contexts],
                block_table_score_state[:num_contexts],
                kv_comp,
                kv_lens[:num_contexts],
                metadata.cached_token_lens_cuda[:num_contexts],
                metadata.cu_seq_lens_cuda,
                cu_new_comp_kv[: num_contexts + 1],
                num_contexts,
                state_tokens_per_block,
                self.head_dim,
                self.compress_ratio,
                max_ctx_comp_kv_lens,
            )

        if num_generations > 0:
            gen_kv_lens = kv_lens[num_contexts:]
            next_n = metadata.num_gen_tokens_per_seq
            # Pass full kv_score (not sliced) with the generation portion of
            # cu_seq_lens so the kernel reads at the correct absolute offsets.
            torch.ops.trtllm.compressor_paged_kv_compress(
                kv_score,
                self.ape,
                paged_kv_state,
                paged_score_state,
                block_table_kv_state[num_contexts:],
                block_table_score_state[num_contexts:],
                kv_comp,
                gen_kv_lens,
                metadata.cu_seq_lens_cuda[num_contexts:],
                cu_new_comp_kv[num_contexts:],
                num_generations,
                state_tokens_per_block,
                self.head_dim,
                self.compress_ratio,
                next_n,
            )

        # Scatter to cache with appropriate quantization (all modes fused)
        start_pos = metadata.past_kv_lens_cuda[self.compress_ratio][:bsz]
        total_tokens = kv_comp.shape[0]

        # Allocate optional returned postprocess buffers for indexer paths.
        kv_out = None
        quant_output = None
        scale_output = None
        # On SM90 the indexer runs the source's postprocess chain itself rather
        # than the fused kernel's; see `_source_postprocess` for why.
        sm90_indexer_fp4 = (
            self.is_indexer
            and self.simulate_source_act_quant
            and self.kv_cache_dtype == KVCacheDtype.FP8_BLOCKWISE
        )
        if self.is_indexer and not sm90_indexer_fp4:
            if self.kv_cache_dtype == KVCacheDtype.NONE:
                kv_out = torch.empty_like(kv_comp)
            elif self.kv_cache_dtype == KVCacheDtype.FP8_BLOCKWISE:
                num_scale_blocks = self.head_dim // 128
                quant_output = torch.empty(
                    total_tokens, self.head_dim, dtype=torch.uint8, device=kv_comp.device
                )
                scale_output = torch.empty(
                    total_tokens, num_scale_blocks, dtype=torch.float32, device=kv_comp.device
                )
            elif self.kv_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
                num_scale_blocks = self.head_dim // 32
                quant_output = torch.empty(
                    total_tokens, self.head_dim // 2, dtype=torch.uint8, device=kv_comp.device
                )
                scale_output = torch.empty(
                    total_tokens, num_scale_blocks, dtype=torch.uint8, device=kv_comp.device
                )

        position_ids = metadata.compressed_position_ids_cuda[self.compress_ratio][:total_tokens]
        compressed_mask = metadata.compressed_mask_cuda[self.compress_ratio][:total_tokens]

        if sm90_indexer_fp4:
            return sm90_quant.quantize_indexer_rows_to_fp8_cache(
                self._source_postprocess(kv_comp, position_ids),
                kv_cache,
                compressed_mask,
                cu_new_comp_kv,
                num_comp_tokens,
                start_pos,
                block_table,
                tokens_per_block=compress_tokens_per_block,
            )

        if self.simulate_source_act_quant and not self.is_indexer:
            # SM90 main compressor: the checkpoint's own postprocess chain.
            # `compressor_postprocess_scatter` runs norm and RoPE in FP32 and
            # rounds once at the store; the source rounds after the norm and
            # again after the RoPE, and the blockwise-64 FP8 step that follows
            # turns that difference into a whole FP8 level on any value near a
            # decision boundary. Measured against the real checkpoint's cached
            # rows: fused 1.835e-02 (ratio 4) and 3.707e-02 (ratio 128, over the
            # registered 0.03), this chain bit-exact on both.
            assert self.kv_cache_dtype == KVCacheDtype.NONE, (
                "the SM90 compressor source chain assumes the BF16 cache preset; "
                f"got {self.kv_cache_dtype.name}, which already quantizes the row"
            )
            sm90_quant.write_source_compressed_rows(
                self._source_postprocess(kv_comp, position_ids),
                kv_cache,
                compressed_mask,
                cu_new_comp_kv,
                num_comp_tokens,
                start_pos,
                block_table,
                total_tokens=total_tokens,
                tokens_per_block=compress_tokens_per_block,
                nope_dim=self.nope_head_dim,
            )
            return kv_comp, None

        # Fused postprocess + scatter: RMSNorm + RoPE + Hadamard + paged cache write
        torch.ops.trtllm.compressor_postprocess_scatter(
            kv_comp,
            kv_out,
            self.norm.weight,
            self.norm.variance_epsilon,
            self.rotary_emb.rotary_cos_sin,
            position_ids,
            self.nope_head_dim,
            self.rope_head_dim,
            kv_cache,
            num_comp_tokens,
            cu_new_comp_kv,
            start_pos,
            block_table,
            compressed_mask,
            compress_tokens_per_block,
            int(self.kv_cache_dtype),
            self.rotate_activation,
            quant_output,
            scale_output,
        )

        if quant_output is not None:
            if self.kv_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
                return quant_output.view(torch.float4_e2m1fn_x2), scale_output
            return quant_output.view(torch.float8_e4m3fn), scale_output
        if kv_out is not None:
            return kv_out, None
        return kv_comp, None
