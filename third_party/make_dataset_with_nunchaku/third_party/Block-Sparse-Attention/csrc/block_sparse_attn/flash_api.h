#pragma once

#include "Tensor.h"

std::vector<Tensor>
mha_fwd(Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        // c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        // c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const bool return_softmax
        // c10::optional<at::Generator> gen_
        );

std::vector<Tensor>
mha_varlen_fwd(Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               // std::optional<Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               Tensor &cu_seqlens_q,  // b+1
               Tensor &cu_seqlens_k,  // b+1
               // std::optional<Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               // std::optional<Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               // std::optional<Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const bool return_softmax);

std::vector<Tensor>
mha_fwd_block(const Tensor &q,         // total_q x num_heads x head_size, total := \sum_{i=0}^{b} s_i
              const Tensor &k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              const Tensor &v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              const Tensor &cu_seqlens_q,  // b+1
              const Tensor &cu_seqlens_k,  // b+1
              const int m_block_dim,
              const int n_block_dim,
              Tensor &head_mask_type, // (num_heads)
              std::optional<Tensor> streaming_info_, // (num_heads, 2)
              std::optional<Tensor> row_blockmask_,   // (batch_size, num_blocksparse_heads, max_seqlen_m / m_block_dim, k)
              const int max_seqlen_q_,
              const int max_seqlen_k_,
              const float p_dropout,
              const float softmax_scale,
              const bool is_causal,
              const bool exact_streaming,
              const bool return_softmax,
              int window_size_left,
              int window_size_right);