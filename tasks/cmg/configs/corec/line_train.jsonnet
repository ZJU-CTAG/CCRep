local data_base_path = "data/cmg/corec/";
local msg_embed_dim = 512;
local code_embed_dim = 768;
local code_encoder_dim = 768;
local code_encoder_out_dim = 768;
local fusion_out_dim = 2 * code_encoder_out_dim;
local bidirectional = true;
local source_token_namespace = "code_tokens";
local target_token_namespace = "msg_tokens";
local pretrained_model = "microsoft/codebert-base";

local diff_max_tokens = 100;
local line_max_tokens = 64;
local max_lines = 32;
local msg_max_tokens = 32;
local keep_tokenizer_head_tail_token = false;

{
  dataset_reader: {
    type: "cmg_fix_line_align_v2",
    line_code_tokenizer: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      max_length: line_max_tokens
    },
    diff_code_tokenizer: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      max_length: diff_max_tokens,
    },
    code_indexers: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      namespace: source_token_namespace
    },
    line_max_tokens: line_max_tokens,
    diff_max_tokens: diff_max_tokens,
    max_lines: max_lines,
    msg_max_tokens: msg_max_tokens,

    use_diff_as_input: true,
    use_op_mask: false,
    op_mask_attend_first_token: true,
    code_lower: false,
    msg_lower: true,

    line_separator: " ",
    empty_line_placeholder: null,
    insert_empty_line_placeholder: false,
    keep_tokenizer_head_tail_token: keep_tokenizer_head_tail_token,
    jointly_align_add_del_lines: false,
    align_equal_lines: false,
    line_extractor_version: 'v1',
  },
  train_data_path: [
        data_base_path + "cleaned_train.diff",
        data_base_path + "cleaned_train.msg"
  ],
  validation_data_path: [
        data_base_path + "cleaned_validate.diff",
        data_base_path + "cleaned_validate.msg"
  ],
  model: {
    type: "hybrid_cc_simple_seq2seq",
    source_embedder: {
      token_embedders: {
        code_tokens: {
          type: "pretrained_transformer",
          model_name: pretrained_model,
          train_parameters: true,
//          max_length: code_max_tokens
        }
      }
    },
    encoder: {
      type: "pass_through",
      input_dim: code_embed_dim
    },

    diff_siso_fusion: {
      type: "diff_flat_line_align",
      encoder_feature_dim: code_encoder_out_dim,
      fusion_as_decoder_input_method: "cat",
      out_dim: 2 * code_encoder_out_dim,

      sivo_FLA_fusion: {
          encoder_feature_dim: code_encoder_out_dim,
          // transformer
          transformer_layer: 2,
          transformer_dim_feedforward: 1024,
          transformer_head: 4,
          transformer_dropout: 0.3,
          transformer_activation: "relu",

          // line-align
          line_max_tokens: line_max_tokens,
          max_lines: max_lines,

          // reduce
          line_token_seq_reduce: { type: 'cls' },
          line_feature_seq_reduce: { type: 'avg' },

          // query-back
          query_attention: {
            type: "multi_head",
            input_size: code_encoder_out_dim,
            dropout: 0.2,
            head_nums: 4
          },
          query_attention_merge_method: 'add',

          // others
          positional_encoding: 'sincos',
          drop_tokenizer_head_tail_token: !keep_tokenizer_head_tail_token,
          trainable_token_separator: true,
          insert_trainable_cls_token: true,
          out_proj_dim: null
      }
    },

    beam_search: {
        beam_size: 5,
        max_steps: 50,
    },
    attention: {
        type: "additive",
        vector_dim: fusion_out_dim,   # query dim, same as decoder hidden_dim, which is set to encoder_out_dim
        matrix_dim: fusion_out_dim,   # key/value dim, same as encoder out dim
    },
    target_namespace: target_token_namespace,
    target_embedding_dim: msg_embed_dim,
    scheduled_sampling_ratio: 0,
    use_bleu: true,
    target_decoder_layers: 2,
  },
  data_loader: {
    batch_size: 16
  },
  validation_data_loader: {
    batch_size: 32,
  },
  trainer: {
    num_epochs: 300,
    patience: null,
    cuda_device: 1,
    validation_metric: "+BLEU",
    optimizer: {
      type: "adam",
      lr: 3e-5
    },
    learning_rate_scheduler: {
        type: "multi_step",
        milestones: [140, 200],
        gamma: 0.5
    },
    num_gradient_accumulation_steps: 2,
    callbacks: [
      {
        type: "epoch_print"
      },
      {
        type: "model_param_stat",
      },
    ],
    checkpointer: null,       // disable epoch-wise checkpoint saving, for disk usage consideration
  },
}