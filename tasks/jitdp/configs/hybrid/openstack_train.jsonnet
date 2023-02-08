local data_base_path = "data/jitdp/openstack/";
local msg_embed_dim = 128;
local msg_encoder_dim = 128;
local code_embed_dim = 768;     // This value should be set to 768 when using CodeBERT
local code_encoder_dim = 768;
local code_out_dim = 768;
local fusion_out_dim = code_out_dim;
local msg_out_dim = 256;
local train_data_file = 'train.pkl';

local pretrained_model = "microsoft/codebert-base";
local code_namespace = "code_tokens";
local msg_namespace = "msg_tokens";

local additional_special_tokens = [];

local line_max_tokens = 64;
local diff_max_tokens = 384;
local max_lines = 64;
local keep_tokenizer_head_tail_token = false;

{
  dataset_reader: {
    type: "jit_dp_imp_flat_line_align",
    code_tokenizer: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      max_length: line_max_tokens,
      tokenizer_kwargs: {
        additional_special_tokens: additional_special_tokens
      }
    },
    code_indexer: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      namespace: code_namespace,
      tokenizer_kwargs: {
        additional_special_tokens: additional_special_tokens
      }
    },

    // ctrl '<s>' and '</s>' is line-wise or diff-wise
    keep_tokenizer_head_tail_token: keep_tokenizer_head_tail_token,
    line_separator: null, //'Ċ',
    empty_line_placeholder: null,
    insert_empty_line_placeholder: false,
    jointly_align_add_del_lines: false,

    line_max_tokens: line_max_tokens,
    diff_max_tokens: diff_max_tokens,
    max_lines: max_lines,

    use_op_mask: true,
    op_mask_attend_first_token: true,
    include_LA_metric: false,
    include_LD_metric: false,
    lower_code: false,
    lower_msg: true,
  },
  train_data_path: data_base_path + train_data_file,
  model: {
    type: "imp_seqin_classifier",
    code_embedder: {
      token_embedders: {
        code_tokens: {
          type: "pretrained_transformer",
          model_name: pretrained_model,
          train_parameters: true,
         tokenizer_kwargs: {
           additional_special_tokens: additional_special_tokens
         }
        }
      }
    },
    code_encoder: {
        type: "pass_through",
        input_dim: code_embed_dim,
    },
    msg_embedder: {
      token_embedders: {
        msg_tokens: {
          type: "embedding",
          embedding_dim: msg_embed_dim,
          vocab_namespace: msg_namespace,
          trainable: true
        }
      }
    },
    msg_encoder: {
        type: "lstm",
        input_size: msg_embed_dim,
        hidden_size: msg_encoder_dim,
        num_layers: 2,
        dropout: 0.5,
        bidirectional: true,
    },
    fusion: {
      type: "op_mask_line_align_hybrid_add_wrapper",
      encoder_feature_dim: code_out_dim,
      reproject_before_add: true,
      reproject_dropout: 0.3,

      op_mask_joint_attention:{
          // 1. transformer
          encoder_feature_dim: code_out_dim,
          transformer_layer: 2,
          transformer_dim_feedforward: 1024,
          transformer_head: 4,
          transformer_dropout: 0.3,
          transformer_activation: "relu",
          keep_unmasked_order: false,
          unmasked_seq_max_len: 128,

          // 2. reduce
          seq_reduce: {
            type: 'avg'
          },
          reduce_before_attention: true,

          // 3. query attention
          query_attention: {
            type: "multi_head",
            input_size: code_out_dim,
            dropout: 0.2,
            head_nums: 4
          },

          // 4. merge
          merge_method: 'add',

          // 5. out proj and norm
          out_proj_and_norm: false,
          out_proj_in_dim: null,
          out_proj_out_dim: null,
      },

      line_align_joint_attention:{
          encoder_feature_dim: code_out_dim,
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
            input_size: code_out_dim,
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
    classifier: {
        type: "linear_sigmoid",
        in_feature_dim: fusion_out_dim + msg_out_dim,
        hidden_dims: [512],
        activations: ['relu'],
        dropouts: [0.5],
        ahead_feature_dropout: 0.5
    },
    loss_func: {
        type: "bce"
    },
    metric: {
        type: "auc"
    }
  },
  data_loader: {
    batch_size: 8,
    shuffle: true,
  },
  trainer: {
    num_epochs: 3,
    patience: null,
    cuda_device: 0,
    validation_metric: "+Auc",
    optimizer: {
      type: "adam",
      lr: 1e-5
    },
    num_gradient_accumulation_steps: 4,
    callbacks: [
      { type: "epoch_print" },
      { type: "model_param_stat" },
    ],
    checkpointer: null,     // checkpointer is set to null to avoid saving model state at each episode
  },

}