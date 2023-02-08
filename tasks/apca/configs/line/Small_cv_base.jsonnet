local data_base_path = "data/apca/Small/cv/cv_1/";
local code_embed_dim = 768;
local code_out_dim = 768;
local fusion_out_dim = code_out_dim;
local train_data_file = 'train_patches.pkl';
local test_data_file = 'test_patches.pkl';

local pretrained_model = "microsoft/codebert-base";
local code_namespace = "code_tokens";

local additional_special_tokens = [];
local seed = 6324;

local line_max_tokens = 64;
local max_lines = 24;
local diff_max_tokens = 256;
local keep_tokenizer_head_tail_token = false;

{
  random_seed: seed,
  numpy_seed: seed,
  pytorch_seed: seed,

  dataset_reader: {
    type: "apca_imp_flat_line_align",
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
    line_max_tokens: line_max_tokens,
    diff_max_tokens: diff_max_tokens,
    max_lines: max_lines,
    line_separator: null,
    empty_line_placeholder: null,
    insert_empty_line_placeholder: false,
    use_op_mask: false,
    keep_tokenizer_head_tail_token: keep_tokenizer_head_tail_token,
    jointly_align_add_del_lines: false,
    align_equal_lines: false
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
    fusion: {
      type: "flat_line_align_joint_concat_attention",
      encoder_feature_dim: code_out_dim,
      // transformer
      transformer_layer: 4,
      transformer_dim_feedforward: 1024,
      transformer_head: 4,
      transformer_dropout: 0.5,
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
    },
    classifier: {
        type: "linear_sigmoid",
        in_feature_dim: fusion_out_dim,
        hidden_dims: [512],
        activations: ['relu'],
        dropouts: [0.5],
        ahead_feature_dropout: 0.5,
    },
    loss_func: {
        type: "bce"
    },
    metric: {
        type: "f1",
        positive_label: 1
    }
  },
  data_loader: {
    batch_size: 16,
    shuffle: true,
  },
  trainer: {
    num_epochs: 20,
    patience: null,
    cuda_device: 3,
    validation_metric: "+f1",
    optimizer: {
      type: "adam",
      lr: 1e-5
    },
    num_gradient_accumulation_steps: 2,
    callbacks: [
      { type: "epoch_print" },
      { type: "model_param_stat" },
    ],
    checkpointer: null,     // checkpointer is set to null to avoid saving model state at each episode
  },

}