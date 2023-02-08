local data_base_path = "data/apca/Large/cv/cv_1/";
local code_embed_dim = 768;
local code_out_dim = 768;
local fusion_out_dim = code_out_dim;
local train_data_file = 'train_patches.pkl';

local pretrained_model = "microsoft/codebert-base";
local code_namespace = "code_tokens";

local additional_special_tokens = [];

local seed = 6324;
local max_tokens = 256;

{
  random_seed: seed,
  numpy_seed: seed,
  pytorch_seed: seed,

  dataset_reader: {
    type: "apca_imp_flat",
    code_tokenizer: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      max_length: max_tokens,
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
    max_tokens: max_tokens,
    use_op_mask: true,
    op_mask_attend_first_token: true,
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
      type: "op_mask_joint_concat_attention",
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

      // 3. query-back attention
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
    num_epochs: 5,
    patience: null,
    cuda_device: 3,                 # todo: change this if needed
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