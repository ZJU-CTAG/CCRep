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

local code_max_tokens = 100;
local msg_max_tokens = 32;

{
  dataset_reader: {
    type: "cmg_hybrid_fix_v2",
    code_tokenizer: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      max_length: code_max_tokens
    },
    code_indexers: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      namespace: source_token_namespace
    },
    code_max_tokens: code_max_tokens,
    msg_max_tokens: msg_max_tokens,
    use_op_seq: false,
    use_op_mask: true,
    op_mask_attend_first_token: true,
    code_lower: false,
    msg_lower: true,
    code_align: false,
    code_namespace: source_token_namespace,
    msg_namespace: target_token_namespace,

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
        }
      }
    },
    encoder: {
      type: "pass_through",
      input_dim: code_embed_dim
    },

    diff_siso_fusion: {
      type: "diff_op_mask_fix",
      encoder_feature_dim: code_encoder_out_dim,
      fusion_as_decoder_input_method: "cat",
      output_dim: 2 * code_encoder_out_dim,

      sivo_op_mask_fusion: {
          // 1. transformer
          encoder_feature_dim: code_encoder_out_dim,
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
            input_size: code_encoder_out_dim,
            dropout: 0.2,
            head_nums: 4
          },

          // 4. merge
          merge_method: 'add',

          // 5. out proj and norm
          output_dim: fusion_out_dim,
          out_proj_and_norm: false,
          out_proj_in_dim: null,
          out_proj_out_dim: null,
      }
    },

    beam_search: {
        beam_size: 5,
        max_steps: 50,
    },
    attention: {
        type: "additive",
        vector_dim: fusion_out_dim,
        matrix_dim: fusion_out_dim,
    },
    target_namespace: target_token_namespace,
    target_embedding_dim: msg_embed_dim,
    scheduled_sampling_ratio: 0,
    use_bleu: true,
    target_decoder_layers: 2,
  },
  data_loader: {
    batch_size: 32
  },
  validation_data_loader: {
    batch_size: 32,
  },
  trainer: {
    num_epochs: 300,
    patience: null,
    cuda_device: 0,
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
    num_gradient_accumulation_steps: 1,
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