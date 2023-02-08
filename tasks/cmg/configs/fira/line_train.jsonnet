local data_base_path = "data/cmg/fira/";
local msg_embed_dim = 768;
local code_embed_dim = 768;
local code_encoder_dim = 768;
local code_encoder_out_dim = 768;
local fusion_out_dim = 2 * code_encoder_out_dim;
local bidirectional = true;
local source_token_namespace = "code_tokens";
local target_token_namespace = "code_tokens";
local pretrained_model = "microsoft/codebert-base";
local bleu_excluded_tokens = ['@start@', '@end@', '<s>', '</s>', '<pad>', '', '@@PADDING@@', '@@UNKNOWN@@'];
local additional_special_tokens = ["<n42>", "<n54>", "<n19>", "<a22>", "<n24>", "<n47>", "<n22>", "<f5>", "<n75>", "<a0>", "<f2>", "<n50>", "<c2>", "<n29>", "<a6>", "<a15>", "<f3>", "<n11>", "<n27>", "<n34>", "<n37>", "<a18>", "<n38>", "<n21>", "<n51>", "<n41>", "<n64>", "<n77>", "<n65>", "<n49>", "<n28>", "<n5>", "<n52>", "<n17>", "<a1>", "<a9>", "<n39>", "<n9>", "<n33>", "<n23>", "<a20>", "<n76>", "<f9>", "<f7>", "<n14>", "<n45>", "<a14>", "<n7>", "<a10>", "<n1>", "<c3>", "<a19>", "<n68>", "<f11>", "<n8>", "<n60>", "<f6>", "<c0>", "<n30>", "<f0>", "<n32>", "<n16>", "<a16>", "<f12>", "<c8>", "<c5>", "<n59>", "<n53>", "<c4>", "<n79>", "<n31>", "<n4>", "<n43>", "<n2>", "<a21>", "<a23>", "<f8>", "<n67>", "<a17>", "<n56>", "<a12>", "<n55>", "<n72>", "<a13>", "<n3>", "<a24>", "<f13>", "<n15>", "<n20>", "<n46>", "<f10>", "<a4>", "<n35>", "<n25>", "<a5>", "<f4>", "<n62>", "<n66>", "<n10>", "<n61>", "<n63>", "<c7>", "<n44>", "<a2>", "<n6>", "<n48>", "<a7>", "<n0>", "<n73>", "<n74>", "<n18>", "<n78>", "<n71>", "<a3>", "<a25>", "<a8>", "<n26>", "<n69>", "<c1>", "<n40>", "<c6>", "<a11>", "<f1>", "<n12>", "<n70>", "<n58>", "<n36>", "<n13>", "<n80>", "<n57>"];

local code_max_tokens = 300;
local msg_max_tokens = 50;
local line_max_tokens = 64;
local max_lines = 32;
local target_vocab_size = 50265 + 130;    # codebert-base vocab size + additional special tokens

local keep_tokenizer_head_tail_token = false;
local debug = false;


{
  dataset_reader: {
    type: "cmg_fix_line_align_fira",
    line_code_tokenizer: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      max_length: code_max_tokens,
      tokenizer_kwargs: {
        additional_special_tokens: additional_special_tokens
      }
    },
    diff_code_tokenizer: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      max_length: code_max_tokens,
      tokenizer_kwargs: {
        additional_special_tokens: additional_special_tokens
      }
    },
    code_indexers: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      namespace: source_token_namespace,
      tokenizer_kwargs: {
        additional_special_tokens: additional_special_tokens
      },
    },
    msg_tokenizer: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      max_length: msg_max_tokens,
      tokenizer_kwargs: {
        additional_special_tokens: additional_special_tokens
      },
    },
    msg_token_indexer: {
      type: "pretrained_transformer",
      model_name: pretrained_model,
      namespace: target_token_namespace,
      tokenizer_kwargs: {
        additional_special_tokens: additional_special_tokens
      },
    },
    line_max_tokens: line_max_tokens,
    diff_max_tokens: code_max_tokens,
    msg_max_tokens: msg_max_tokens,
    max_lines: max_lines,
    use_diff_as_input: true,

    // Op-mask configurations
    use_op_mask: false,
    op_mask_attend_first_token: true,

    code_lower: false,
    msg_lower: true,
    code_namespace: source_token_namespace,
    msg_namespace: target_token_namespace,

    // Line-align configurations
    line_separator: "\n",           # Join the add/del lines with this for splitting into lines when line-aligning
    insert_empty_line_placeholder: false,
    empty_line_placeholder: null,
    keep_tokenizer_head_tail_token: keep_tokenizer_head_tail_token,
    jointly_align_add_del_lines: false,
    align_equal_lines: false,
    line_extractor_version: 'fira_v1',

    // Fira-data configurations
    clean_raw_diff: true,
    clean_raw_diff_version: "v3",
    add_msg_start_end_tokens: false,
    use_fira_lemmatization: true,
    use_identifier_placeholder: true,
    keep_identifer_before_placeholder: true,
    minimum_replace_length: 3,

    debug: debug,
  },
  train_data_path: [
        data_base_path + "train_diff.json",
        data_base_path + "train_msg.json"
  ],
  validation_data_path: [
        data_base_path + "validate_diff.json",
        data_base_path + "validate_msg.json"
  ],
  model: {
    type: "imp_cc_hybrid_seq2seq",
    code_embedder: {
      token_embedders: {
        code_tokens: {
          type: "pretrained_transformer",
          model_name: pretrained_model,
          train_parameters: true,
          tokenizer_kwargs: {
              additional_special_tokens: additional_special_tokens
          },
        }
      }
    },
    code_encoder: {
      type: "pass_through",
      input_dim: code_embed_dim
    },

    fusion: {
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
    msg_decoder: {
        type: "cc_auto_regressive_seq_decoder",
        decoder_net: {
            type: "transformer",
            decoding_dim: fusion_out_dim,
            target_embedding_dim: msg_embed_dim,
            feedforward_hidden_dim: 1536,
            num_layers: 4,
            num_attention_heads: 8,
            use_positional_encoding: true,
            dropout_prob: 0.1,
            residual_dropout_prob: 0.2,
            attention_dropout_prob: 0.1
        },
        target_embedder: {
            embedding_dim: msg_embed_dim,
            num_embeddings: target_vocab_size,
            projection_dim: null,
            padding_index: 1,
            trainable: true,
            vocab_namespace: target_token_namespace
        },
        target_namespace: target_token_namespace,
        scheduled_sampling_ratio: 0,
        target_vocab_size: target_vocab_size,
        beam_search: {
            beam_size: 5,
            max_steps: 50,
        },
        tensor_based_metric: {
            type: "bleu_norm",
            token_namespace: target_token_namespace,
            cal_bleu_per_batch: true,
            subtoken_merge_method: "codebert",
            excluded_tokens: bleu_excluded_tokens,
        },
        msg_start_token_index: 0,   # start_token: <s>
        msg_end_token_index: 2,     # end_token: </s>

        copy_mode: 'copy_by_id',
        encoder_output_dim: fusion_out_dim,
        copynet_hidden_dim: 2048,
        excluded_copy_indices: [0,1,2,3], # <s>, </s>, <pad>, <unk>
    },
  },
  data_loader: {
    batch_size: 6
  },
  validation_data_loader: {
    batch_size: 30,
  },
  trainer: {
    num_epochs: 10,
    patience: null,
    cuda_device: 2,
    validation_metric: "+BLEU",
    optimizer: {
      type: "adam",
      lr: 3e-5
    },
    learning_rate_scheduler: {
        type: "multi_step",
        milestones: [50],
        gamma: 0.5
    },
    num_gradient_accumulation_steps: 5,
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