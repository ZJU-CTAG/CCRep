import warnings
from typing import Dict, List, Tuple, Optional
from overrides import overrides

import numpy
import torch
import torch.nn.functional as F
from torch.nn import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric
from allennlp.common import Lazy

from allennlp_models.generation.modules.decoder_nets.decoder_net import DecoderNet
from allennlp_models.generation.modules.seq_decoders.seq_decoder import SeqDecoder

from core.comp.nn.copynet import FiraCopyNet
from utils.allennlp_utils.copy_utils import generate_extended_target_label_by_id, compute_copy_extended_log_probs, \
    revert_extend_predictions_by_id
from utils.allennlp_utils.loss_utils import sequence_cross_entropy_with_log_probs


@SeqDecoder.register("cc_auto_regressive_seq_decoder")
class CcAutoRegressiveSeqDecoder(SeqDecoder):
    """
    An autoregressive decoder that can be used for most seq2seq tasks.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    decoder_net : `DecoderNet`, required
        Module that contains implementation of neural network for decoding output elements
    target_embedder : `Embedding`
        Embedder for target tokens.
    target_namespace : `str`, optional (default = `'tokens'`)
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    beam_search : `BeamSearch`, optional (default = `Lazy(BeamSearch)`)
        This is used to during inference to select the tokens of the decoded output sequence.
    tensor_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    scheduled_sampling_ratio : `float` optional (default = `0.0`)
        Defines ratio between teacher forced training and real output usage. If its zero
        (teacher forcing only) and `decoder_net`supports parallel decoding, we get the output
        predictions in a single forward pass of the `decoder_net`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        decoder_net: DecoderNet,
        target_embedder: Embedding,
        target_namespace: str = "tokens",
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch),
        tie_output_embedding: bool = False,
        scheduled_sampling_ratio: float = 0,
        label_smoothing_ratio: Optional[float] = None,
        tensor_based_metric: Metric = None,
        token_based_metric: Metric = None,
        # msg_token_id_key: str = 'tokens',
        # If you use shared embedding with pretrained Transformer, you may need to set this to "token_ids"
        target_vocab_size: Optional[int] = None,
        msg_start_token_index: Optional[int] = None,
        msg_end_token_index: Optional[int] = None,
        copy_mode: str = 'none',                    # Only need for copynet
        encoder_output_dim: Optional[int] = None,   # Only need for copynet
        copynet_hidden_dim: Optional[int] = None,   # Only need for copynet
        excluded_copy_indices: List[int] = [],      # Only need for copynet
        copy_count_stat_alpha: float = 0.5,         # Deprecated
        # use_shared_embedding: bool = False,
        target_projecting_input_dim: Optional[int] = None,  # Used for expanded (like cat) decoder output
        **kwargs
    ) -> None:
        super().__init__(target_embedder)

        self._vocab = vocab
        self.target_vocab_size = target_vocab_size
        self.excluded_copy_indices = excluded_copy_indices

        # Decodes the sequence of encoded hidden states into e new sequence of hidden states.
        self._decoder_net = decoder_net
        self._target_namespace = target_namespace
        self._label_smoothing_ratio = label_smoothing_ratio

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self._vocab.get_token_index(START_SYMBOL, self._target_namespace) if msg_start_token_index is None else msg_start_token_index
        self._end_index = self._vocab.get_token_index(END_SYMBOL, self._target_namespace) if msg_end_token_index is None else msg_end_token_index
        # For backwards compatibility, check if beam_size or max_decoding_steps were passed in as
        # kwargs. If so, update the BeamSearch object before constructing and raise a DeprecationWarning
        deprecation_warning = (
            "The parameter {} has been deprecated."
            " Provide this parameter as argument to beam_search instead."
        )
        beam_search_extras = {}
        if "beam_size" in kwargs:
            beam_search_extras["beam_size"] = kwargs["beam_size"]
            warnings.warn(deprecation_warning.format("beam_size"), DeprecationWarning)
        if "max_decoding_steps" in kwargs:
            beam_search_extras["max_steps"] = kwargs["max_decoding_steps"]
            warnings.warn(deprecation_warning.format("max_decoding_steps"), DeprecationWarning)
        self._beam_search = beam_search.construct(
            end_index=self._end_index, vocab=self._vocab, **beam_search_extras
        )

        target_vocab_size = self._vocab.get_vocab_size(self._target_namespace) if target_vocab_size is None else target_vocab_size

        if self.target_embedder.get_output_dim() != self._decoder_net.target_embedding_dim:
            raise ConfigurationError(
                "Target Embedder output_dim doesn't match decoder module's input."
            )

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self.target_projecting_input_dim = target_projecting_input_dim
        output_projecting_dim = target_projecting_input_dim or self._decoder_net.get_output_dim()
        self._output_projection_layer = Linear(
            output_projecting_dim, target_vocab_size
        )

        if tie_output_embedding:
            if self._output_projection_layer.weight.shape != self.target_embedder.weight.shape:
                raise ConfigurationError(
                    "Can't tie embeddings with output linear layer, due to shape mismatch"
                )
            self._output_projection_layer.weight = self.target_embedder.weight

        # These metrics will be updated during training and validation
        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric

        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        self.copy_mode = copy_mode
        if copy_mode != 'none':
            self.copy_net = FiraCopyNet(encoder_output_dim, output_projecting_dim, copynet_hidden_dim)
        self.copy_stat = {'copy_pred_count': 0, 'copy_label_count': 0}

    def update_copy_stat(self, key, count):
        self.copy_stat[key] += int(count)

    def reset_copy_stat(self):
        self.copy_stat = {'copy_pred_count': 0, 'copy_label_count': 0}

    def maybe_expand_decoder_output_with_fusion(self,
                                                decoder_output: torch.Tensor,
                                                state: Dict[str, torch.Tensor]):
        # Cat fused output and decoder output to jointly predict next word if given.
        if 'fused_cat_context' in state:
            cc_fused_features = state['fused_cat_context']
            if decoder_output.ndim == 3:
                seq_len = decoder_output.size(1)
                cc_fused_features = cc_fused_features.unsqueeze(1).expand(-1, seq_len, -1)
            decoder_output = torch.cat((decoder_output, cc_fused_features), dim=-1)
        return decoder_output

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the beam search, does beam search and returns beam search results.
        """
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        if self.copy_mode == 'none':
            beam_search_step_func = self.take_step
        else:
            beam_search_step_func = self.take_step_with_copy

        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, beam_search_step_func
        )
        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _forward_loss(
            self,
            state: Dict[str, torch.Tensor],
            target_tokens: TextFieldTensors,
            source_tokens: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (batch_size, max_target_sequence_length)
        targets = util.get_token_ids_from_text_field_tensors(target_tokens)

        # Prepare embeddings for targets. They will be used as gold embeddings during decoder training
        # shape: (batch_size, max_target_sequence_length, embedding_dim)
        target_embedding = self.target_embedder(targets)

        # shape: (batch_size, max_target_batch_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)

        if self._scheduled_sampling_ratio == 0 and self._decoder_net.decodes_parallel:
            _, decoder_output = self._decoder_net(
                previous_state=state,
                previous_steps_predictions=target_embedding[:, :-1, :],
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
                previous_steps_mask=target_mask[:, :-1],
            )
            decoder_output = self.maybe_expand_decoder_output_with_fusion(decoder_output, state)

            # shape: (group_size, max_target_sequence_length, num_classes)
            logits = self._output_projection_layer(decoder_output)
        else:
            decoder_step_outputs = []
            batch_size = source_mask.size()[0]
            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1

            # Initialize target predictions with the start index.
            # shape: (batch_size,)
            last_predictions = source_mask.new_full(
                (batch_size,), fill_value=self._start_index, dtype=torch.long
            )

            # shape: (steps, batch_size, target_embedding_dim)
            steps_embeddings = torch.Tensor([])

            step_logits: List[torch.Tensor] = []

            for timestep in range(num_decoding_steps):
                if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                    # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                    # during training.
                    # shape: (batch_size, steps, target_embedding_dim)
                    state["previous_steps_predictions"] = steps_embeddings

                    # shape: (batch_size, )
                    effective_last_prediction = last_predictions
                else:
                    # shape: (batch_size, )
                    effective_last_prediction = targets[:, timestep]

                    if timestep == 0:
                        state["previous_steps_predictions"] = torch.Tensor([])
                    else:
                        # shape: (batch_size, steps, target_embedding_dim)
                        state["previous_steps_predictions"] = target_embedding[:, :timestep]

                # shape: (batch_size, num_classes)
                output_projections, state, decoder_step_output = self._prepare_output_projections(
                    effective_last_prediction, state
                )

                # list of tensors, shape: (batch_size, 1, num_classes)
                step_logits.append(output_projections.unsqueeze(1))
                decoder_step_outputs.append(decoder_step_output.unsqueeze(1))

                # shape (predicted_classes): (batch_size,)
                _, predicted_classes = torch.max(output_projections, 1)

                # shape (predicted_classes): (batch_size,)
                last_predictions = predicted_classes

                # shape: (batch_size, 1, target_embedding_dim)
                last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

                # This step is required, since we want to keep up two different prediction history: gold and real
                if steps_embeddings.shape[-1] == 0:
                    # There is no previous steps, except for start vectors in `last_predictions`
                    # shape: (group_size, 1, target_embedding_dim)
                    steps_embeddings = last_predictions_embeddings
                else:
                    # shape: (group_size, steps_count, target_embedding_dim)
                    steps_embeddings = torch.cat([steps_embeddings, last_predictions_embeddings], 1)

            decoder_output = torch.cat(decoder_step_outputs, 1)
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

        if self.copy_mode == 'copy_by_id':
            sources = util.get_token_ids_from_text_field_tensors(source_tokens)
            ext_targets, copy_mask = generate_extended_target_label_by_id(sources,
                                                                          targets, self.target_vocab_size,
                                                                          self.excluded_copy_indices)
            if not self.training:
                self.update_copy_stat('copy_label_count', copy_mask.sum().item())

            copy_logits, gate = self.copy_net(encoder_outputs, decoder_output, source_mask)
            log_probs = compute_copy_extended_log_probs(logits, copy_logits, gate)
            loss = self._get_loss_by_probs(log_probs, ext_targets, target_mask)
        elif self.copy_mode == 'none':
            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
        else:
            raise NotImplementedError(f'copy_mode={self.copy_mode}')

        # TODO: We will be using beam search to get predictions for validation, but if beam size in 1
        # we could consider taking the last_predictions here and building step_predictions
        # and use that instead of running beam search again, if performance in validation is taking a hit
        output_dict = {"loss": loss}

        return output_dict

    def _prepare_output_projections(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, steps_count, decoder_output_dim)
        previous_steps_predictions = state.get("previous_steps_predictions")

        # shape: (batch_size, 1, target_embedding_dim)
        last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in `last_predictions`
            # shape: (group_size, 1, target_embedding_dim)
            previous_steps_predictions = last_predictions_embeddings
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat(
                [previous_steps_predictions, last_predictions_embeddings], 1
            )

        decoder_state, decoder_output = self._decoder_net(
            previous_state=state,
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
            previous_steps_predictions=previous_steps_predictions,
        )
        state["previous_steps_predictions"] = previous_steps_predictions

        # Update state with new decoder state, override previous state
        state.update(decoder_state)

        if self._decoder_net.decodes_parallel:
            decoder_output = decoder_output[:, -1, :]

        decoder_output = self.maybe_expand_decoder_output_with_fusion(decoder_output, state)
        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_output)

        return output_projections, state, decoder_output

    def _get_loss(
        self, logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of `targets` is expected to be greater than that of `logits` because the
        decoder does not need to compute the output corresponding to the last timestep of
        `targets`. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask, label_smoothing=self._label_smoothing_ratio
        )

    def _get_loss_by_probs(
        self, log_probs: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.BoolTensor
    ) -> torch.Tensor:
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return sequence_cross_entropy_with_log_probs(
            log_probs, relevant_targets, relevant_mask, label_smoothing=self._label_smoothing_ratio
        )

    def get_output_dim(self):
        return self.target_projecting_input_dim or self._decoder_net.get_output_dim()

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        # Parameters

        last_predictions : `torch.Tensor`
            A tensor of shape `(group_size,)`, which gives the indices of the predictions
            during the last time step.
        state : `Dict[str, torch.Tensor]`
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape `(group_size, *)`, where `*` can be any other number
            of dimensions.
        step : `int`
            The time step in beam search decoding.

        # Returns

        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of `(log_probabilities, updated_state)`, where `log_probabilities`
            is a tensor of shape `(group_size, num_classes)` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while `updated_state` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though `group_size` is not necessarily
            equal to `batch_size`, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state, decoder_step_output = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    def take_step_with_copy(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # shape: (group_size, num_classes)
        source_token_ids = state['sources']
        beam_size = self._beam_search.beam_size
        # todo: We need revert the extended predictions to in-vocab predictions. Note shape: [batch * beam_size]
        # Since reverting depending on batch ordering,
        # we need to make a 2D tensor shaped [batch, -1] to adapt it
        rev_last_predictions, _ = revert_extend_predictions_by_id(last_predictions.unsqueeze(-1), source_token_ids, self.target_vocab_size)
        last_predictions = rev_last_predictions.view(-1,)
        output_projections, state, decoder_step_output = self._prepare_output_projections(last_predictions, state)

        encoder_outputs, source_mask = state["encoder_outputs"], state["source_mask"]
        copy_logits, gate = self.copy_net(encoder_outputs, decoder_step_output.unsqueeze(1), source_mask)
        # shape: (group_size, num_classes + extend_size)
        class_log_probabilities = compute_copy_extended_log_probs(output_projections.unsqueeze(1), copy_logits, gate)
        class_log_probabilities = class_log_probabilities.squeeze(1)

        return class_log_probabilities, state

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(
                    self._tensor_based_metric.get_metric(reset=reset)  # type: ignore
                )
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        if self.copy_mode != 'none' and not self.training:
            all_metrics.update(self.copy_stat)
            if reset:
                self.reset_copy_stat()
        return all_metrics

    @overrides
    def forward(
        self,
        encoder_out: Dict[str, torch.LongTensor],
        target_tokens: TextFieldTensors = None,
        source_tokens: TextFieldTensors = None,
        meta_data: Optional[List[Dict]] = None,
    ) -> Dict[str, torch.Tensor]:
        state = encoder_out
        decoder_init_state = self._decoder_net.init_decoder_state(state)
        state.update(decoder_init_state)

        if target_tokens:
            state_forward_loss = (
                state if self.training else {k: v.clone() for k, v in state.items()}
            )
            output_dict = self._forward_loss(state_forward_loss, target_tokens, source_tokens)
        else:
            output_dict = {}

        if not self.training:
            source_token_ids = util.get_token_ids_from_text_field_tensors(source_tokens)
            state.update({
                'sources': source_token_ids
            })
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

            if target_tokens:
                targets = util.get_token_ids_from_text_field_tensors(target_tokens)
                if self._tensor_based_metric is not None:
                    # shape: (batch_size, beam_size, max_sequence_length)
                    top_k_predictions = output_dict["predictions"]
                    # shape: (batch_size, max_predicted_sequence_length)
                    best_predictions = top_k_predictions[:, 0, :]

                    if self.copy_mode == 'copy_by_id':
                        reverted_best_predictions, copy_count = revert_extend_predictions_by_id(best_predictions, source_token_ids, self.target_vocab_size)
                        best_predictions = reverted_best_predictions
                        self.update_copy_stat('copy_pred_count', copy_count)
                    elif self.copy_mode != 'none':
                        raise NotImplementedError(f'copy_mode={self.copy_mode}')

                    self._tensor_based_metric(best_predictions, targets)  # type: ignore

                if self._token_based_metric is not None:
                    output_dict = self.post_process(output_dict)
                    predicted_tokens = output_dict["predicted_tokens"]

                    self._token_based_metric(  # type: ignore
                        predicted_tokens,
                        self.indices_to_tokens(targets[:, 1:]),
                    )

        return output_dict

    @overrides
    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        if self.copy_mode != 'none':
            raise NotImplementedError(f"post_process method has not been adapted for copy_mode={self.copy_mode}")

        predicted_indices = output_dict["predictions"]
        all_predicted_tokens = self.indices_to_tokens(predicted_indices)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def indices_to_tokens(self, batch_indeces: numpy.ndarray) -> List[List[str]]:

        if not isinstance(batch_indeces, numpy.ndarray):
            batch_indeces = batch_indeces.detach().cpu().numpy()

        all_tokens = []
        for indices in batch_indeces:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            tokens = [
                self._vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            all_tokens.append(tokens)

        return all_tokens
