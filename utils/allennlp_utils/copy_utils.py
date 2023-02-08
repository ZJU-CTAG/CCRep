from typing import List

import torch
from allennlp.nn import util

from allennlp_models.generation.models.copynet_seq2seq import CopyNetSeq2Seq


def generate_extended_target_label_by_id(source_token_ids: torch.LongTensor,
                                         target_token_ids: torch.LongTensor,
                                         vocab_size: int,
                                         excluded_indices: List[int] = [0]):
    """
    Migrated from "allennlp_models.generation.models.copynetseq2seq"
    -------------------------------------------------------------------------

    Modify the gold target tokens relative to the extended vocabulary.

    For gold targets that are OOV but were copied from the source, the OOV index
    will be changed to the index of the first occurence in the source sentence,
    offset by the size of the target vocabulary.

    # Parameters

    target_tokens : `torch.Tensor`
        Shape: `(batch_size, target_sequence_length)`.
    source_token_ids : `torch.Tensor`
        Shape: `(batch_size, source_sequence_length)`.
    target_token_ids : `torch.Tensor`
        Shape: `(batch_size, target_sequence_length)`.

    # Returns

    torch.Tensor
        Modified `target_tokens` with OOV indices replaced by offset index
        of first match in source sentence.
    """
    batch_size, target_sequence_length = target_token_ids.size()
    source_sequence_length = source_token_ids.size(1)

    pad_mask = torch.ones_like(target_token_ids, device=target_token_ids.device).bool()
    for ex_index in excluded_indices:
        pad_mask &= target_token_ids != ex_index

    # shape: (batch_size, target_sequence_length, source_sequence_length)
    expanded_source_token_ids = source_token_ids.unsqueeze(1).expand(
        batch_size, target_sequence_length, source_sequence_length
    )
    # shape: (batch_size, target_sequence_length, source_sequence_length)
    expanded_target_token_ids = target_token_ids.unsqueeze(-1).expand(
        batch_size, target_sequence_length, source_sequence_length
    )
    # shape: (batch_size, target_sequence_length, source_sequence_length)
    matches = expanded_source_token_ids == expanded_target_token_ids
    # shape: (batch_size, target_sequence_length)
    copied = matches.sum(-1) > 0
    # shape: (batch_size, target_sequence_length)
    mask = pad_mask & copied
    # shape: (batch_size, target_sequence_length)
    first_match = ((matches.cumsum(-1) == 1) & matches).to(torch.uint8).argmax(-1)
    # shape: (batch_size, target_sequence_length)
    new_target_tokens = (
            target_token_ids * ~mask +                  # Indices no need to copy
            (first_match.long() + vocab_size) * mask    # Indices to copy
    )
    return new_target_tokens, mask

def compute_copy_extended_log_probs(gen_logits, copy_logits, gate):
    gen_probs = torch.softmax(gen_logits, dim=-1)
    copy_probs = torch.softmax(copy_logits, dim=-1)
    # Gate-weighted probs, not logits.
    probs = torch.cat((gate[:, :, 0].unsqueeze(-1) * gen_probs,
                       gate[:, :, 1].unsqueeze(-1) * copy_probs), dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-10, max=1))
    return log_probs

def revert_extend_predictions_by_id(predictions: torch.LongTensor,
                                    sources: torch.LongTensor,
                                    target_vocab_size: int):
    extend_mask = predictions >= target_vocab_size
    extend_indices = torch.nonzero(extend_mask)
    # Compute shift for copy items within each source sequence
    source_ref_indices_dim1 = predictions[extend_indices[:,0], extend_indices[:,1]] - target_vocab_size
    source_ref_indices_full = torch.clone(extend_indices)
    # Keeping the sequence indices, we only change the shifts in sequence for each copy item
    source_ref_indices_full[:,1] = source_ref_indices_dim1
    # Fill the positions of extended predictions with corresponding sources
    predictions[extend_indices[:,0], extend_indices[:,1]] = sources[source_ref_indices_full[:,0], source_ref_indices_full[:,1]]
    return predictions, len(extend_indices)

if __name__ == '__main__':
    src = torch.LongTensor([[1,2,3,4,5,0],[1,3,5,0,0,0]])
    tgt = torch.LongTensor([[1,5,6,0],[1,3,0,0]])
    ext_tgt = generate_extended_target_label_by_id(src, tgt, 10, [0])
    rev_tgt, copy_count = revert_extend_predictions_by_id(ext_tgt[0], src, 10)
