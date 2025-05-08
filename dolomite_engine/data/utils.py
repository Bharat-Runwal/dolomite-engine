import math
from typing import Iterable

import torch
from enum import IntEnum


from ..enums import LossMask, Mode
from ..hf_models import convert_padding_free_lists_to_tensors
import numpy as np

class Role(IntEnum):
    system = 0
    user = 1
    assistant = 2
    image = 3
    PACK_SEP = 1000  # This is used to separate two conversations packed together in to one sample

def collate_fn(
    batch: list[dict],
    mode: Mode,
    loss_mask: LossMask,
    eos_token_id: int,
    is_encoder_decoder: bool,
    use_padding_free_transformer: bool,
    labels_mask_value: int = -100,
    pad_to_multiple_of: int = 1,
    device: torch.device = None,
) -> dict:
    """prepares the batch with padding to pass into the forward function of the HuggingFace model

    Args:
        batch (list[dict]): input tokens and output tokens. Output tokens are optional when running generation but required for training.

    Returns:
        dict: dict containing input_ids, attention_mask and labels if outputs is specified
    """

    inputs = [i["input"] for i in batch]
    outputs = [i["output"] for i in batch] if mode == Mode.training else None

    # labels is None when outputs is None
    labels = None

    device = torch.cuda.current_device() if device is None else device

    if use_padding_free_transformer:
        if is_encoder_decoder:
            raise NotImplementedError("padding free transformer only supports decoder only models")
        else:
            input_ids = inputs
            attention_mask = None

            if loss_mask == LossMask.output_only:
                labels = [
                    [labels_mask_value] * (len(array_in) - len(array_out)) + array_out
                    for array_in, array_out in zip(inputs, outputs)
                ]
            elif loss_mask == LossMask.no_mask:
                labels = inputs
            else:
                raise ValueError(f"unexpected loss_mask ({loss_mask})")

            tokens_to_add = 0
            if pad_to_multiple_of > 1:
                total_tokens = sum([len(array) for array in input_ids])
                tokens_to_add = (math.ceil(total_tokens / pad_to_multiple_of) * pad_to_multiple_of) - total_tokens

            # we pad the last example in the batch on the right
            # NOTE this can be done since the attention is causal
            input_ids[-1].extend([eos_token_id] * tokens_to_add)
            labels[-1].extend([labels_mask_value] * tokens_to_add)

            input_ids, position_ids, _, labels, cu_seqlens, max_seqlen = convert_padding_free_lists_to_tensors(
                input_ids=input_ids, labels=labels, device=device
            )

        result = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }
        if labels is not None:
            result["labels"] = labels
    else:
        if is_encoder_decoder:
            if pad_to_multiple_of > 1:
                raise NotImplementedError("pad_to_multiple_of is not implemented for encoder-decoder models")

            input_max_length = max(list(map(len, inputs)))

            input_ids = [[eos_token_id] * (input_max_length - len(array)) + array for array in inputs]
            attention_mask = [[0] * (input_max_length - len(array)) + [1] * len(array) for array in inputs]

            if outputs is not None:
                assert (
                    loss_mask == LossMask.output_only
                ), "only output_only loss mask is supported with encoder decoder models"

                output_max_length = max(list(map(len, outputs)))
                # right padding for labels
                labels = [array + [labels_mask_value] * (output_max_length - len(array)) for array in outputs]
        else:
            max_length = max(list(map(len, inputs)))
            if pad_to_multiple_of > 1:
                max_length = math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of

            input_ids = [[eos_token_id] * (max_length - len(array)) + array for array in inputs]
            attention_mask = [[0] * (max_length - len(array)) + [1] * len(array) for array in inputs]

            if outputs is not None:
                if loss_mask == LossMask.output_only:
                    labels = [[labels_mask_value] * (max_length - len(array)) + array for array in outputs]
                elif loss_mask == LossMask.no_mask:
                    labels = inputs
                else:
                    raise ValueError(f"unexpected loss_mask ({loss_mask})")

        result = {
            "input_ids": torch.tensor(input_ids, device=device),
            "attention_mask": torch.tensor(attention_mask, device=device),
        }
        if labels is not None:
            result["labels"] = torch.tensor(labels, device=device)

    return result

def round_to_multiple_of(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y



def multimodal_collator(
    args,
    tokenizer,
    data,
    scalar_loss_mask=0.0,
    return_attention_mask_in_length: bool = False,
    loss_role: str = "assistant",
    no_loss_beyond_token_id: int = None,
    no_loss_on_token_ids: list = [],
    vision_patch_size: int = 32,
):
    assert loss_role in ["assistant", "user", "all"]
    pad_id = tokenizer.pad
    seq_len = args.seq_length

    if args.variable_seq_lengths:
        max_sample_length = max(len(x["text"]) for x in data)
        seq_len = min(args.seq_length, round_to_multiple_of(max_sample_length, 16))
    seq_len += 1  # +1 to get seq_len tokens after shifting (token[t+1] is label for token[t])

    # pad data to seq_len, create attention mask
    batch_size = len(data)
    # INPUTS
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    role = torch.full_like(attention_mask, -1)
    input = torch.full_like(attention_mask, pad_id)
    vision_patch_indices = torch.full_like(attention_mask, -1)
    vision_patches = [] # list of vision patches (this is dynamic, so we can't use torch.full_like)

    # For loss and example segmentation
    # 1 means optimize loss, 0 means no loss
    loss_mask = torch.full_like(attention_mask, scalar_loss_mask, dtype=torch.float)
    # example id for each token, used for packed sequences
    example_ids = torch.zeros_like(attention_mask)

    attention_mask_in_length = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for i, x in enumerate(data):
        t = x["text"]
        r = x["role"]
        # print(f"batch {i} text shape", t.shape)
        # text shape (32723,)
        cur_vision_patch_indices = x["vision_patch_indices"]
        # vision_patch shape (59006976,)
        cur_vision_patch = x["vision_patch"].reshape(
            -1,
            vision_patch_size * vision_patch_size * 3
        )
        # print("cur_vision_patch_indices shape", cur_vision_patch_indices.shape)

        l = len(t)

        # Increment cur_vision_patch_indices by the number of vision patches already seen
        # since we are appending vision patches to a list
        cur_vision_patch_indices += len(vision_patches)
        
        # print("seq_len", seq_len)
        # print("token len", l)
        if l < seq_len:
            attention_mask[i, l:] = 0
            input[i, :l] = torch.from_numpy(t)
            role[i, :l] = torch.from_numpy(r)
            vision_patch_indices[i, :l] = torch.from_numpy(cur_vision_patch_indices)
        else:
            input[i] = torch.from_numpy(t[:seq_len])
            role[i] = torch.from_numpy(r[:seq_len])

            vision_patch_indices[i] = torch.from_numpy(cur_vision_patch_indices[:seq_len])
            # # find value in cur_vision_patch_indices[:seq_len] that are not -1
            # # and use that to index into cur_vision_patch
            # indices_non_0 = torch.where(cur_vision_patch_indices != -1)
            # patch_indices_kept = cur_vision_patch_indices[indices_non_0]
            # vision_patches.extend(cur_vision_patch[indices])
            
        # note: we just append everything for simplicity
        # since the dataset are pre-packed, so it less likely to waste memory
        vision_patches.extend(cur_vision_patch)

        # Segmentation for packed sequences
        current_example_id = 0
        cur_count = 0
        for j in range(min(l, seq_len)):
            # Switch to a new example if we encounter a PACK_SEP token
            if role[i, j] == Role.PACK_SEP.value:
                # Add the count of tokens in the previous example
                attention_mask_in_length[i, current_example_id] = cur_count
                # Switch to the next example
                current_example_id += 1
                cur_count = 0 # reset

            example_ids[i, j] = current_example_id
            cur_count += 1

        # Check if j is the last token in the sequence
        # If so, subtract 1 from the current example's count
        if j == seq_len - 1:
            attention_mask_in_length[i, current_example_id] = cur_count - 1
        # Handle the case where the last token is not a PACK_SEP token
        else:
            attention_mask_in_length[i, current_example_id] = cur_count

    # Loss mask
    # - only calculate loss for loss role
    if loss_role == "all":
        loss_mask = torch.ones_like(attention_mask, dtype=torch.float)
    else:
        loss_role = Role[loss_role].value
        loss_mask[role == loss_role] = 1.0
    
    if no_loss_beyond_token_id:
        no_loss_beyond_token_id = int(no_loss_beyond_token_id)
        loss_mask[input >= no_loss_beyond_token_id] = 0.0
    if no_loss_on_token_ids:
        for token_id in no_loss_on_token_ids:
            loss_mask[input == token_id] = 0.0

    # - completely ignore padding tokens
    loss_mask[input == pad_id] = 0.0

    # -- Previous handled by get_batch
    # Shift input to the right by one
    tokens = input[:, :-1].contiguous()
    attention_mask = attention_mask[:, :-1]
    assert torch.all(attention_mask_in_length[:, -1] == 0)
    attention_mask_in_length = attention_mask_in_length[:, :-1]
    example_ids = example_ids[:, :-1]
    attention_mask, position_ids = get_attention_mask_and_position_ids(
        tokens, attention_mask, example_ids
    )
    # convert to torch.int64
    attention_mask = attention_mask.to(torch.int64)

    # labels = input[:, 1:].contiguous() therefore we need to shift the loss_mask similarly
    loss_mask = loss_mask[:, 1:].contiguous()
    vision_patch_indices = vision_patch_indices[:, :-1]
    # aggregate vision patches
    vision_patches = torch.tensor(np.array(vision_patches), dtype=torch.float32)
    vision_patches = vision_patches.view(-1, vision_patch_size * vision_patch_size * 3)

    # print("vision_patches shape", vision_patches.shape)
    # print("vision_patch_indices shape", vision_patch_indices.shape)
    # print("attention_mask shape", attention_mask.shape)
    # print("position_ids shape", position_ids.shape)
    # print("vision_patches", vision_patches)
    # print("vision_patch_indices", vision_patch_indices)
    
    return {
        "text": input,
        "attention_mask": attention_mask if not return_attention_mask_in_length \
            else attention_mask_in_length,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "vision_patch_indices": vision_patch_indices,
        "vision_patches": vision_patches,
    }


# Heavily inspired by Andreas Köpf: https://github.com/andreaskoepf/epfl-megatron/tree/local_changes/
def get_attention_mask_and_position_ids(data, attention_mask, example_ids):
    """
    Constructs causal attention masks and position IDs for sequences, based on provided example IDs.

    The function creates a causal attention mask to ensure each token in a sequence only attends 
    to previous tokens and itself. When sequences are packed, the attention mask also ensures 
    that tokens from one sequence do not attend to tokens from a subsequent packed sequence. 

    Additionally, position IDs are generated such that they reset for each new example in the packed sequences.

    Args:
    - data (torch.Tensor): Input data tensor of shape (batch_size, seq_length).
    - attention_mask (torch.Tensor): Initial attention mask of shape (batch_size, seq_length) where
                                     values close to 1 indicate tokens and values close to 0 indicate padding.
    - example_ids (torch.Tensor): Tensor of shape (batch_size, seq_length) indicating the IDs of packed examples.

    Returns:
    - attention_mask (torch.Tensor): Updated binary attention mask of shape (batch_size, 1, seq_length, seq_length).
    - position_ids (torch.Tensor): Position IDs tensor of shape (batch_size, seq_length) where IDs reset for each 
                                   new example in the packed sequences.
    """

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Expand example_ids for comparison
    expanded_example_ids = example_ids.unsqueeze(2).expand(micro_batch_size, seq_length, seq_length)
    
    # Create a comparison mask where each position is compared to every other position in the sequence
    comparison_mask = (expanded_example_ids == expanded_example_ids.transpose(1, 2)).float()

    # Attention mask based on example_ids
    causal_mask = torch.tril(comparison_mask).float()
    
    # Merge the two masks
    merged_mask = attention_mask.unsqueeze(2) * causal_mask

    # Convert attention mask to binary, True entries will masked
    attention_mask = (merged_mask < 0.5).to(data.device)

    # Position ids. reset for each new example
    position_ids = torch.zeros_like(data, dtype=torch.long)
    for i in range(micro_batch_size):
        pos = 0
        for j in range(seq_length):
            position_ids[i, j] = pos
            pos += 1
            # Check if this token is the last one in an example
            if j < seq_length - 1 and example_ids[i, j] != example_ids[i, j+1]:
                pos = 0  # reset
    position_ids.to(data.device)

    return attention_mask, position_ids



def custom_iterator(x: Iterable | None, infinite: bool) -> Iterable:
    """converts and iterable into a non-ending infinite iterable, will return None if input is None

    Args:
        x (Iterable): the iterable to convert
        infinite (bool): whether to return an infinite iterator

    Returns:
        Iterable: the converted iterable

    Yields:
        Iterator[Iterable]: an element from the original iterator
    """

    if x is None:
        return None

    def infinite_iterator(q):
        while True:
            for i in q:
                yield i

    iterator_function = infinite_iterator if infinite else iter
    return iterator_function(x)


def get_next_batch(x: Iterable | None) -> dict:
    """get next batch

    Args:
        x (Iterable): dataloader

    Returns:
        dict: batch
    """

    # train_dataloader is always None on TP ranks other than 0
    if x is None:
        return None

    return next(x)
