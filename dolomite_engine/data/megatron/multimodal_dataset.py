import numpy as np
import torch
from torch.utils.data import Dataset


class MultimodalInstructionDataset(Dataset):
    def __init__(self, name: str, sample_indices: np.ndarray,
                 indexed_datasets: dict[str, Dataset], seq_length: int):

        self.indexed_text = indexed_datasets["text"]
        self.indexed_role = indexed_datasets["role"]
        self.indexed_vision_patch_indices = indexed_datasets["vision_patch_indices"]
        self.indexed_vison_patch = indexed_datasets["vision_patch"]

        # validate indices
        assert np.min(sample_indices) >= 0
        assert np.max(sample_indices) < len(self.indexed_text)
        assert len(self.indexed_text) == len(self.indexed_role)

        self.name = name
        self.sample_indices = sample_indices
        self.seq_length = seq_length

    def __len__(self) -> int:
        return self.sample_indices.shape[0]

    def __getitem__(self, idx) -> dict:
        # Get the shuffled index.
        idx = self.sample_indices[idx]
        text = self.indexed_text.get(idx)
        role = self.indexed_role.get(idx)
        vision_patch_indices = self.indexed_vision_patch_indices.get(idx)
        vision_patch = self.indexed_vison_patch.get(idx)
        assert text is not None and role is not None and text.shape == role.shape
        assert vision_patch_indices is not None and vision_patch is not None
        assert vision_patch_indices.shape == text.shape
        return {
            "text": text.astype(np.int64),
            "role": role.astype(np.int64),
            "vision_patch_indices": vision_patch_indices.astype(np.int64),
            "vision_patch": vision_patch.astype(np.float32)
        }

