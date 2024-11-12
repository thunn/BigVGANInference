import numpy as np
import torch
from bigvganinference.env import AttrDict
from enum import Enum
from bigvganinference.bigvgan import BigVGAN

from typing import Dict, Optional, Union
from bigvganinference.meldataset import get_mel_spectrogram


class BigVGANHFModel(str, Enum):
    """
    BigVGAN HF models.
    """

    V2_44KHZ_128BAND_512X = "nvidia/bigvgan_v2_44khz_128band_512x"
    V2_44KHZ_128BAND_256X = "nvidia/bigvgan_v2_44khz_128band_256x"
    V2_24KHZ_100BAND_256X = "nvidia/bigvgan_v2_24khz_100band_256x"
    V2_22KHZ_80BAND_256X = "nvidia/bigvgan_v2_22khz_80band_256x"
    V2_22KHZ_80BAND_FMAX8K_256X = "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x"
    V2_24KHZ_100BAND = "nvidia/bigvgan_24khz_100band"
    V2_22KHZ_80BAND = "nvidia/bigvgan_22khz_80band"
    BASE_24KHZ_100BAND = "nvidia/bigvgan_base_24khz_100band"
    BASE_22KHZ_80BAND = "nvidia/bigvgan_base_22khz_80band"

    def __str__(self):
        return self.value


class BigVGANInference(BigVGAN):
    """
    BigVGAN inference.
    """

    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__(h, use_cuda_kernel)

        # set to eval and remove weight norm
        self.eval()
        self.remove_weight_norm()

    def get_mel_spectrogram(self, wav: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Wrapper function to preprocess audio and convert to mel spectrogram.

        Args:
            wav (torch.Tensor | np.ndarray): Audio waveform.

        Returns:
            torch.Tensor: Mel spectrogram.
        """

        # ensure wav is FloatTensor with shape [B(1), T_time]
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)

        # If batch dimension is missing, add it
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        # ensure that audio is mono (batch size of 1)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0).unsqueeze(0)

        mel = get_mel_spectrogram(wav, self.h)

        # ensure mel is on the same device as the model
        device = next(self.parameters()).device
        mel = mel.to(device)

        return mel
