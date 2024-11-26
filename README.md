# BigVGAN Inference
An unofficial minimal package for using BigVGAN at inference time

[![PyPI version](https://img.shields.io/pypi/v/bigvganinference)](https://pypi.org/project/bigvganinference/)
![License](https://img.shields.io/pypi/l/bigvganinference)
![Python versions](https://img.shields.io/pypi/pyversions/bigvganinference)

## Installation

```bash
pip install bigvganinference
```

or install from source:

```bash
git clone https://github.com/thunn/BigVGANInference.git
cd BigVGANInference
poetry install
```

## Usage

Loading model is as simple as:
```python
from bigvganinference import BigVGANInference

# -- model loading ---

# model is loaded, set to eval and weight norm is removed
model = BigVGANInference.from_pretrained(
    'nvidia/BigVGAN-V2-44KHZ-128BAND-512X', use_cuda_kernel=False
)

# also supports loading from local directory
model = BigVGANInference.from_pretrained(
    "path/to/local/model", use_cuda_kernel=False
)

# --- usage example ---

path_to_audio = "path/to/audio.wav"
wav, sr = librosa.load(path_to_audio, sr=model.h.sampling_rate, mono=True)

# get mel spectrogram using bigvgan's implementation
# mel: [B(1), MEL_BANDS, T_time]
mel = model.get_mel_spectrogram(wav)

# generate waveform from mel
# note: torch.inference_mode() is used internally
# output_audio: [B(1), 1, T_time]
output_audio = model(input_mel)

# get numpy array
output_audio_np = output_audio.squeeze(0).cpu().numpy()
```

See the [example](https://github.com/thunn/BigVGANInference/blob/main/example/inference.py) for full usage example.

## Acknowledgements
This is an unofficial implementation based on [original BigVGAN repository](https://github.com/NVIDIA/BigVGAN).

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/thunn/BigVGANInference/blob/main/LICENSE) file for details.
