# BigVGANInference
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
from bigvganinference import BigVGANInference, BigVGANHFModel

# model is loaded, set to eval and weight norm is removed
model = BigVGANInference.from_pretrained(
    BigVGANHFModel.V2_44KHZ_128BAND_512X, use_cuda_kernel=False
)


output_audio = model(input_mel)
```

See the [example](https://github.com/thunn/BigVGANInference/blob/main/example/inference.py) for full usage example.

## Acknowledgements
This is an unofficial implementation based on [original BigVGAN repository](https://github.com/NVIDIA/BigVGAN).

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/thunn/BigVGANInference/blob/main/LICENSE) file for details.
