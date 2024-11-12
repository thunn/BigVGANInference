# BigVGANInference
An unofficial minimal package for using BigVGAN at inference time


## Installation

```python
pip install bigvganinference
```

or install from source:

```python
git clone https://github.com/thunn/BigVGANInference.git
cd BigVGANInference
poetry install
```

## Usage

Loading model is as simple as:
```python
from bigvganinference.inference import BigVGANInference, BigVGANHFModel

model = BigVGANInference.from_pretrained(
    BigVGANHFModel.V2_44KHZ_128BAND_512X, use_cuda_kernel=False
)
```

See the [example](example/inference.py) for full usage example.

## Acknowledgements
This is an unofficial implementation based on [original BigVGAN repository](https://github.com/NVIDIA/BigVGAN).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.