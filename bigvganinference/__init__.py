import importlib.metadata

from bigvganinference.inference import BigVGANHFModel, BigVGANInference

__version__ = importlib.metadata.version("bigvganinference")

__all__ = ["BigVGANInference", "BigVGANHFModel"]
