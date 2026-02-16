from .encoder import CategoryEmbedding

__all__ = ["CategoryEmbedding"]

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"  # During development before install