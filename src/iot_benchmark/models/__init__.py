from .uot_iot import UOTIOTBenchmarkModel

__all__ = ["GHIOTModel", "PersistIOTModel", "UOTIOTBenchmarkModel", "UOTIOTModel"]


def __getattr__(name: str):
    if name == "GHIOTModel":
        from .gh_iot import GHIOTModel
        return GHIOTModel
    if name == "PersistIOTModel":
        from .persist_iot import PersistIOTModel
        return PersistIOTModel
    if name == "UOTIOTModel":
        from .uot_iot_canonical import UOTIOTModel
        return UOTIOTModel
    raise AttributeError(name)
