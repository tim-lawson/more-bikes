"""Model-loader classes."""

# pylint:disable=too-few-public-methods

from abc import ABC


class ModelLoader(ABC):
    """Abstract model-loader class."""

    def __init__(self, station_id: int):
        self.station_id = station_id


class ModelLoaderFull(ModelLoader):
    """`full` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)
        raise NotImplementedError


class ModelLoaderFullTemp(ModelLoader):
    """`full_temp` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)
        raise NotImplementedError


class ModelLoaderShortFull(ModelLoader):
    """`short_full` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)
        raise NotImplementedError


class ModelLoaderShort(ModelLoader):
    """`short` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)
        raise NotImplementedError


class ModelLoaderShortFullTemp(ModelLoader):
    """`short_full_temp` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)
        raise NotImplementedError


class ModelLoaderShortTemp(ModelLoader):
    """`short_temp` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)
        raise NotImplementedError
