"""MSL dataset loader -- shares format with SMAP."""

from typing import List

from vector.data.loaders.smap import _load_smap_msl
from vector.data.registry import BaseLoader, SequenceData, register


@register("MSL")
class MSLLoader(BaseLoader):
    """Loader for NASA MSL spacecraft dataset (27 sequences, 55 dims)."""

    def load(self, data_dir: str) -> List[SequenceData]:
        return _load_smap_msl(data_dir, "MSL")
