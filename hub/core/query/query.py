from typing import Callable, List
from tqdm import tqdm
from hub.core.dataset.dataset import Dataset
from hub.core.io import IOBlock, SampleStreaming
from hub.core.index import Index

import numpy

enable_cache = True

NP_ACCESS = Callable[[str], numpy.ndarray]


class DatasetQuery:
    def __init__(self, dataset, query: str):
        self._dataset = dataset
        self._query = query
        self._cquery = compile(query, "", "eval")
        self._tensors = [tensor for tensor in dataset.tensors.keys() if tensor in query]
        self._np_access: List[NP_ACCESS] = [
            _get_np(dataset, block) for block in expand(dataset, self._tensors)
        ]

    def execute(self, pg_callback=None, progressbar: bool=False, progressbar_bin: int = 1000):
        idx_map: List[int] = list()

        if progressbar: 
            bar = tqdm(total=len(self._dataset))
        
        for f in self._np_access:
            cache = {tensor: f(tensor) for tensor in self._tensors}
            for local_idx, idx in enumerate(f("index")):
                p = {tensor: cache[tensor][local_idx] for tensor in self._tensors}
                if eval(self._cquery, p):
                    idx_map.append(idx)
                    
                if progressbar:
                    bar.update()
                    
                if pg_callback and idx%progressbar_bin==0:
                    pg_callback(progressbar_bin)

        return idx_map


def _get_np(dataset: Dataset, block: IOBlock):
    idx = block.indices()

    def f(tensor):
        if tensor == "index":
            return numpy.array(idx)
        else:
            tensor_obj = dataset.tensors[tensor]
            tensor_obj.index = Index()
            return tensor_obj[idx].numpy(aslist=tensor_obj.is_dynamic)

    return f


def expand(dataset, tensor: List[str]) -> List[IOBlock]:
    return SampleStreaming(dataset, tensor).list_blocks()
