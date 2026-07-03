from magicgui import magicgui
from napari.types import ImageData
from napari.layers import Image
from napari import current_viewer
from qtpy.QtWidgets import QWidget
import torch
import numpy as np

from cellstream.cwt.utils import generate_cwt_image_cellstreams

from napari.qt.threading import thread_worker

@magicgui(
    call_button="Generate CWT Features",
    blocks={"tooltip": "Enter 'auto' or a number of blocks"},
    wavelet_choice={"visible": False},
    wavelet_parameters={"visible": False},
    nv={"visible": False},
)
def generate_cwt_features_widget(
    min_scale: int = 80,
    max_scale: int = 180,
    num_filter_banks: int = 1,
    carrier_channel: int = 0,
    blocks: str = '50',
    normalize_amplitudes: bool = False,
    use_gpu: bool = False,
    bank_method: str = 'max_pool',
    downsample_by: float = 1.0,
    normalize_histogram: bool = True,
    mean_center: bool = False,
    return_amplitude: bool = True,
    return_scales: bool = True,
    return_phase: bool = False,
    return_z_score: bool = True,
    wavelet_choice: str = 'gmw',
    wavelet_parameters= None,
    nv: int = 32,
):
    
    viewer = current_viewer()
    if viewer is None:
        raise RuntimeError("No active napari viewer found")

    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image):
        raise RuntimeError("No active image layer selected")

    img = layer.data

    if img.ndim != 4:
        print("Expected image shape (T, C, X, Y)")
        return
    T, C, X, Y=img.shape

    # Convert numpy to torch tensor
    img_tensor = torch.from_numpy(img.astype('float32'))

    #prepare channel_outputs parameter
    channel_outputs=dict()
    for c in range(C):
        channel_returns=list()
        if return_amplitude==True:
            channel_returns.append('amp')
        if return_scales==True:
            channel_returns.append('freq')
        if return_phase==True:
            channel_returns.append('phase')
        if return_z_score==True:
            channel_returns.append('z_score')
        channel_outputs[c]=channel_returns
    
    #prepare wavelet parameters:
    if wavelet_parameters==None:
        wavelet=wavelet_choice #pure string
    else:
        wavelet=(wavelet_choice,wavelet_parameters) # tuple

    blocks_val = 'auto' if str(blocks).strip(" '\"").lower() == 'auto' else int(str(blocks).strip(" '\""))

    from napari.utils import progress
    from qtpy.QtCore import QObject, Signal
    pbar = progress(total=0)

    class ProgressEmitter(QObject):
        total_signal = Signal(int)
        update_signal = Signal(int)
        
    emitter = ProgressEmitter()

    class MockTqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            if iterable is not None:
                try:
                    self.total = len(iterable)
                except TypeError:
                    self.total = 0
                self.iterable = iterable
            else:
                self.total = total or 0
                self.iterable = None
            
            self.n = 0
            emitter.total_signal.emit(self.total)
            
        def update(self, n=1):
            emitter.update_signal.emit(n)
            
        def __iter__(self):
            if self.iterable is not None:
                for x in self.iterable:
                    yield x
                    self.update(1)
        
        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    @thread_worker
    def _cwt_worker():
        import cellstream.cwt.utils as cwt_utils
        original_tqdm = getattr(cwt_utils, 'tqdm', None)
        cwt_utils.tqdm = MockTqdm
        print(f"Running CWT blockwise feature generation with {wavelet} and {nv}...")
        try:
            results = generate_cwt_image_cellstreams(
                img=img_tensor,
                min_scale=min_scale,
                max_scale=max_scale,
                num_filter_banks=num_filter_banks,
                normalize_amplitudes=normalize_amplitudes,
                blocks=blocks_val,
                use_gpu=use_gpu,
                bank_method=bank_method,
                downsample_by=downsample_by,
                normalize_histogram=normalize_histogram,
                mean_center=mean_center,
                carrier_channel=carrier_channel,
                channel_outputs=channel_outputs,
                wavelet=wavelet,
                nv=nv
            )
            return results
        finally:
            if original_tqdm is not None:
                cwt_utils.tqdm = original_tqdm

    worker = _cwt_worker()

    def set_total(val):
        pbar.total = val

    emitter.total_signal.connect(set_total)
    emitter.update_signal.connect(pbar.update)
    worker.finished.connect(pbar.close)
    return worker