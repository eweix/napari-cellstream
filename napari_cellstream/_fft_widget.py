from magicgui import magicgui
import numpy as np
from napari.layers import Image
from napari import current_viewer

from cellstream.fft import generate_fft_features
from cellstream.image import downsample
import torch


from napari.qt.threading import thread_worker

@magicgui(
    call_button="Generate FFT features",
    blocks={"tooltip": "Enter 'auto' or a number of blocks"},
    downsample_by={"step": 0.000001}
   )
def fft_gui_widget(
    normalize_histogram=True,
    max_bin=128,
    use_gpu: bool = False,
    blocks: str = 'auto',
    downsample_by: float=1,

    return_amplitude: bool = True,
    return_norm_amp: bool = False,
    return_phase: bool = False,
    return_z_score: bool = True
):
    
    viewer = current_viewer()
    if viewer is None:
        raise RuntimeError("No active napari viewer found")

    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image):
        raise RuntimeError("No active image layer selected")

    fft_features_to_process=[]
    if return_amplitude==True:
        fft_features_to_process.append('full_amplitude')
    if return_norm_amp==True:
        fft_features_to_process.append('normalized_amplitude')
    if return_phase==True:
        fft_features_to_process.append('phase')
    if return_z_score==True:
        fft_features_to_process.append('z_score')
        
        
    if use_gpu==True:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # For Macs with M1/M2 GPUs
        else:
            device = torch.device("cpu")
            print("No GPU detected; switching to CPU mode...")
    else:
        device = torch.device("cpu")
        
    fft_gui_widget._abort_flag = False

    print(f"Performing blocked FFT processing using device: {device}")
        
    image_data = layer.data
    
    if isinstance(image_data, np.ndarray):
        image_data = torch.from_numpy(image_data.astype('float32'))
    
    if downsample_by < 1:
        print(f"Downsampling image by {downsample_by}...")
        image_data=downsample(image_data,downsample_by)
        
    blocks_val = 'auto' if str(blocks).strip(" '\"").lower() == 'auto' else int(str(blocks).strip(" '\""))

    if blocks_val != 'auto':
        num_pixels=image_data.shape[-2] * image_data.shape[-1]
        batch_size=int(num_pixels/blocks_val)
    else:
        batch_size = 'auto'

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
            if getattr(fft_gui_widget, '_abort_flag', False):
                raise RuntimeError("Job cancelled by user.")
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
    def _fft_worker():
        import cellstream.fft.utils as fft_utils
        original_tqdm = getattr(fft_utils, 'tqdm', None)
        fft_utils.tqdm = MockTqdm
        try:
            feature_dict = generate_fft_features(
                image_data,
                normalize_histogram=normalize_histogram,
                max_bin=max_bin,
                batch_size=batch_size,
                fft_features_to_process=fft_features_to_process,
                device=device
            )
            return feature_dict
        finally:
            if original_tqdm is not None:
                fft_utils.tqdm = original_tqdm
            if getattr(fft_gui_widget, '_abort_flag', False):
                if use_gpu:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

    worker = _fft_worker()

    def set_total(val):
        pbar.total = val
        pbar.set_description("Processing batches")

    def do_cancel():
        fft_gui_widget._abort_flag = True
        cancel_fft_button.text = "Cancelling..."
        cancel_fft_button.enabled = False
        worker.quit()

    cancel_fft_button.text = "Cancel Job"
    cancel_fft_button.enabled = True
    cancel_fft_button.show()
    cancel_fft_button.changed.connect(do_cancel)

    def cleanup():
        pbar.close()
        cancel_fft_button.hide()
        try:
            cancel_fft_button.changed.disconnect(do_cancel)
        except Exception:
            pass

    pbar.set_description("Pre-allocating arrays...")
    emitter.total_signal.connect(set_total)
    emitter.update_signal.connect(pbar.update)
    worker.finished.connect(cleanup)
    return worker

from magicgui.widgets import PushButton
cancel_fft_button = PushButton(text="Cancel Job")
cancel_fft_button.hide()
fft_gui_widget.append(cancel_fft_button)
