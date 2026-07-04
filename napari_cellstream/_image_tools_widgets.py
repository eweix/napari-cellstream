from magicgui import magicgui
from magicgui.widgets import PushButton
from napari.layers import Image
from napari import current_viewer
from qtpy.QtWidgets import QWidget
import torch
import numpy as np
from napari.qt.threading import thread_worker

from cellstream.image import downsample
from cellstream.hilbert import hilbert_transform
from cellstream.filters import create_ir_filter, apply_fir_filter
from cellstream.viz import color_by_axis

from napari.utils import progress
from qtpy.QtCore import QObject, Signal

class ProgressEmitter(QObject):
    total_signal = Signal(int)
    update_signal = Signal(int)

def _setup_progress(widget_obj, description="Processing..."):
    pbar = progress(total=0)
    pbar.set_description(description)
    emitter = ProgressEmitter()
    
    def set_total(val):
        pbar.total = val

    emitter.total_signal.connect(set_total)
    emitter.update_signal.connect(pbar.update)
    
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
            if getattr(widget_obj, '_abort_flag', False):
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

    return pbar, emitter, MockTqdm

def _add_cancel_button(widget_obj, worker, pbar):
    cancel_btn = PushButton(text="Cancel Job")
    cancel_btn.show()
    widget_obj.append(cancel_btn)
    
    def do_cancel():
        widget_obj._abort_flag = True
        cancel_btn.text = "Cancelling..."
        cancel_btn.enabled = False
        worker.quit()
        
    cancel_btn.changed.connect(do_cancel)
    
    def cleanup():
        pbar.close()
        cancel_btn.hide()
        try:
            widget_obj.remove(cancel_btn)
            cancel_btn.changed.disconnect(do_cancel)
        except Exception:
            pass

    worker.finished.connect(cleanup)


# 1. Downsample Tool
@magicgui(call_button="Downsample active image")
def downsample_gui_widget(
    downsample_by: float = 1,
    is_mask: bool = False,
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    img = torch.from_numpy(layer.data.astype('float32'))
    downsample_gui_widget._abort_flag = False

    pbar, emitter, _ = _setup_progress(downsample_gui_widget, "Downsampling...")

    @thread_worker
    def _downsample_worker():
        try:
            emitter.total_signal.emit(1)
            ds = downsample(tensor=img, scale=downsample_by, is_mask=is_mask)
            if getattr(downsample_gui_widget, '_abort_flag', False):
                raise RuntimeError("Cancelled")
            emitter.update_signal.emit(1)
            return ds.detach().cpu().numpy()
        finally:
            if getattr(downsample_gui_widget, '_abort_flag', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker = _downsample_worker()
    _add_cancel_button(downsample_gui_widget, worker, pbar)
    return worker


# 2. False Color Tool
@magicgui(call_button="False-color spectrum")
def false_color_widget(
    min_slice: int = 0,
    max_slice: int = 40,
    colormap: str = 'turbo'
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    img = torch.from_numpy(layer.data.astype('float32'))
    img_ndim = img.dim()
    if img_ndim == 4:
        img = img[min_slice:max_slice]
    elif img_ndim == 5:
        img = img[:,:,min_slice:max_slice,:,:]

    false_color_widget._abort_flag = False
    pbar, emitter, MockTqdm = _setup_progress(false_color_widget, "Generating colors...")

    @thread_worker
    def _false_color_worker():
        import cellstream.viz as viz
        original_tqdm = getattr(viz, 'tqdm', None)
        viz.tqdm = MockTqdm
        try:
            cc = color_by_axis(img=img, cmap=colormap)
            return cc.detach().cpu().numpy()
        finally:
            if original_tqdm is not None:
                viz.tqdm = original_tqdm
            if getattr(false_color_widget, '_abort_flag', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker = _false_color_worker()
    _add_cancel_button(false_color_widget, worker, pbar)
    return worker


# 3. Hilbert Transform Tool
@magicgui(
    call_button="Compute Hilbert Transform",
    return_type={"choices": ["amp_phase", "real_imag", "complex"]}
)
def hilbert_transform_widget(
    normalize_histogram: bool = True,
    batch_size: str = 'auto',
    return_type: str = 'amp_phase',
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    img = torch.from_numpy(layer.data.astype('float32'))
    hilbert_transform_widget._abort_flag = False
    pbar, emitter, MockTqdm = _setup_progress(hilbert_transform_widget, "Hilbert Transform...")

    @thread_worker
    def _hilbert_worker():
        import cellstream.hilbert as ht_module
        original_tqdm = getattr(ht_module, 'tqdm', None)
        ht_module.tqdm = MockTqdm
        try:
            bz = 'auto' if str(batch_size).lower() == 'auto' else int(batch_size)
            ht = hilbert_transform(
                image=img,
                normalize_histogram=normalize_histogram,
                batch_size=bz,
                return_type=return_type
            )
            return ht
        finally:
            if original_tqdm is not None:
                ht_module.tqdm = original_tqdm
            if getattr(hilbert_transform_widget, '_abort_flag', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker = _hilbert_worker()
    _add_cancel_button(hilbert_transform_widget, worker, pbar)
    return worker

# 4. FIR Filter Tool
@magicgui(call_button="Apply FIR Filter")
def fir_filter_widget(
    cutoff_freq: float = 0.1,
    high_pass: bool = False,
    window_size: int = 101,
    batch_size: str = 'auto',
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    img = torch.from_numpy(layer.data.astype('float32'))
    fir_filter_widget._abort_flag = False
    pbar, emitter, MockTqdm = _setup_progress(fir_filter_widget, "Applying Filter...")

    @thread_worker
    def _filter_worker():
        import cellstream.filters as flt_module
        original_tqdm = getattr(flt_module, 'tqdm', None)
        flt_module.tqdm = MockTqdm
        try:
            ir = create_ir_filter(cutoff_freq, high_pass=high_pass, window_size=window_size)
            bz = 'auto' if str(batch_size).lower() == 'auto' else int(batch_size)
            filtered = apply_fir_filter(img, ir, batch_size=bz)
            return filtered.cpu().numpy()
        finally:
            if original_tqdm is not None:
                flt_module.tqdm = original_tqdm
            if getattr(fir_filter_widget, '_abort_flag', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker = _filter_worker()
    _add_cancel_button(fir_filter_widget, worker, pbar)
    return worker

# 5. Phase Defects Tool
@magicgui(call_button="Compute Phase Defects")
def phase_defects_widget(
    window_size: int = 5,
    row_blocks: str = 'auto'
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    img = torch.from_numpy(layer.data.astype('float32'))
    phase_defects_widget._abort_flag = False
    pbar, emitter, MockTqdm = _setup_progress(phase_defects_widget, "Computing Winding Number...")

    @thread_worker
    def _phase_worker():
        from cellstream.phase import winding_number
        import cellstream.phase as phase_module
        original_tqdm = getattr(phase_module, 'tqdm', None)
        phase_module.tqdm = MockTqdm
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            rb = 'auto' if str(row_blocks).lower() == 'auto' else int(row_blocks)
            wn = winding_number(
                phase_img=img,
                n=window_size,
                row_blocks=rb,
                device=device
            )
            return wn.cpu().numpy()
        finally:
            if original_tqdm is not None:
                phase_module.tqdm = original_tqdm
            if getattr(phase_defects_widget, '_abort_flag', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker = _phase_worker()
    _add_cancel_button(phase_defects_widget, worker, pbar)
    return worker

# 6. Image Registration Tool
@magicgui(call_button="Register Timeseries")
def image_registration_widget(
    reg_channel: int = 0,
    downsample_factor: float = 0.25,
    registration_mode: str = 'RIGID_BODY',
    reference_mode: str = 'previous'
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    img = torch.from_numpy(layer.data.astype('float32'))
    image_registration_widget._abort_flag = False
    pbar, emitter, _ = _setup_progress(image_registration_widget, "Registering...")

    @thread_worker
    def _registration_worker():
        from cellstream.registration import register_and_transform_image_timeseries
        try:
            emitter.total_signal.emit(1)
            registered = register_and_transform_image_timeseries(
                img=img,
                reg_channel=reg_channel,
                downsample_factor=downsample_factor,
                registration_mode=registration_mode,
                reference_mode=reference_mode
            )
            if getattr(image_registration_widget, '_abort_flag', False):
                raise RuntimeError("Cancelled")
            emitter.update_signal.emit(1)
            return registered.detach().cpu().numpy()
        finally:
            if getattr(image_registration_widget, '_abort_flag', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker = _registration_worker()
    _add_cancel_button(image_registration_widget, worker, pbar)
    return worker

# 7. Pixel Spectrum Profiling Tool
@magicgui(call_button="Profile Pixel Spectra")
def pixel_profile_widget(
    min_bin: int = 4,
    max_bin: int = 40,
    fft_batch_size: str = 'auto',
    c_val: float = 35.0,
    filter_method: str = 'product',
    filter_channel: int = 0,
    peak_constraint: str = 'exactly_one'
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    img = torch.from_numpy(layer.data.astype('float32'))
    pixel_profile_widget._abort_flag = False
    pbar, emitter, MockTqdm = _setup_progress(pixel_profile_widget, "Profiling Pixels...")

    @thread_worker
    def _profile_worker():
        from cellstream.pixels.process import profile_image_pixels
        import cellstream.fft.utils as fft_utils
        original_tqdm = getattr(fft_utils, 'tqdm', None)
        fft_utils.tqdm = MockTqdm
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            bz = 'auto' if str(fft_batch_size).lower() == 'auto' else int(fft_batch_size)
            df = profile_image_pixels(
                img=img,
                c_val=c_val,
                min_bin=min_bin,
                max_bin=max_bin,
                filter_method=filter_method,
                filter_channel=filter_channel,
                peak_constraint=peak_constraint,
                fft_batch_size=bz,
                device=device
            )
            return df
        finally:
            if original_tqdm is not None:
                fft_utils.tqdm = original_tqdm
            if getattr(pixel_profile_widget, '_abort_flag', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker = _profile_worker()
    _add_cancel_button(pixel_profile_widget, worker, pbar)
    return worker

# 8. Landscape Generation Tool
_pixel_choices = ['E_amp', 'D_amp', 'E_norm_amp', 'D_norm_amp', 'E_z', 'D_z', 'E', 'D', 'E_sd', 'D_sd', 'F_bin']

@magicgui(
    call_button="Generate 2D Landscape",
    x_col={"choices": _pixel_choices},
    y_col={"choices": _pixel_choices},
    z_col={"choices": _pixel_choices},
    landscape_cmap={"choices": ['viridis', 'turbo', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm']}
)
def landscape_generation_widget(
    x_col: str = 'E',
    y_col: str = 'D',
    z_col: str = 'E_amp',
    landscape_cmap: str = 'viridis',
    landscape_bins: int = 100,
    min_count: int = 10,
    percentile_min: float = 1.0,
    percentile_max: float = 99.0
):
    return {
        'action': 'generate_landscape',
        'x_col': x_col,
        'y_col': y_col,
        'z_col': z_col,
        'cmap': landscape_cmap,
        'bins': landscape_bins,
        'min_count': min_count,
        'percentiles': (percentile_min, percentile_max)
    }
