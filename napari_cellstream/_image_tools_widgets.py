from magicgui import magicgui
from magicgui.widgets import PushButton
from napari.layers import Image, Labels
from napari import current_viewer
from qtpy.QtWidgets import QWidget
import torch
import numpy as np
from napari.qt.threading import thread_worker

from cellstream.image import downsample
from cellstream.hilbert import hilbert_transform
from cellstream.filters import create_ir_filter, apply_fir_filter, create_bandpass_filter
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
    axis: int = 0,
    colormap: str = 'turbo'
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    img = torch.from_numpy(layer.data.astype('float32'))
    img_ndim = img.dim()
    
    # Handle negative axis indexing properly for generic slicing
    _axis = axis if axis >= 0 else img_ndim + axis
    if not (0 <= _axis < img_ndim):
        raise ValueError(f"Invalid axis {_axis} for {img_ndim}D image")

    # Generically slice along the chosen axis
    slices = [slice(None)] * img_ndim
    slices[_axis] = slice(min_slice, max_slice)
    img = img[tuple(slices)]

    false_color_widget._abort_flag = False
    pbar, emitter, MockTqdm = _setup_progress(false_color_widget, "Generating colors...")

    @thread_worker
    def _false_color_worker():
        import cellstream.viz as viz
        original_tqdm = getattr(viz, 'tqdm', None)
        viz.tqdm = MockTqdm
        try:
            cc = color_by_axis(img=img, axis=axis, cmap=colormap)
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
@magicgui(
    call_button="Apply FIR Filter",
    filter_type={"choices": ["low_pass", "high_pass", "bandpass"]},
    cutoff_freq_low={"label": "Low Cutoff (Frac. Nyquist)"},
    cutoff_freq_high={"label": "High Cutoff (Frac. Nyquist)"}
)
def fir_filter_widget(
    filter_type: str = "low_pass",
    cutoff_freq_low: float = 0.05,
    cutoff_freq_high: float = 0.15,
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
            if filter_type == 'bandpass':
                ir = create_bandpass_filter(low_cutoff=cutoff_freq_low, high_cutoff=cutoff_freq_high, window_size=window_size)
            elif filter_type == 'high_pass':
                ir = create_ir_filter(cutoff_freq=cutoff_freq_low, high_pass=True, window_size=window_size)
            else: # low_pass
                ir = create_ir_filter(cutoff_freq=cutoff_freq_low, high_pass=False, window_size=window_size)
                
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

@fir_filter_widget.filter_type.changed.connect
def _on_filter_type_changed(value: str):
    fir_filter_widget.cutoff_freq_high.enabled = (value == 'bandpass')

fir_filter_widget.cutoff_freq_high.enabled = False

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

# 9. Hann Filter Tool
@magicgui(call_button="Apply Hann Filter")
def hann_filter_widget(
    norm_histogram: bool = False,
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    img = torch.from_numpy(layer.data.astype('float32'))
    hann_filter_widget._abort_flag = False
    pbar, emitter, MockTqdm = _setup_progress(hann_filter_widget, "Applying Hann Filter...")

    @thread_worker
    def _hann_worker():
        import cellstream.utils as cs_utils
        import importlib
        importlib.reload(cs_utils)
        from cellstream.utils import hann_image_series
        
        original_tqdm = getattr(cs_utils, 'tqdm', None)
        cs_utils.tqdm = MockTqdm
        try:
            res = hann_image_series(img=img, norm_histogram=norm_histogram)
            return res.detach().cpu().numpy()
        finally:
            if original_tqdm is not None:
                cs_utils.tqdm = original_tqdm
            if getattr(hann_filter_widget, '_abort_flag', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker = _hann_worker()
    _add_cancel_button(hann_filter_widget, worker, pbar)
    return worker


# 10. Temporal Convolution Tool
@magicgui(call_button="Convolve Timeseries")
def temporal_convolution_widget(
    kernel_weights: str = "1.0, -1.0",
    batch_size: str = '512',
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    img = torch.from_numpy(layer.data.astype('float32'))
    temporal_convolution_widget._abort_flag = False
    pbar, emitter, MockTqdm = _setup_progress(temporal_convolution_widget, "Convolving Timeseries...")

    @thread_worker
    def _conv_worker():
        import cellstream.utils as cs_utils
        import importlib
        importlib.reload(cs_utils)
        from cellstream.utils import convolve_along_timeseries
        
        original_tqdm = getattr(cs_utils, 'tqdm', None)
        cs_utils.tqdm = MockTqdm
        try:
            # Parse kernel
            try:
                weights = [float(x.strip()) for x in kernel_weights.split(',')]
            except ValueError:
                raise ValueError("Kernel weights must be a comma-separated list of numbers.")
            k_tensor = torch.tensor(weights, dtype=torch.float32)
            
            bz = 'auto' if str(batch_size).lower() == 'auto' else int(batch_size)
            if bz == 'auto': bz = 512
            
            res = convolve_along_timeseries(
                video_tensor=img,
                kernel_weights=k_tensor,
                batch_size=bz
            )
            return res.detach().cpu().numpy()
        finally:
            if original_tqdm is not None:
                cs_utils.tqdm = original_tqdm
            if getattr(temporal_convolution_widget, '_abort_flag', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker = _conv_worker()
    _add_cancel_button(temporal_convolution_widget, worker, pbar)
    return worker

# 11. Phase Velocity Tool
@magicgui(
    call_button="Extract Flow Field"
)
def phase_velocity_widget(
    smooth_sigma: float = 1.0,
    vector_spacing: int = 16,
    show_vectors: bool = False,
    show_angle_image: bool = False,
    show_magnitude_image: bool = False,
    show_wavelength_image: bool = False,
    show_streamlines: bool = False,
    show_phase_streamlines: bool = False,
    show_static_streamlines: bool = False,
    show_transport_highways: bool = False,
    show_ftle: bool = False,
    backward_ftle: bool = False,
    ftle_integration_time: int = 20,
    stream_particles: int = 20000,
    stream_decay: float = 0.85,
    stream_inject_rate: float = 0.05,
    use_mask: bool = True,
    stream_mask: Labels = None
):
    viewer = current_viewer()
    if viewer is None: raise RuntimeError("No active napari viewer found")
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image): raise RuntimeError("No active image layer selected")

    raw_shape = layer.data.shape
    img = torch.from_numpy(layer.data.astype('float32'))
    
    is_4d = len(raw_shape) == 4
    is_z_first = False
    
    if is_4d:
        if raw_shape[0] <= raw_shape[1]:
            img = img[0]
            is_z_first = True
        else:
            img = img[:, 0]
            is_z_first = False
            
    img = img.squeeze()
    
    if img.ndim != 3:
        raise ValueError(f"Expected a 3D phase image (T, Y, X). Got shape {raw_shape}.")

    # Resolve mask on main thread (Qt-safe) before entering worker
    mask_np = None
    if use_mask:
        mask_layer = stream_mask
        if mask_layer is None or mask_layer not in viewer.layers:
            # Fallback: auto-detect first Labels layer
            for l in viewer.layers:
                if isinstance(l, Labels):
                    mask_layer = l
                    break
        if mask_layer is not None:
            mask_np = mask_layer.data.astype('float32')

    phase_velocity_widget._abort_flag = False
    pbar, emitter, MockTqdm = _setup_progress(phase_velocity_widget, "Extracting Flow...")

    @thread_worker
    def _flow_worker():
        from cellstream.flow.analytic import phase_velocity, generate_streamlines, generate_instantaneous_streamlines, generate_phase_colored_streamlines, compute_ftle
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            v, speed, wavenumber = phase_velocity(img, smooth_sigma=smooth_sigma, device=device)
            T_out, _, Y_out, X_out = v.shape
            # Subsample visual arrows slightly
            step = max(1, vector_spacing)
            t_idx, y_idx, x_idx = np.meshgrid(
                np.arange(T_out), 
                np.arange(0, Y_out, step), 
                np.arange(0, X_out, step), 
                indexing='ij'
            )

            v_np = v.detach().cpu().numpy()
            
            outputs = []
            
            # 1. Images (Angle & Magnitude)
            if show_angle_image or show_magnitude_image:
                # v_np is (T, 2, Y, X)
                vy_full = v_np[:, 1, :, :]
                vx_full = v_np[:, 0, :, :]
                
                mag = np.sqrt(vx_full**2 + vy_full**2)
                ang = np.arctan2(vy_full, vx_full)
                
                # Reshape back to 4D if original was 4D
                if is_4d:
                    if is_z_first:
                        mag = np.expand_dims(mag, axis=0)
                        ang = np.expand_dims(ang, axis=0)
                    else:
                        mag = np.expand_dims(mag, axis=1)
                        ang = np.expand_dims(ang, axis=1)
                
                if show_magnitude_image:
                    outputs.append({
                        'action': 'add_image',
                        'data': mag,
                        'name': 'Velocity Magnitude',
                        'colormap': 'magma',
                        'scale': layer.scale,
                        'translate': layer.translate
                    })
                
                if show_angle_image:
                    outputs.append({
                        'action': 'add_image',
                        'data': ang,
                        'name': 'Velocity Angle',
                        'colormap': 'hsv',
                        'scale': layer.scale,
                        'translate': layer.translate
                    })
                    
            if show_wavelength_image:
                wave_np = wavenumber.detach().cpu().numpy()
                wavelength = 2 * np.pi / (wave_np + 1e-8)
                
                # Cap the maximum wavelength for display (100 pixels is visually huge)
                wavelength = np.clip(wavelength, 0, 100)
                
                if is_4d:
                    if is_z_first:
                        wavelength = np.expand_dims(wavelength, axis=0)
                    else:
                        wavelength = np.expand_dims(wavelength, axis=1)
                        
                outputs.append({
                    'action': 'add_image',
                    'data': wavelength,
                    'name': 'Spatial Wavelength',
                    'colormap': 'turbo',
                    'scale': layer.scale,
                    'translate': layer.translate
                })
                    
            # 2. Dynamic Streamlines (Comet Tails)
            mask_tensor = None
            if mask_np is not None:
                mask_tensor = torch.from_numpy(mask_np).squeeze()
                if is_4d:
                    if mask_tensor.ndim == 4:
                        mask_tensor = mask_tensor[0] if is_z_first else mask_tensor[:, 0]
            
            if show_streamlines or show_transport_highways:
                stream_img = generate_streamlines(
                    v, 
                    num_particles=stream_particles, 
                    decay=stream_decay, 
                    device=device,
                    mask=mask_tensor,
                    inject_rate=stream_inject_rate
                )
                
                if show_streamlines:
                    stream_np = stream_img.cpu().numpy()
                    
                    if is_4d:
                        if is_z_first:
                            stream_np = np.expand_dims(stream_np, axis=0)
                        else:
                            stream_np = np.expand_dims(stream_np, axis=1)
                            
                    outputs.append({
                        'action': 'add_image',
                        'data': stream_np,
                        'name': 'Dynamic Streamlines',
                        'colormap': 'inferno',
                        'scale': layer.scale,
                        'translate': layer.translate
                    })
                    
                if show_transport_highways:
                    # Time-average the pathlines to get stable Lagrangian coherent structures
                    highways = stream_img.mean(dim=0).cpu().numpy()
                    outputs.append({
                        'action': 'add_image',
                        'data': highways,
                        'name': 'Transport Highways (LCS)',
                        'colormap': 'magma',
                        'scale': layer.scale[-2:],
                        'translate': layer.translate[-2:]
                    })
                    
            if show_phase_streamlines:
                phase_stream_img = generate_phase_colored_streamlines(
                    v, 
                    phase=img,
                    num_particles=stream_particles, 
                    decay=stream_decay, 
                    device=device,
                    mask=mask_tensor,
                    inject_rate=stream_inject_rate
                )
                phase_stream_np = phase_stream_img.cpu().numpy()
                
                # Input shape is (T, 3, Y, X)
                # Output shape for napari should be (T, Y, X, 3) for RGB display
                rgb_np = np.moveaxis(phase_stream_np, 1, -1)
                
                if is_4d:
                    if is_z_first:
                        rgb_np = np.expand_dims(rgb_np, axis=0)
                    else:
                        rgb_np = np.expand_dims(rgb_np, axis=1)
                        
                outputs.append({
                    'action': 'add_image',
                    'data': rgb_np,
                    'name': 'Phase-Colored Flow',
                    'rgb': True,
                    'scale': layer.scale,
                    'translate': layer.translate
                })
                    
            if show_static_streamlines:
                static_img = generate_instantaneous_streamlines(
                    v,
                    num_particles=stream_particles,
                    steps=50,
                    device=device,
                    mask=mask_tensor
                )
                static_np = static_img.cpu().numpy()
                
                if is_4d:
                    if is_z_first:
                        static_np = np.expand_dims(static_np, axis=0)
                    else:
                        static_np = np.expand_dims(static_np, axis=1)
                        
                outputs.append({
                    'action': 'add_image',
                    'data': static_np,
                    'name': 'Static Streamlines',
                    'colormap': 'inferno',
                    'scale': layer.scale,
                    'translate': layer.translate
                })
            
            if show_ftle:
                ftle_img = compute_ftle(
                    v,
                    integration_time=ftle_integration_time,
                    device=device,
                    mask=mask_tensor,
                    backward=backward_ftle
                )
                ftle_np = ftle_img.cpu().numpy()
                
                if is_4d:
                    if is_z_first:
                        ftle_np = np.expand_dims(ftle_np, axis=0)
                    else:
                        ftle_np = np.expand_dims(ftle_np, axis=1)
                        
                outputs.append({
                    'action': 'add_image',
                    'data': ftle_np,
                    'name': 'FTLE',
                    'colormap': 'inferno',
                    'scale': layer.scale,
                    'translate': layer.translate
                })
            
            # 3. Vectors
            if show_vectors:
                vx = v_np[:, 0, ::step, ::step].flatten()
                vy = v_np[:, 1, ::step, ::step].flatten()
                    
                t_flat = t_idx.flatten()
                y_flat = y_idx.flatten()
                x_flat = x_idx.flatten()
                dt = np.zeros_like(vx)
                
                if is_4d:
                    z_flat = np.zeros_like(t_flat)
                    if is_z_first:
                        starts = np.stack([z_flat, t_flat, y_flat, x_flat], axis=1)
                        directions = np.stack([dt, dt, vy, vx], axis=1)
                    else:
                        starts = np.stack([t_flat, z_flat, y_flat, x_flat], axis=1)
                        directions = np.stack([dt, dt, vy, vx], axis=1)
                else:
                    starts = np.stack([t_flat, y_flat, x_flat], axis=1)
                    directions = np.stack([dt, vy, vx], axis=1)
                    
                napari_vectors = np.stack([starts, directions], axis=1)
                angles = np.arctan2(vy, vx)
                
                outputs.append({
                    'action': 'add_vectors', 
                    'data': napari_vectors, 
                    'name': 'Velocity Vectors',
                    'features': {'angle': angles},
                    'edge_color': 'angle',
                    'edge_colormap': 'hsv',
                    'scale': layer.scale,
                    'translate': layer.translate
                })
            
            return outputs
            
        finally:
            if getattr(phase_velocity_widget, '_abort_flag', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker = _flow_worker()
    _add_cancel_button(phase_velocity_widget, worker, pbar)
    return worker
