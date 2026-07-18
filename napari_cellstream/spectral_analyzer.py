# -*- coding: utf-8 -*-
"""
@author: coylelab @ UW-Madison
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import nd2
import tifffile
import time
import zarr

from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                           QComboBox, QSpinBox, QCheckBox, QHBoxLayout,
                           QGroupBox, QDoubleSpinBox, QFormLayout,QFileDialog,
                           QScrollArea, QSplitter, QTreeWidget, QTreeWidgetItem,
                           QFrame)

from qtpy.QtCore import Qt, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from qtpy.QtWidgets import QSizePolicy


from ssqueezepy import cwt

from ._fft_widget import fft_gui_widget
from ._cwt_widget import generate_cwt_features_widget
from ._image_tools_widgets import (
    downsample_gui_widget,
    false_color_widget,
    hilbert_transform_widget,
    fir_filter_widget,
    phase_defects_widget,
    image_registration_widget,
    pixel_profile_widget,
    landscape_generation_widget,
    hann_filter_widget,
    temporal_convolution_widget,
    phase_velocity_widget
)

from qtpy.QtWidgets import QStackedWidget

import logging

logger = logging.getLogger(__name__)

# Define wavelet parameter options with default values and ranges
WAVELET_PARAMS = {
    'gmw': { "gamma": (3,0,1000,1),
             "beta": (60 ,0,1000,1)
            },
    'morlet': { "mu": (13.4,0,1000,1)
            },
    
    
    'bump': { "mu": (5,0,1000,1),
             "s": (1 ,0,1000,1),
             "om": (0,0,1000,1)
            },
    
    'cmhat': { "mu": (1,0,1000,1),
             "s": (1 ,0,1000,1)
             },
    
    'hhhat': { "mu": (5,0,1000,1)
            }}



class SpectralWidget(QWidget):
    def __init__(self, napari_viewer,use_gpu=False):
        super().__init__()
        self.viewer = napari_viewer
        self.use_gpu=use_gpu
        self.cid = None
        self.canvas = None
        self.last_x = None
        self.last_y = None
        self.last_layer = None
        
        #timelines
        self.current_time_index = 0
        self.time_cursor_lines = []
        self.spectrogram_images = []
        self.cwt_magnitudes = []
        self.toolbar = None
        self._axes = None
        self._n_channels = 0
        self._linking = False
        self.viewer.dims.events.current_step.connect(self.update_time_cursor) #time line
        self.last_update_time = 0  # store time of last update
        self.update_interval = .25  # seconds (5 Hz)
        
        # Feature dictionary tree results store
        self.cwt_count = 0
        self.fft_count = 0
        self.results_dict = {}
        
        # Default settings
        self.wavelet = "gmw"
        self.nv = 32
        self.do_plot_zscore = True
        self.wavelet_params = {}
        self.show_nyquist = False
        
        # Create controls
        self.create_controls()
        
        # Activation toggle
        self.activate_button = QPushButton("Activate Pixel Inspector")
        self.activate_button.setCheckable(True)
        self.activate_button.setChecked(True)
        self.activate_button.toggled.connect(self.toggle_activation)
        
        # Status indicator
        self.status_label = QLabel("Status: ACTIVE - Shift+Click on image")
        
        # Plot container
        self.plot_container = QWidget()
        self.plot_container.setLayout(QVBoxLayout())
        self.plot_container.setMinimumWidth(100)
        
        #fft widget 
        self.fft_gui = fft_gui_widget
        self.fft_gui.use_gpu.value = self.use_gpu
        if not self.use_gpu:
            self.fft_gui.use_gpu.enabled = False
        self.fft_gui.called.connect(self.handle_fft_result)
        
        fft_group = QGroupBox("     Generate FFT features")
        fft_group.setCheckable(True)
        fft_group.setChecked(False)
        self.fft_gui.native.setVisible(False)
        fft_group.toggled.connect(self.fft_gui.native.setVisible)
        
        fft_layout = QVBoxLayout()
        fft_layout.addWidget(self.fft_gui.native)
        fft_group.setLayout(fft_layout)

        #cwt widget 
        self.cwt_gui = generate_cwt_features_widget
        self.cwt_gui.use_gpu.value = self.use_gpu
        if not self.use_gpu:
            self.cwt_gui.use_gpu.enabled = False

        #self.cwt_gui.wavelet_tuple=self.get_wavelet_tuple() ##fix this later
        self.cwt_gui.called.connect(self.handle_cwt_result)
        
        cwt_group = QGroupBox("     Generate CWT features")
        cwt_group.setCheckable(True)
        cwt_group.setChecked(False)
        self.cwt_gui.native.setVisible(False)
        cwt_group.toggled.connect(self.cwt_gui.native.setVisible)
        
        cwt_layout = QVBoxLayout()
        cwt_layout.addWidget(self.cwt_gui.native)
        cwt_group.setLayout(cwt_layout)
        
        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.open_file_dialog)

        # Save plots button
        self.save_plot_button = QPushButton("Save Timeseries")
        self.save_plot_button.clicked.connect(self.save_timeseries)
        self.plot_container.layout().addWidget(self.save_plot_button, 0, Qt.AlignTop)
        
        # Left column: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)
        left_panel.setLayout(left_layout)
        
        left_layout.addWidget(self.load_button)
        left_layout.addSpacing(32)
        left_layout.addWidget(self.controls_group)
        left_layout.addWidget(fft_group)
        left_layout.addWidget(cwt_group)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(100)
        scroll_area.setWidget(left_panel)

        # Right column: Results History Tree
        self.results_panel = QGroupBox("     Results History")
        results_layout = QVBoxLayout()
        self.results_panel.setLayout(results_layout)

        self.results_tree = QTreeWidget()
        self.results_tree.setColumnCount(2)
        self.results_tree.setHeaderLabels(["Key/Channel/Feature", "Value/Type"])
        self.results_tree.setColumnWidth(0, 180)
        self.results_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        results_layout.addWidget(self.results_tree)

        # Buttons under the tree
        buttons_layout = QHBoxLayout()
        
        self.write_zarr_button = QPushButton("Write to Zarr")
        self.write_zarr_button.clicked.connect(self.save_result_to_zarr)
        buttons_layout.addWidget(self.write_zarr_button)

        self.load_zarr_button = QPushButton("Load Zarr")
        self.load_zarr_button.clicked.connect(self.load_result_from_zarr)
        buttons_layout.addWidget(self.load_zarr_button)
        
        self.delete_result_button = QPushButton("Trash")
        self.delete_result_button.setStyleSheet("color: #ff6b6b;")
        self.delete_result_button.clicked.connect(self.delete_selected_result)
        buttons_layout.addWidget(self.delete_result_button)
        
        results_layout.addLayout(buttons_layout)

        ### Main layout using splitter ###
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_area)
        splitter.addWidget(self.plot_container)
        splitter.setSizes([300, 300])  

        main = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        main.setLayout(main_layout)
        
        ### Top widget: two side-by-side panels

        top_widget = QWidget()
        top_row = QHBoxLayout(top_widget)
        top_row.addWidget(splitter)
        

        ###Bottom widget that spans both
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout()
        
        # Image Tools Widget
        image_tools_group = QGroupBox("  Image Toolbox")
        image_tools_layout = QVBoxLayout()
        
        self.image_tools_combo = QComboBox()
        self.image_tools_combo.addItems([
            "Downsample Image", 
            "False-Color Spectra", 
            "Hilbert Transform",
            "FIR Filter",
            "Phase Defects",
            "Image Registration",
            "Pixel Profile Spectra",
            "Generate 2D Landscape",
            "Hann Filter",
            "Temporal Convolution",
            "Phase Velocity"
        ])
        
        self.image_tools_stack = QStackedWidget()
        
        for w in [
            downsample_gui_widget.native,
            false_color_widget.native,
            hilbert_transform_widget.native,
            fir_filter_widget.native,
            phase_defects_widget.native,
            image_registration_widget.native,
            pixel_profile_widget.native,
            landscape_generation_widget.native,
            hann_filter_widget.native,
            temporal_convolution_widget.native,
            phase_velocity_widget.native
        ]:
            container = QWidget()
            lay = QVBoxLayout(container)
            lay.setAlignment(Qt.AlignTop)
            lay.addWidget(w)
            self.image_tools_stack.addWidget(container)
        
        self.image_tools_combo.currentIndexChanged.connect(self.image_tools_stack.setCurrentIndex)
        
        def _dispatch_tool_result(result, tool_name):
            if hasattr(result, 'returned') and hasattr(result, 'start'):
                # It's a thread worker, connect it and start it
                restore_func = self._show_activity_dock()
                result.returned.connect(lambda r: self.handle_image_tool_result(r, tool_name))
                result.finished.connect(restore_func)
                result.start()
            else:
                self.handle_image_tool_result(result, tool_name)
                
        _tool_widgets = [
            (downsample_gui_widget, "Downsampled"),
            (false_color_widget, "False Colored"),
            (hilbert_transform_widget, "Hilbert Transform"),
            (fir_filter_widget, "FIR Filter"),
            (phase_defects_widget, "Phase Defects"),
            (image_registration_widget, "Registered Image"),
            (pixel_profile_widget, "Pixel Profile Spectra"),
            (landscape_generation_widget, "Generated Landscape"),
            (hann_filter_widget, "Hann Filter"),
            (temporal_convolution_widget, "Temporal Convolution"),
            (phase_velocity_widget, "Phase Velocity"),
        ]
        for widget, name in _tool_widgets:
            try:
                widget.called.disconnect()
            except (ValueError, TypeError):
                pass
            widget.called.connect(lambda r, n=name: _dispatch_tool_result(r, n))

        # Wrap the stack in a QScrollArea so large widgets don't break the layout
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_tools_stack)
        
        image_tools_layout.addWidget(self.image_tools_combo)
        image_tools_layout.addWidget(scroll_area)
        image_tools_group.setLayout(image_tools_layout)

        #Add widgets to bottom panel
        bottom_layout.addWidget(self.results_panel, 2)
        bottom_layout.addWidget(image_tools_group, 2)
        bottom_panel.setLayout(bottom_layout)
        bottom_panel.setMinimumWidth(100)

        ### Connect top and bottom widgets
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_panel)
        splitter.setSizes([550, 250])  
        
        #finalize and display layout
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.setMinimumWidth(150)

        #initialize connections
        self.toggle_activation(True)
        self.propagate_wavelet_params_to_cwt_widget()
        self.fmax=self.fft_gui.max_bin.value

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.nd2)")
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, path: str):
        logger.info(f"Loading image from: {path}")
        *iname, iext = path.split('.')
        if iext == 'nd2':
            image = nd2.imread(path)
        elif iext == 'tif':
            image = tifffile.imread(path)
        else:
            logger.error("Invalid type")
            return
        #check dimension -- insert dummy channel dimension if need by
        if image.ndim==3:
            image=np.expand_dims(image,axis=1)
            logger.info("Inserting channel dimension...")
        
        self.viewer.add_image(image, name=iname[-1])
        return
    
    def save_timeseries(self):
        if self.last_layer is None or self.last_x is None or self.last_y is None:
            logger.warning("No pixel selected to save.")
            return
            
        data = self.last_layer.data
        if data.ndim == 3:
            data = data[:, np.newaxis, ...]  # Add channel dimension
        T, C, X, Y = data.shape
        
        # Get time series data for all channels
        ts_data = []
        for c in range(C):
            ts = data[:, c, self.last_x, self.last_y].astype(np.float64)
            ts_data.append(ts)
            
        ts_array = np.column_stack(ts_data)
        
        default_name = f"pixel_{self.last_x}_{self.last_y}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Timeseries", 
                default_name,
                "CSV Files (*.csv);;Numpy Files (*.npy);;All Files (*)"
            )
            
        if file_path:
            if file_path.endswith('.npy'):
                np.save(file_path, ts_array)
            else:
                header = ",".join([f"Channel_{c}" for c in range(C)])
                np.savetxt(file_path, ts_array, delimiter=",", header=header, comments="")
            logger.info(f"Timeseries saved to: {file_path}")
    
    def _show_activity_dock(self):
        restore_funcs = []
        try:
            # Modern napari (>=0.4.16)
            if hasattr(self.viewer.window, '_qt_viewer') and hasattr(self.viewer.window._qt_viewer, 'dockActivity'):
                dock = self.viewer.window._qt_viewer.dockActivity
                if not dock.isVisible():
                    dock.show()
                    restore_funcs.append(lambda: dock.setVisible(False))
            # Older napari
            elif hasattr(self.viewer.window, '_qt_window') and hasattr(self.viewer.window._qt_window, '_activity_dialog'):
                dialog = self.viewer.window._qt_window._activity_dialog
                if not dialog.isVisible():
                    dialog.setVisible(True)
                    restore_funcs.append(lambda: dialog.setVisible(False))
        except Exception as e:
            logger.debug(f"Could not show activity dock: {e}")

        def restore():
            for f in restore_funcs:
                try:
                    f()
                except Exception:
                    pass
        return restore

    ###FFT widget components
    def handle_fft_result(self, result):
        if hasattr(result, 'returned') and hasattr(result, 'start'):
            # It's a thread worker
            restore_func = self._show_activity_dock()
            result.returned.connect(self._process_fft_result)
            result.finished.connect(restore_func)
            result.start()
        else:
            self._process_fft_result(result)

    def _process_fft_result(self, result):
        if not isinstance(result, dict):
            logger.error("FFT did not return a valid result")
            return

        # Add to results tree
        self.fft_count += 1
        root_name = f"FFT_Result_{self.fft_count}"
        root_item = QTreeWidgetItem(self.results_tree)
        root_item.setText(0, root_name)
        root_item.setText(1, f"Dict ({len(result)} keys)")
        self.populate_tree(root_item, result)
        self.results_dict[id(root_item)] = result

        for key, data in result.items():
            if key == '_attrs':
                continue
            name = f"FFT_{key}"
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            self.viewer.add_image(data, name=name)

    ###CWT widget components
    def propagate_wavelet_params_to_cwt_widget(self):
        self.cwt_gui.nv.value=self.nv
        wavelet_tuple=self.get_wavelet_tuple()
        wavelet_choice=wavelet_tuple[0]
        wavelet_params=wavelet_tuple[1]
        self.cwt_gui.wavelet_choice.value=wavelet_choice
        self.cwt_gui.wavelet_parameters.value=wavelet_params

    def handle_cwt_result(self, results):
        if hasattr(results, 'returned') and hasattr(results, 'start'):
            restore_func = self._show_activity_dock()
            results.returned.connect(self._process_cwt_result)
            results.finished.connect(restore_func)
            results.start()
        else:
            self._process_cwt_result(results)

    def _process_cwt_result(self, results):
        # Add to results tree
        self.cwt_count += 1
        wavelet_name = self.cwt_gui.wavelet_choice.value
        nv_val = self.cwt_gui.nv.value
        root_name = f"CWT_Result_{self.cwt_count} (wavelet={wavelet_name}, nv={nv_val})"
        root_item = QTreeWidgetItem(self.results_tree)
        root_item.setText(0, root_name)
        root_item.setText(1, f"Dict ({len(results)} channels)")
        self.populate_tree(root_item, results)
        self.results_dict[id(root_item)] = results

        #reorganize spectra
        consolidated = {}
        for channel_key, ch_data in results.items():
            if channel_key == '_attrs':
                continue
            for key, arr in ch_data.items():
                if key not in consolidated:
                    consolidated[key] = []
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                elif hasattr(arr, "numpy"):
                    arr = arr.numpy()
                consolidated[key].append(arr)
        
        #convert to array
        for key in consolidated:
            consolidated[key] = np.stack(consolidated[key], axis=0)
    
        # Add results to Napari viewer
        for feature, array in consolidated.items():
            layer_name = f"{feature}"
            self.viewer.add_image(
                array.swapaxes(0,1),  #  organize as (T, num_filter_banks "Z", C, X, Y)
                name=layer_name,
                scale=[20,1,1],
                metadata={"source": "cwt", "feature": feature}
            )

    ### Image Tools unified handler
    def handle_image_tool_result(self, result, tool_name):
        import pandas as pd
        
        # If the result is a list or tuple, process each item recursively
        if isinstance(result, (list, tuple)):
            for res in result:
                self.handle_image_tool_result(res, tool_name)
            return
            
        if isinstance(result, dict) and result.get('action') == 'add_vectors':
            napari_vectors = result['data']
            name = result.get('name', tool_name)
            
            kwargs = {
                'edge_width': 1.5,
                'length': 1,
                'name': name,
                'edge_color': result.get('edge_color', 'cyan')
            }
            
            if 'features' in result:
                kwargs['features'] = result['features']
            if 'edge_colormap' in result:
                kwargs['edge_colormap'] = result['edge_colormap']
            if 'scale' in result:
                kwargs['scale'] = result['scale']
            if 'translate' in result:
                kwargs['translate'] = result['translate']
                
            self.viewer.add_vectors(napari_vectors, **kwargs)
            return

        if isinstance(result, dict) and result.get('action') == 'add_image':
            image_data = result['data']
            name = result.get('name', tool_name)
            
            kwargs = {'name': name}
            if 'colormap' in result:
                kwargs['colormap'] = result['colormap']
            if 'scale' in result:
                kwargs['scale'] = result['scale']
            if 'translate' in result:
                kwargs['translate'] = result['translate']
            if 'rgb' in result:
                kwargs['rgb'] = result['rgb']
                
            self.viewer.add_image(image_data, **kwargs)
            return

        if isinstance(result, dict) and result.get('action') == 'generate_landscape':
            current_item = self.results_tree.currentItem()
            if current_item is None:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Selection", "Please select a Pixel Profile DataFrame in the Results tree first.")
                return
            
            df = self.results_dict.get(id(current_item))
            if not isinstance(df, pd.DataFrame):
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Invalid Selection", "Selected item is not a Pixel Profile DataFrame.")
                return
            
            plot_config = result
            
            # Now compute and plot it!
            from cellstream.pixels.utils import compute_2d_landscape, plot_2d_landscape
            import matplotlib.pyplot as plt
            try:
                x_col = plot_config['x_col']
                y_col = plot_config['y_col']
                z_col = plot_config['z_col']
                
                stats = compute_2d_landscape(
                    df, 
                    x_col=x_col, 
                    y_col=y_col, 
                    z_col=z_col,
                    bins=plot_config.get('bins', 100),
                    min_count=plot_config.get('min_count', 10),
                    percentiles=plot_config.get('percentiles', (1.0, 99.0))
                )
                
                # Add stats to results tree so it can be saved
                root = self.results_tree.invisibleRootItem()
                stats_item = QTreeWidgetItem([f"Landscape Stats ({z_col})", f"Dict (shape: {stats['mean'].shape})"])
                root.addChild(stats_item)
                self.results_dict[id(stats_item)] = stats
                stats_item.setExpanded(True)
                self.results_tree.setCurrentItem(stats_item)
                
                plot_2d_landscape(
                    stats, 
                    title=f"Landscape: {z_col} over {x_col} vs {y_col}",
                    x_label=x_col,
                    y_label=y_col,
                    colorbar_label=z_col,
                    cmap=plot_config.get('cmap', 'viridis')
                )
                plt.show()
            except Exception as e:
                logger.error(f"Failed to plot 2D landscape: {e}")
            return

        if isinstance(result, pd.DataFrame):
            desc = f"DataFrame {result.shape}"
            logger.info(f"{tool_name} returned a DataFrame with {result.shape[0]} rows.")
            
            root = self.results_tree.invisibleRootItem()
            tool_item = QTreeWidgetItem([tool_name, desc])
            root.addChild(tool_item)
            self.results_dict[id(tool_item)] = result
            
        elif isinstance(result, dict):
            desc = f"Dict ({len(result)} channels)"
            root = self.results_tree.invisibleRootItem()
            tool_item = QTreeWidgetItem([tool_name, desc])
            root.addChild(tool_item)
            self.results_dict[id(tool_item)] = result
            
            for key, array in result.items():
                if isinstance(array, torch.Tensor):
                    array = array.cpu().numpy()
                elif hasattr(array, 'numpy'):
                    array = array.numpy()
                
                # add to viewer
                self.viewer.add_image(
                    array,
                    name=f"{tool_name} [{key}]",
                    metadata={"source": tool_name, "channel": key}
                )
                
                child_item = QTreeWidgetItem([str(key), f"Array {array.shape}"])
                tool_item.addChild(child_item)
                self.results_dict[id(child_item)] = array
                
        else:
            desc = f"Array {result.shape}"
            self.viewer.add_image(
                result,  
                name=f"{tool_name} Result",
                metadata={"source": tool_name}
            )
            root = self.results_tree.invisibleRootItem()
            tool_item = QTreeWidgetItem([tool_name, desc])
            root.addChild(tool_item)
            self.results_dict[id(tool_item)] = result
        
        # Expand and select
        tool_item.setExpanded(True)
        self.results_tree.setCurrentItem(tool_item)

    ###pixel inspector components
    def create_controls(self):
        """Create the parameter controls"""
        self.controls_group = QGroupBox("     Spectrum Settings")
        self.controls_group.setCheckable(True)
        self.controls_group.setChecked(True)

        # Inner content widget — toggled to show/hide just like FFT/CWT groups
        controls_content = QWidget()
        controls_content.setVisible(True)
        self.controls_group.toggled.connect(controls_content.setVisible)
        controls_layout = QVBoxLayout(controls_content)
        
        # Wavelet selector
        wavelet_layout = QHBoxLayout()
        wavelet_layout.addWidget(QLabel("Wavelet:"))
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(list(WAVELET_PARAMS.keys()))
        self.wavelet_combo.setCurrentText(self.wavelet)
        self.wavelet_combo.currentTextChanged.connect(self.wavelet_changed)
        wavelet_layout.addWidget(self.wavelet_combo)
        controls_layout.addLayout(wavelet_layout)
        
        # Wavelet parameters container
        self.params_container = QGroupBox("Wavelet Parameters")
        self.params_layout = QFormLayout()
        self.params_container.setLayout(self.params_layout)
        controls_layout.addWidget(self.params_container)
        self.create_wavelet_params_controls()
        
        # Scales selector
        nv_layout = QHBoxLayout()
        nv_layout.addWidget(QLabel("Number of Scales:"))
        self.nv_spin = QSpinBox()
        self.nv_spin.setRange(1, 1000)
        self.nv_spin.setValue(self.nv)
        self.nv_spin.valueChanged.connect(self.nv_changed)
        nv_layout.addWidget(self.nv_spin)
        controls_layout.addLayout(nv_layout)
        
        # Zscore scale checkbox
        self.zscore_check = QCheckBox("Normalize Spectra")
        self.zscore_check.setChecked(self.do_plot_zscore)
        self.zscore_check.stateChanged.connect(self.zscore_changed)
        controls_layout.addWidget(self.zscore_check)
        
        # X-axis Nyquist units checkbox
        self.nyquist_check = QCheckBox("Show X-axis as Fraction of Nyquist")
        self.nyquist_check.setChecked(self.show_nyquist)
        self.nyquist_check.stateChanged.connect(self.nyquist_changed)
        controls_layout.addWidget(self.nyquist_check)
        
        # Spectrogram contrast controls
        contrast_group = QGroupBox("Spectrogram Contrast")
        contrast_layout_inner = QVBoxLayout()
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.contrast_combo = QComboBox()
        self.contrast_combo.addItems(["Auto", "Manual"])
        self.contrast_combo.currentTextChanged.connect(self.contrast_mode_changed)
        mode_layout.addWidget(self.contrast_combo)
        contrast_layout_inner.addLayout(mode_layout)
        
        # Dual-handle range slider (like napari contrast limits)
        from superqt import QDoubleRangeSlider
        self.contrast_slider_widget = QWidget()
        slider_layout = QVBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        self.contrast_slider = QDoubleRangeSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 1)
        self.contrast_slider.setValue((0, 1))
        self.contrast_slider.valueChanged.connect(self._on_contrast_slider_changed)
        slider_layout.addWidget(self.contrast_slider)
        self.contrast_limits_label = QLabel("0.000 – 1.000")
        self.contrast_limits_label.setStyleSheet("font-size: 10px; color: #aaa;")
        slider_layout.addWidget(self.contrast_limits_label)
        self.contrast_slider_widget.setLayout(slider_layout)
        self.contrast_slider_widget.setVisible(False)
        contrast_layout_inner.addWidget(self.contrast_slider_widget)
        
        contrast_group.setLayout(contrast_layout_inner)
        controls_layout.addWidget(contrast_group)
        
        # Time cursor tracking toggle
        self.track_cursor_check = QCheckBox("Track Playback Cursor")
        self.track_cursor_check.setChecked(True)
        self.track_cursor_check.setToolTip("Uncheck to disable plot redraws while scrubbing the movie")
        controls_layout.addWidget(self.track_cursor_check)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh Plots")
        self.refresh_button.clicked.connect(self.refresh_plots)
        controls_layout.addWidget(self.refresh_button)
        
        # Wrap content widget in the group's own layout
        group_layout = QVBoxLayout()
        group_layout.addWidget(controls_content)
        self.controls_group.setLayout(group_layout)
    
    def create_wavelet_params_controls(self):
        """Create controls for the current wavelet's parameters"""
        # Clear existing controls
        while self.params_layout.rowCount() > 0:
            self.params_layout.removeRow(0)
        
        # Get parameters for current wavelet
        params = WAVELET_PARAMS.get(self.wavelet, {})
        self.wavelet_params = {}
        
        # Create controls for each parameter
        for param_name, param_config in params.items():
            # Unpack configuration: (default, min, max, step)
            default, min_val, max_val, step = param_config
            
            # Create appropriate spin box
            if isinstance(default, int):
                spinbox = QSpinBox()
                spinbox.setRange(int(min_val), int(max_val))
                spinbox.setSingleStep(int(step))
            else:
                spinbox = QDoubleSpinBox()
                spinbox.setRange(min_val, max_val)
                spinbox.setSingleStep(step)
                spinbox.setDecimals(2)
            
            spinbox.setValue(default)
            spinbox.valueChanged.connect(self.param_changed)
            
            # Store the control and its parameter name
            self.wavelet_params[param_name] = spinbox
            self.params_layout.addRow(QLabel(f"{param_name}:"), spinbox)
        
        # Show/hide container based on whether there are parameters
        self.params_container.setVisible(len(params) > 0)
    

    ### handle dynamic updating of plots and setting cwt_params
    def wavelet_changed(self, value):
        self.wavelet = value
        self.create_wavelet_params_controls()
        self.refresh_plots()
        self.propagate_wavelet_params_to_cwt_widget() # propagate params to CWT_widget

    def param_changed(self, value):
        self.refresh_plots()
        self.propagate_wavelet_params_to_cwt_widget()

    def nv_changed(self, value):
        self.nv = value
        self.refresh_plots()
        self.propagate_wavelet_params_to_cwt_widget()
    
    def zscore_changed(self, state):
        self.do_plot_zscore = (state == 2)  # 2 is checked
        self.refresh_plots()
        self.propagate_wavelet_params_to_cwt_widget()
        
    def nyquist_changed(self, state):
        self.show_nyquist = (state == 2)
        self.refresh_plots()
    
    def contrast_mode_changed(self, mode):
        """Handle contrast mode combo box changes."""
        self.contrast_mode = mode
        self.contrast_slider_widget.setVisible(mode == "Manual")
        if mode == "Manual" and self.cwt_magnitudes:
            # Initialize slider range with generous padding so user can
            # drag handles beyond the current data range
            all_min = min(np.nanmin(m) for m in self.cwt_magnitudes)
            all_max = max(np.nanmax(m) for m in self.cwt_magnitudes)
            span = all_max - all_min if all_max > all_min else 1.0
            padded_min = all_min - 0.5 * span
            padded_max = all_max + 0.5 * span
            self.contrast_slider.blockSignals(True)
            self.contrast_slider.setRange(padded_min, padded_max)
            self.contrast_slider.setValue((all_min, all_max))
            self.contrast_slider.blockSignals(False)
            self._update_contrast_label(all_min, all_max)
        self.apply_contrast_from_mode()
    
    def apply_contrast_from_mode(self):
        """Apply contrast limits to spectrograms based on current mode."""
        if not self.spectrogram_images or not self.cwt_magnitudes:
            return
        
        mode = self.contrast_combo.currentText()
        
        if mode == "Manual":
            # Expand slider range if new data exceeds current bounds,
            # but do NOT move the handles — preserves user's chosen contrast
            all_min = min(np.nanmin(m) for m in self.cwt_magnitudes)
            all_max = max(np.nanmax(m) for m in self.cwt_magnitudes)
            span = all_max - all_min if all_max > all_min else 1.0
            range_min = min(self.contrast_slider.minimum(), all_min - 0.5 * span)
            range_max = max(self.contrast_slider.maximum(), all_max + 0.5 * span)
            self.contrast_slider.blockSignals(True)
            self.contrast_slider.setRange(range_min, range_max)
            self.contrast_slider.blockSignals(False)
            vmin, vmax = self.contrast_slider.value()
        
        for im, cwt_mag in zip(self.spectrogram_images, self.cwt_magnitudes):
            if mode == "Auto":
                v0, v1 = np.nanmin(cwt_mag), np.nanmax(cwt_mag)
            elif mode == "Manual":
                v0, v1 = vmin, vmax
            else:
                return
            im.set_clim(v0, v1)
        
        if self.canvas:
            self.canvas.draw_idle()
    
    def _on_contrast_slider_changed(self, value):
        """Handle contrast range slider value changes."""
        vmin, vmax = value
        self._update_contrast_label(vmin, vmax)
        if self.contrast_combo.currentText() == "Manual":
            self.apply_contrast_from_mode()
    
    def _update_contrast_label(self, vmin, vmax):
        """Update the label showing current contrast limits."""
        self.contrast_limits_label.setText(f"{vmin:.3f} \u2013 {vmax:.3f}")
    
    def _on_xlim_changed(self, changed_ax):
        """Sync x-limits across linked axes (same row across channels, plus time\u2194CWT)."""
        if self._linking or self._axes is None:
            return
        self._linking = True
        try:
            xlim = changed_ax.get_xlim()
            # Find which row the changed axis belongs to
            changed_row = None
            for r in range(3):
                for c in range(self._n_channels):
                    if self._axes[r, c] is changed_ax:
                        changed_row = r
                        break
                if changed_row is not None:
                    break
            
            if changed_row is None:
                return
            
            # Time-domain (row 0) and CWT (row 2) share the time axis;
            # FFT (row 1) links only across channels within its own row
            if changed_row in (0, 2):
                sync_rows = [0, 2]
            else:
                sync_rows = [1]
            
            for c in range(self._n_channels):
                for row in sync_rows:
                    ax = self._axes[row, c]
                    if ax is not changed_ax:
                        ax.set_xlim(xlim)
        finally:
            self._linking = False
    
    def refresh_plots(self):
        """Refresh plots with current settings"""
        if self.last_layer and self.last_x is not None and self.last_y is not None:
            self.process_pixel(self.last_layer, self.last_x, self.last_y)
            self.propagate_wavelet_params_to_cwt_widget()
    
    def get_wavelet_tuple(self):
        """Create the wavelet tuple with current parameters"""
        params_dict = {}
        for param_name, spinbox in self.wavelet_params.items():
            params_dict[param_name] = spinbox.value()
        return (self.wavelet, params_dict)
    
    def toggle_activation(self, active):
        if active:
            if self.cid is None:
                self.cid = self.viewer.mouse_drag_callbacks.append(self.on_click)
            self.status_label.setText("Status: ACTIVE - Shift+Click on image")
            self.activate_button.setText("Deactivate Pixel Inspector")
        else:
            if self.cid is not None:
                self.viewer.mouse_drag_callbacks.remove(self.cid)
                self.cid = None
            self.status_label.setText("Status: INACTIVE - Click button to activate")
            self.activate_button.setText("Activate Pixel Inspector")

    def update_time_cursor(self, event=None):
        if self.canvas is None or not hasattr(self, "time_cursor_lines"):
            return
        
        # Skip redraws if tracking is disabled
        if not self.track_cursor_check.isChecked():
            return
        
        now=time.time()
        if now - self.last_update_time < self.update_interval:
           return  # Skip update if too soon
       
        current_t = self.viewer.dims.current_step[0]  # assumes time axis is 0
        self.current_time_index = current_t
    
        for line in self.time_cursor_lines:
            line.set_xdata([current_t])
    
        self.canvas.draw_idle()
        self.last_update_time = now
    
    def on_click(self, viewer, event):
        """Handle click events with Shift modifier"""
        if event.button != 1 or "Shift" not in event.modifiers:
            return
            
        try:
            layer = viewer.layers.selection.active
            if layer is None:
                return
                
            # Convert click position to data coordinates
            pos_data = layer.world_to_data(event.position)
            spatial_coords = pos_data[-2:]
            x, y = map(int, spatial_coords)
            
            # Store the clicked position for refreshing
            self.last_layer = layer
            self.last_x = x
            self.last_y = y
            
            # Process the pixel data
            self.process_pixel(layer, x, y)
                
        except Exception as e:
            logger.error(f"Error processing click: {e}")

    def process_pixel(self, layer, x, y):
        """Process and display data for a single pixel"""
        # Get and reshape data
        data = layer.data
        if data.ndim == 3:
            data = data[:, np.newaxis, ...]  # Add channel dimension
        T, C, X, Y = data.shape
        
        # Validate coordinates
        if not (0 <= x < X and 0 <= y < Y):
            logger.warning(f"Coordinates out of bounds: ({x}, {y})")
            return
            
        # Clear tracking lists (fixes memory leak of orphaned Line2D references)
        self.time_cursor_lines = []
        self.spectrogram_images = []
        self.cwt_magnitudes = []
        self._axes = None
        
        # Clear previous plot and toolbar
        if self.canvas:
            if self.toolbar:
                self.plot_container.layout().removeWidget(self.toolbar)
                self.toolbar.deleteLater()
                self.toolbar = None
            self.plot_container.layout().removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None

        #make new plot
        plt.style.use('dark_background')
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) #dynamic layout
        self.plot_container.layout().addWidget(self.canvas)
        
        # Add navigation toolbar for interactive zoom/pan
        self.toolbar = NavigationToolbar(self.canvas, self.plot_container)
        self.plot_container.layout().addWidget(self.toolbar)
        
        # Create subplots: 3 rows (time, FFT, CWT) x C columns
        if C > 1:
            axes = fig.subplots(3, C, squeeze=False)
        else:
            axes = fig.subplots(3, 1, squeeze=False)
        
        for c in range(C):
            # Extract and normalize time series
            ts = data[:, c, x, y].astype(np.float64)
            ts -= np.mean(ts)
            
            # Time domain plot
            logger.debug("Plotting time domain")
            ax = axes[0, c] if C > 1 else axes[0, 0]
            ax.clear()
            ax.plot(ts)
            ax.set_title(f"Channel {c} - Time Domain" if C > 1 else "Time Domain")
            ax.set_xlabel("Time")
            
            cursor = ax.axvline(self.current_time_index, color='red', linestyle='dotted')
            self.time_cursor_lines.append(cursor)

            logger.debug("Plotting freq domain")
            # Frequency domain plot
            ax = axes[1, c] if C > 1 else axes[1, 0]
            ax.clear()
            fft = np.abs(np.fft.rfft(ts))
            
            if self.do_plot_zscore:
                fft_mean=np.mean(fft)
                fft_std=np.std(fft)
                fft = (fft-fft_mean)/fft_std
            
            self.fmax=min(self.fft_gui.max_bin.value, fft.shape[0]) #adjust powerspectrum
            
            if getattr(self, 'show_nyquist', False):
                x_vals = np.linspace(0, 1.0, len(fft))[:self.fmax]
                ax.set_xlabel("Frequency (Fraction of Nyquist)")
                cursor_idx = min(self.current_time_index, self.fmax - 1)
                cursor_x = x_vals[cursor_idx] if cursor_idx >= 0 else 0
            else:
                x_vals = np.arange(self.fmax)
                ax.set_xlabel("FFT bin number")
                cursor_x = min(self.current_time_index, self.fmax)
            
            ax.plot(x_vals, fft[:self.fmax],color='#FF91A4')
            ax.set_title("Frequency Domain")
            
            cursor = ax.axvline(cursor_x, color='red', linestyle='dotted')
            self.time_cursor_lines.append(cursor)
            

            # CWT plot
            ax = axes[2, c] if C > 1 else axes[2, 0]
            ax.clear()
            
            # Get wavelet tuple with current parameters
            wavelet_tuple = self.get_wavelet_tuple()
            
            logger.debug("Plotting cwt domain")
            # Compute CWT with the selected wavelet and parameters
            Wx, _ = cwt(ts, wavelet=wavelet_tuple, nv=self.nv)

            if isinstance(Wx, torch.Tensor):
                cwt_mag = Wx.abs().cpu().numpy()
            elif isinstance(Wx, np.ndarray):
                cwt_mag = np.abs(Wx)

            if self.do_plot_zscore:
                cwt_mag_mean=np.mean(cwt_mag,axis=0)
                cwt_mag_std=np.std(cwt_mag,axis=0)
                cwt_mag=(cwt_mag-cwt_mag_mean)/cwt_mag_std
           
            logger.debug("Finalizing plots...")
            im = ax.imshow(cwt_mag, aspect='auto', origin='lower', cmap='viridis')
            self.spectrogram_images.append(im)
            self.cwt_magnitudes.append(cwt_mag)
            ax.set_title(f"CWT: {wavelet_tuple[0]}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Scale")
            
            cursor = ax.axvline(self.current_time_index, color='red', linestyle='dotted')
            self.time_cursor_lines.append(cursor)
            
            #fig.colorbar(im, ax=ax)
        
        # Store axes reference and link time/CWT x-axes across channels
        self._axes = axes
        self._n_channels = C
        for c in range(C):
            for row in range(3):  # all rows: time, FFT, CWT — linked across channels
                axes[row, c].callbacks.connect('xlim_changed', self._on_xlim_changed)
        
        # Apply current contrast settings to spectrograms
        self.apply_contrast_from_mode()
            
        fig.tight_layout()
        self.canvas.draw()

    def closeEvent(self, event):
        """Clean up when widget is closed"""
        if self.cid is not None:
            try:
                self.viewer.mouse_drag_callbacks.remove(self.cid)
            except (ValueError, RuntimeError):
                pass
        
        try:
            self.viewer.dims.events.current_step.disconnect(self.update_time_cursor)
        except (ValueError, RuntimeError):
            pass

        super().closeEvent(event)

    def populate_tree(self, root_item, data):
        """Recursively populate the tree items under the root_item."""
        if isinstance(data, dict):
            for k, v in data.items():
                child = QTreeWidgetItem(root_item)
                child.setText(0, str(k))
                self.populate_tree(child, v)
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            shape_str = "x".join(map(str, data.shape))
            dtype_str = str(data.dtype)
            type_name = "Tensor" if isinstance(data, torch.Tensor) else "Array"
            root_item.setText(1, f"{type_name} [{shape_str}] ({dtype_str})")
        elif isinstance(data, (int, float, str, list, tuple)):
            root_item.setText(1, str(data))
        else:
            root_item.setText(1, f"{type(data).__name__}: {str(data)}")

    def on_tree_item_double_clicked(self, item, column):
        """Handle double clicks on leaf tree items to add them to the canvas."""
        # Only add leaf items (items with no children)
        if item.childCount() > 0:
            return

        # Find the path of keys from root to leaf
        path = []
        curr = item
        while curr is not None:
            path.append(curr.text(0))
            curr = curr.parent()
        path.reverse()

        # Find the root item to get the data dictionary
        root_item = item
        while root_item.parent() is not None:
            root_item = root_item.parent()

        data = self.results_dict.get(id(root_item))
        if data is None:
            return

        # Traverse the dictionary using the path (skipping path[0] which is the root name)
        val = data
        for key in path[1:]:
            if isinstance(val, dict):
                if key in val:
                    val = val[key]
                elif key.isdigit() and int(key) in val:
                    val = val[int(key)]
                else:
                    logger.warning(f"Key {key} not found in dictionary data.")
                    return
            else:
                logger.warning("Encountered non-dict before leaf was reached.")
                return

        # Add to canvas if it's a tensor or array
        if isinstance(val, (torch.Tensor, np.ndarray)):
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            
            # Form layer name by joining the path elements
            layer_name = "_".join(path)
            self.viewer.add_image(val, name=layer_name)
            logger.info(f"Added {layer_name} to napari canvas.")
        else:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Not an Image/Array",
                f"Selected item value is: {val}\n(Only arrays and tensors can be added to the canvas)"
            )

    def save_result_to_zarr(self):
        """Save the selected result or the root ancestor of the selected item to a Zarr store."""
        current_item = self.results_tree.currentItem()
        if current_item is None:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Selection", "Please select a result or item from the tree.")
            return

        # Find root item
        root_item = current_item
        while root_item.parent() is not None:
            root_item = root_item.parent()

        data = self.results_dict.get(id(root_item))
        if data is None:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Data", "Could not find the data associated with the selected item.")
            return

        import pandas as pd
        if isinstance(data, pd.DataFrame):
            # Save DataFrames as Parquet instead of Zarr
            suggested_name = root_item.text(0).replace(" ", "_").replace("=", "-").replace(",", "") + ".parquet"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save DataFrame",
                suggested_name,
                "Parquet Files (*.parquet);;All Files (*)"
            )
            if not file_path:
                return
            try:
                data.to_parquet(file_path, index=False)
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.information(self, "Save Successful", f"Successfully saved DataFrame as Parquet to:\n{file_path}")
            except Exception as e:
                logger.error(f"Error writing DataFrame to Parquet: {e}")
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Save Failed", f"Failed to save Parquet:\n{str(e)}")
            return
            
        if isinstance(data, dict) and 'edges' in data and 'mean' in data:
            # It's a landscape stats dict
            suggested_name = root_item.text(0).replace(" ", "_").replace("=", "-").replace(",", "") + ".pbz2"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Landscape Stats",
                suggested_name,
                "Compressed Pickle (*.pbz2);;All Files (*)"
            )
            if not file_path:
                return
            try:
                from cellstream.pixels.utils import save_landscape
                save_landscape(data, file_path)
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.information(self, "Save Successful", f"Successfully saved Landscape Stats to:\n{file_path}")
            except Exception as e:
                logger.error(f"Error saving landscape stats: {e}")
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Save Failed", f"Failed to save Landscape Stats:\n{str(e)}")
            return

        # Suggest filename based on root item name for normal arrays
        suggested_name = root_item.text(0).replace(" ", "_").replace("=", "-").replace(",", "") + ".zarr"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Zarr Store",
            suggested_name,
            "Zarr Store (*.zarr);;All Files (*)"
        )
        if not file_path:
            return

        # Write to zarr
        try:
            try:
                from cellstream.io import write_to_zarr
                write_to_zarr(data, file_path)
            except ImportError:
                # Fallback to local implementation
                self.local_write_to_zarr(data, file_path)
            
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(self, "Save Successful", f"Successfully saved to:\n{file_path}")
        except Exception as e:
            logger.error(f"Error writing to zarr: {e}")
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save Failed", f"Failed to save to Zarr:\n{str(e)}")

    def load_result_from_zarr(self):
        """Load a saved Zarr store directory and display its tree structure in the table."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Zarr Store Directory",
            ""
        )
        if not dir_path:
            return

        try:
            # Open zarr store
            store = zarr.DirectoryStore(dir_path)
            root = zarr.open(store=store, mode="r")
            
            # Load zarr to dictionary recursively
            data = self.local_load_zarr_to_dict(root)
            
            # Generate root name based on directory name
            import os
            dir_name = os.path.basename(dir_path)
            if not dir_name:
                dir_name = os.path.basename(os.path.dirname(dir_path))
            
            root_name = f"Loaded_{dir_name}"
            
            # Create root item in the tree
            root_item = QTreeWidgetItem(self.results_tree)
            root_item.setText(0, root_name)
            
            if isinstance(data, dict):
                root_item.setText(1, f"Dict ({len(data)} keys)")
            else:
                root_item.setText(1, f"Zarr Array")
                
            self.populate_tree(root_item, data)
            self.results_dict[id(root_item)] = data
            
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(self, "Load Successful", f"Successfully loaded Zarr store from:\n{dir_path}")
            
        except Exception as e:
            logger.error(f"Error loading from zarr: {e}")
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load Failed", f"Failed to load Zarr store:\n{str(e)}")

    def delete_selected_result(self):
        """Delete the selected result from the tree and purge it from memory/viewer."""
        current_item = self.results_tree.currentItem()
        if current_item is None:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Selection", "Please select a result to delete.")
            return

        # Gather all numpy arrays we are deleting to check if they're in the viewer
        data_objects = []

        def collect_and_remove(item):
            item_id = id(item)
            if item_id in self.results_dict:
                obj = self.results_dict.pop(item_id)
                if isinstance(obj, (np.ndarray, torch.Tensor)):
                    if isinstance(obj, torch.Tensor):
                        data_objects.append(obj.cpu().numpy())
                    else:
                        data_objects.append(obj)
                del obj

            for i in range(item.childCount()):
                collect_and_remove(item.child(i))

        collect_and_remove(current_item)

        # Remove layers from napari viewer that share memory with deleted objects
        layers_to_remove = []
        for layer in self.viewer.layers:
            if not isinstance(layer.data, np.ndarray):
                continue
                
            for obj in data_objects:
                # Check if layer data is the object or shares the same memory base
                if layer.data is obj or (layer.data.base is not None and layer.data.base is obj):
                    layers_to_remove.append(layer)
                    break
                elif obj.base is not None and layer.data.base is obj.base:
                    layers_to_remove.append(layer)
                    break

        for layer in layers_to_remove:
            self.viewer.layers.remove(layer)

        # Remove from tree
        parent = current_item.parent()
        if parent is not None:
            parent.removeChild(current_item)
        else:
            index = self.results_tree.indexOfTopLevelItem(current_item)
            self.results_tree.takeTopLevelItem(index)

        # Force garbage collection to free memory immediately
        import gc
        gc.collect()

    def local_write_to_zarr(self, data, path, chunks=True, compressor="default"):
        """Fallback local implementation of write_to_zarr."""
        if compressor == "default":
            compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)

        if isinstance(data, (torch.Tensor, np.ndarray)):
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            z = zarr.open(path, mode="w", shape=data.shape, dtype=data.dtype, 
                          chunks=chunks, compressor=compressor)
            z[:] = data
        elif isinstance(data, dict):
            store = zarr.DirectoryStore(path)
            root = zarr.group(store=store, overwrite=True)
            self.local_write_dict_to_zarr_group(root, data, chunks=chunks, compressor=compressor)
        else:
            raise TypeError(f"Unsupported data type for write_to_zarr: {type(data)}")

    def local_write_dict_to_zarr_group(self, group, d, chunks=True, compressor=None):
        """Fallback local implementation of write dict to zarr group."""
        for k, v in d.items():
            key = str(k)
            if isinstance(v, dict):
                subgroup = group.create_group(key)
                self.local_write_dict_to_zarr_group(subgroup, v, chunks=chunks, compressor=compressor)
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                group.array(key, v, chunks=chunks, compressor=compressor)
            elif isinstance(v, (int, float, str, list, tuple)):
                group.attrs[key] = v
            else:
                try:
                    arr = np.array(v)
                    group.array(key, arr, chunks=chunks, compressor=compressor)
                except Exception:
                    print(f"Warning: Could not save key {key} of type {type(v)} to Zarr.")

    def local_load_zarr_to_dict(self, zarr_item):
        """Recursively loads a Zarr group or array into standard nested Python dictionaries/arrays."""
        if hasattr(zarr_item, "items"):
            d = {}
            for name, child in zarr_item.items():
                # Protect against pulling virtual keys if the item is a custom wrapper
                if name == "_attrs":
                    continue
                d[name] = self.local_load_zarr_to_dict(child)
            
            # Group all attributes into a single nested dictionary
            if hasattr(zarr_item, "attrs") and len(zarr_item.attrs) > 0:
                d["_attrs"] = {}
                for attr_key, attr_val in zarr_item.attrs.items():
                    d["_attrs"][attr_key] = attr_val
                    
            return d
            
        elif hasattr(zarr_item, "shape") and hasattr(zarr_item, "dtype"):
            # Ensure we wrap the read safely across versions
            return np.asarray(zarr_item[:])
        else:
            return zarr_item

