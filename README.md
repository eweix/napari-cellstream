# napari-cellstream

A napari spectral analyzer for pixel-level cellstream diagnostics. This plugin provides tools for analyzing time-series imagery using Fast Fourier Transforms (FFT) and Continuous Wavelet Transforms (CWT), specifically designed for use with the [cellstream](https://github.com/CoyleLab-UW-Madison/cellstream) image processing toolbox.

---

## Features

### 1. Pixel Inspector
- **Interactive Analysis:** Use `Shift + Click` on any image layer to inspect the temporal behavior of a single pixel.
- **Multi-Domain View:** Instantly visualize:
  - **Time Domain:** Raw intensity over time.
  - **Frequency Domain (FFT):** Power spectrum for identifying dominant frequencies.
  - **Scale-Time Domain (CWT):** Spectrogram showing how frequency content evolves over time.
- **Customizable Wavelets:** Support for various wavelet families (GMW, Morlet, Bump, etc.) with adjustable parameters.

### 2. Spectral Feature Generation
- **FFT Features:** Generate full-image feature maps for amplitude, phase, and Z-scores across specified frequency bins.
- **CWT Features:** Perform block-wise CWT processing to extract features across multiple scales.
- **GPU Acceleration:** Leverage GPU support (CUDA or MPS) for high-performance spectral decomposition.

### 3. Visualization Tools
- **False-Color Spectrum:** Visualize spectral axes (time, frequency, or scale) using color-coded projections.
- **Downsampling:** Efficiently downsample large datasets in time or space for faster processing and visualization.

### 4. Zarr-Based Results I/O 
- **Efficient Storage:** Manage analysis results efficiently using Zarr stores.
- **Chunked Processing:** Support for reading and writing large multidimensional features without exhausting memory.

---

## Installation

While `napari-cellstream` can be installed via pip:

```bash
pip install git+https://github.com/CoyleLab-UW-Madison/napari-cellstream.git
```

**Recommended Setup:** To ensure all GPU dependencies (like PyTorch and `torch-scatter`) are correctly configured, we highly recommend setting up your environment using the `environment.yml` provided in the core [`cellstream`](https://github.com/CoyleLab-UW-Madison/cellstream) repository.

---

## Usage

1. **Launch napari:**
   ```bash
   napari
   ```
2. **Open the plugin:**
   Go to `Plugins` -> `napari-cellstream: Spectral Viewer`.
3. **Pixel Inspector:**
   - Click **Activate Pixel Inspector**.
   - Select an image layer with a time dimension (T, C, X, Y or T, X, Y).
   - `Shift + Click` on the image to view the spectral analysis in the sidebar.
4. **Generate Features:**
   - Use the **Generate FFT features** or **Generate CWT features** sections to create new image layers based on spectral analysis.
5. Zarr-based saving/loading of results
   - Use the zarr-store widget section to manage results

---

## Dependencies

This plugin is intended to be used with the [cellstream](https://github.com/CoyleLab-UW-Madison/cellstream) package for core image processing utilities.

---


