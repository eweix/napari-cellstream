# napari_cellstream/_plugin.py

from qtpy.QtWidgets import QMessageBox
from napari import current_viewer
import os
import logging

logger = logging.getLogger(__name__)

_USE_GPU = None

def make_spectral_widget(viewer=None):
    global _USE_GPU

    #handle viewer injection
    logger.info("make_spectral_widget called. viewer passed? %s", viewer is not None)
    if viewer is None:
        viewer = current_viewer()
        if viewer is None:
            raise RuntimeError("No active viewer found via napari.current_viewer()")

    #set ssq_gpu environment if not already set
    if _USE_GPU is None:
        msg = QMessageBox()
        msg.setWindowTitle("Enable GPU Support?")
        msg.setText("Would you like to enable GPU acceleration (if available)?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        ret = msg.exec_()

        if ret == QMessageBox.Yes:
            _USE_GPU = True
        else:
            _USE_GPU = False

    if _USE_GPU:
        os.environ["SSQ_GPU"] = "1"
    else:
        os.environ["SSQ_GPU"] = "0"

    # Show splash screen while loading the heavy imports
    from qtpy.QtWidgets import QSplashScreen, QApplication
    from qtpy.QtGui import QPixmap
    from qtpy.QtCore import Qt
    
    splash = None
    splash_path = os.path.join(os.path.dirname(__file__), "splash.png")
    if os.path.exists(splash_path):
        pixmap = QPixmap(splash_path)
        
        # Scale to 50% screen width
        app = QApplication.instance()
        if app:
            screen_geom = app.primaryScreen().availableGeometry()
            target_width = int(screen_geom.width() * 0.5)
            # Only scale if the screen width makes sense
            if target_width > 100:
                pixmap = pixmap.scaledToWidth(target_width, Qt.SmoothTransformation)
        
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.show()
        splash.repaint()
        
        if app:
            app.processEvents()
        else:
            QApplication.processEvents()

    # Apply torch tensor patch for napari
    try:
        from cellstream.viz import patch_napari_for_torch
        patch_napari_for_torch()
        logger.info("Applied cellstream.viz.patch_napari_for_torch()")
    except ImportError:
        logger.warning("Could not import cellstream.viz.patch_napari_for_torch()")
    except Exception as e:
        logger.error(f"Failed to apply torch patch: {e}")

    # Importing widget until *after* setting the ssq_gpu env var
    from .spectral_analyzer import SpectralWidget
    widget = SpectralWidget(viewer, use_gpu=_USE_GPU)
    
    if splash:
        splash.finish(widget)
        
    return widget
