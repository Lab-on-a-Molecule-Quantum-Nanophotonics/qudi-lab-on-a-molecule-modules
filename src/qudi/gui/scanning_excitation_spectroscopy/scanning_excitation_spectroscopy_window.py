
__all__ = ['ScanningExcitationSpectroscopyMainWindow']

import os
import importlib

import numpy as np

from PySide2 import QtCore
from PySide2 import QtWidgets
from PySide2 import QtGui

from qudi.util.paths import get_artwork_dir
from qudi.util.widgets.advanced_dockwidget import AdvancedDockWidget

try:
    importlib.reload(cateye_settings)
except NameError:
    import qudi.gui.scanning_excitation_spectroscopy.cateye_settings_widget as cateye_settings
try:
    importlib.reload(cateye_report)
except NameError:
    import qudi.gui.scanning_excitation_spectroscopy.cateye_report_widget as cateye_report
try:
    importlib.reload(mode_scan)
except NameError:
    import qudi.gui.scanning_excitation_spectroscopy.mode_scan_widget as mode_scan

class ScanningExcitationSpectroscopyMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Scanning Excitation Spectroscopy')
        self.setDockNestingEnabled(True)
        self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)
        icon_path = os.path.join(get_artwork_dir(), 'icons')
        self._dockwidgets = [
            ("scan_settings", "Scan Settings", cateye_settings.CateyeScanSettingsWidget),
            ("laser_report", "Laser report", cateye_report.CateyeReportWidget),
            ("mode_scan", "Latest mode scan", mode_scan.ModeScanWidget),

        ]
        for name,title,cls in self._dockwidgets:
            self._add_dockwidget(name,title,cls)
        # Create QActions
        close_icon = QtGui.QIcon(os.path.join(icon_path, 'application-exit'))
        self.action_close = QtWidgets.QAction(icon=close_icon, text='Close Window', parent=self)

        restore_icon = QtGui.QIcon(os.path.join(icon_path, 'view-refresh'))
        self.action_measurement_mode = QtWidgets.QAction(icon=restore_icon, text='Measurement display', parent=self)
        self.action_measurement_mode.setToolTip('Show only widgets used in measurements.')

        setting_icon = QtGui.QIcon(os.path.join(icon_path, 'utilities-terminal'))
        self.action_calibration_mode = QtWidgets.QAction(icon=setting_icon,
                                                              text='Calibration display',
                                                              parent=self)
        self.action_calibration_mode.setToolTip('Show only widgets used in calibration.')

        fit_settings_icon = QtGui.QIcon(os.path.join(icon_path, 'configure'))
        self.action_show_fit_settings = QtWidgets.QAction(icon=fit_settings_icon,
                                                          text='Show Fit Settings',
                                                          parent=self)
        self.action_show_fit_settings.setToolTip('Show the Fit Settings.')
        save_spec_icon = QtGui.QIcon(os.path.join(icon_path, 'document-save'))
        self.action_save_spectrum = QtWidgets.QAction(icon=save_spec_icon,
                                                      text='Save Spectrum',
                                                      parent=self)
        self.action_save_spectrum.setToolTip('Save the currently shown spectrum.')
        # Create the menu bar
        menu_bar = QtWidgets.QMenuBar()
        menu = menu_bar.addMenu('File')
        menu.addAction(self.action_save_spectrum)
        menu.addSeparator()
        menu.addAction(self.action_close)
        menu = menu_bar.addMenu('View')
        menu.addAction(self.action_show_fit_settings)
        menu.addSeparator()
        for name,_,_ in self._dockwidgets:
            menu.addAction(getattr(self, f"action_show_{name}"))
        menu.addSeparator()
        menu.addAction(self.action_measurement_mode)
        menu.addAction(self.action_calibration_mode)
        self.setMenuBar(menu_bar)
        # connecting up the internal signals
        self.action_close.triggered.connect(self.close)
        self.action_measurement_mode.triggered.connect(self.set_measurement_mode)
        self.action_calibration_mode.triggered.connect(self.set_calibration_mode)
        
        self.reset_docks()
        
        self.show()

    def _toggle_dock_visibility(self, visible, hidden):
        for name in visible:
            getattr(self, f"action_show_{name}").setChecked(True)
        for name in hidden:
            getattr(self, f"action_show_{name}").setChecked(False)
    def set_measurement_mode(self):
        visible = ["laser_report"]
        hidden = ["scan_settings", "mode_scan"]
        self._toggle_dock_visibility(visible, hidden)
    def set_calibration_mode(self):
        visible = ["laser_report", "scan_settings", "mode_scan"]
        hidden = []
        self._toggle_dock_visibility(visible, hidden)
    def reset_docks(self):
        for name,_,_ in self._dockwidgets:
            getattr(self, f"{name}_dockwidget").setFloating(False)
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, getattr(self, f"{name}_dockwidget"))
    def _add_dockwidget(self, name, title, cls):
        setattr(self, name + "_widget", cls())
        setattr(self, name + "_dockwidget", AdvancedDockWidget(title, parent=self))
        getattr(self, name + "_dockwidget").setWidget(getattr(self, name + "_widget"))
        setattr(self, "action_show_" + name, QtWidgets.QAction(text=f'Show {title}', parent=self))
        getattr(self, "action_show_" + name).setToolTip(f'Show/Hide {title} dock.')
        getattr(self, "action_show_" + name).setCheckable(True)
        getattr(self, "action_show_" + name).setChecked(True)
        getattr(self, "action_show_" + name).triggered[bool].connect(getattr(self, name + "_dockwidget").setVisible)
        getattr(self, name + "_dockwidget").sigClosed.connect(lambda: getattr(self, "action_show_" + name).setChecked(False))

if __name__ == '__main__':
    import sys
    import os
    from qudi.util.paths import get_artwork_dir
    import qudi.core.application
    import qudi.logic.cateye_laser_logic as cateye_logic

    scan = cateye_logic.ScanConfiguration(
        grating = np.array([4400, 5000, 5600]),
        span = np.array([1.0,1.0,1.0]),
        offset = np.array([0.0, 0.0, 0.0]),
        frequency = np.array([5.0, 5.0, 5.0]),
        bias = np.array([0.0, 0.0, 0.0]),
        sample_time = np.array([10e-3, 10e-3, 10e-3]),
        repeat = np.array([1, 1, 1]),
        calibration = False,
    )

    stylesheet_path = os.path.join(get_artwork_dir(), 'styles', 'qdark.qss')
    with open(stylesheet_path, 'r') as file:
        stylesheet = file.read()
    path = os.path.join(os.path.dirname(stylesheet_path), 'qdark').replace('\\', '/')
    stylesheet = stylesheet.replace('{qdark}', path)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    mw = ScanningExcitationSpectroscopyMainWindow()
    mw.scan_settings_widget.set_scan_list([f"{scan.date_created}: {scan.name}"])
    mw.scan_settings_widget.set_current_scan(0, scan)
    mw.show()
    sys.exit(app.exec_())
