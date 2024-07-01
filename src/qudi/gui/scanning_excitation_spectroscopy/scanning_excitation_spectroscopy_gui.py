from PySide2 import QtCore

from qudi.core.module import GuiBase
from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
__all__ = ['ScanningExcitationSpectroscopyGui']

import importlib
from time import perf_counter
from PySide2 import QtCore

from qudi.core.module import GuiBase
from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.core.configoption import ConfigOption
from qudi.util.widgets.fitting import FitConfigurationDialog, FitWidget
# Ensure specialized QMainWindow widget is reloaded as well when reloading this module
try:
    importlib.reload(window_module)
except NameError:
    import qudi.gui.scanning_excitation_spectroscopy.scanning_excitation_spectroscopy_window as window_module

class ScanningExcitationSpectroscopyGui(GuiBase):
    _scanning_logic = Connector(name="scanning_logic", interface="CateyeLaserLogic")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw = None
        self._fsd = None

    def on_activate(self):
        self._mw = window_module.ScanningExcitationSpectroscopyMainWindow()
        # logic -> window
        scanning_logic = self._scanning_logic()
        scanning_logic.sigNewModeScanAvailable.connect(self.display_latest_mode_scan)
        scanning_logic.sigNewModeHopsAvailable.connect(self.display_latest_mode_hops)
        scanning_logic.sigPositionUpdated.connect(self._mw.laser_report_widget.update_data)
        scanning_logic.sigScanningUpdated.connect(self.set_scanning_state)
        scanning_logic.sigCurrentScanUpdated.connect(self.update_current_scan)
        scanning_logic.sigCurrentScanUpdated.connect(self.update_scan_list)
        # window -> logic
        # scan management
        self._mw.scan_settings_widget.sigNewScan.connect(scanning_logic.create_new_scan)
        self._mw.scan_settings_widget.sigDelScan.connect(scanning_logic.delete_current_scan)
        self._mw.scan_settings_widget.sigStartScan.connect(scanning_logic.start_scan)
        self._mw.scan_settings_widget.sigSetCurrentScan.connect(scanning_logic.set_current_scan)
        self._mw.scan_settings_widget.sigNewStep.connect(scanning_logic.create_new_step)
        self._mw.scan_settings_widget.sigDelStep.connect(scanning_logic.delete_step)
        self._mw.scan_settings_widget.sigMoveStepUp.connect(scanning_logic.move_step_up)
        self._mw.scan_settings_widget.sigMoveStepDown.connect(scanning_logic.move_step_down)
        self._mw.scan_settings_widget.sigSetStepGrating.connect(scanning_logic.set_step_grating)
        self._mw.scan_settings_widget.sigSetStepSpan.connect(scanning_logic.set_step_span)
        self._mw.scan_settings_widget.sigSetStepOffset.connect(scanning_logic.set_step_offset)
        self._mw.scan_settings_widget.sigSetStepFrequency.connect(scanning_logic.set_step_frequency)
        self._mw.scan_settings_widget.sigSetStepBias.connect(scanning_logic.set_step_bias)
        self._mw.scan_settings_widget.sigSetStepSampleTime.connect(scanning_logic.set_step_sample_time)
        self._mw.scan_settings_widget.sigSetStepRepeat.connect(scanning_logic.set_step_repeat)

    def display_latest_mode_scan(self):
        self._mw.mode_scan_widget.update_data(self._scanning_logic().last_mode_scan)

    def display_latest_mode_hops(self):
        self._mw.mode_scan_widget.update_mode_hops(self._scanning_logic().mode_hops)

    def set_scanning_state(self, bool):
        pass

    def update_current_scan(self):
        scan = self._scanning_logic().current_scan
        scan_index = self._scanning_logic().current_scan_index
        self._mw.scan_settings_widget.set_current_scan(scan_index, scan)

    def update_scan_list(self):
        names = self._scanning_logic().scan_history_names
        self._mw.scan_settings_widget.set_scan_list(names)

