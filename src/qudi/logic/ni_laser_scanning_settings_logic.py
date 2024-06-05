import datetime

from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.core.module import LogicBase
from qudi.interface.simple_scanner_interface import SimpleScannerSettings, SimpleScannerConstraints

from qtpy import QtCore
import numpy as np

class NILaserScanningSettingsLogic(LogicBase):
    scan_min = StatusVar("scan_min", -10)
    scan_max = StatusVar("scan_max", 10)
    sample_size = StatusVar("sample_size", 1000)
    exposure = StatusVar("exposure", 0.1)

    _ni_laser_spectrometer_interfuse = Connector(name="ni_laser_spectrometer_interfuse", interface="SimpleScannerInterface")
    _manual_control = Connector(name="manual_control", interface="ProcessControlInterface")
    _plot_logic = Connector(name='plot', interface='QDPlotLogic')
    
    _sigCalibrate = QtCore.Signal()
    
    calibrated = QtCore.Signal(np.ndarray)

    def on_activate(self):
        self.ni_laser_spectrometer_interfuse = self._ni_laser_spectrometer_interfuse()
        self.manual_control = self._manual_control()
        self.update_settings()
        self._sigCalibrate.connect(self.ni_laser_spectrometer_interfuse.calibrate, QtCore.Qt.QueuedConnection)
        self.ni_laser_spectrometer_interfuse.sigCalibrationChanged.connect(self._calibrated, QtCore.Qt.QueuedConnection)
    def on_deactivate(self):
        self._sigCalibrate.disconnect()
    def set_tension(self, value):
        self.manual_control.set_setpoint("tension", value)
    def get_tension(self):
        return self.manual_control.get_setpoint("tension")
    def set_center_wavelength(self, value):
        self.manual_control.set_setpoint("grating wavelength", value)
    def get_center_wavelength(self):
        return self.manual_control.get_setpoint("grating wavelength")
    def set_scan_min(self, value):
        self.scan_min = value
        self.update_settings()
    def set_scan_max(self, value):
        self.scan_max = value
        self.update_settings()
    def set_sample_size(self, value):
        self.sample_size = value
        self.update_settings()
    def get_exposure(self):
        return self.exposure 
    def set_exposure(self, value):
        self.exposure = value
        self.ni_laser_spectrometer_interfuse.exposure_time = value
    def scan_constraints(self):
        return (
            self.ni_laser_spectrometer_interfuse.constraints.scan_min, 
            self.ni_laser_spectrometer_interfuse.constraints.scan_max
        )
    def sample_size_constraints(self):
        return (
            self.ni_laser_spectrometer_interfuse.constraints.sample_rate_min, 
            self.ni_laser_spectrometer_interfuse.constraints.sample_rate_max
        )
    def update_settings(self):
        settings = SimpleScannerSettings(
            first_point = self.scan_min,
            last_point = self.scan_max,
            sample_rate = self.ni_laser_spectrometer_interfuse._sample_rate,
            number_of_samples = self.sample_size,
        )
        return self.ni_laser_spectrometer_interfuse.set_settings(settings)
        
    def set_continuous_reading(self):
        self.ni_laser_spectrometer_interfuse.reset_continuous()
        
    def calibrate(self):
        self._sigCalibrate.emit()
    def _calibrated(self):
        position, wavelength = self.ni_laser_spectrometer_interfuse.get_calibration()
        self._plot_logic().add_plot()
        plotindex = self._plot_logic().plot_count - 1
        self._plot_logic().set_data(plotindex, (position, wavelength), name="Grating calibration %s" % datetime.datetime.now())
        self._plot_logic().set_labels(plotindex, x="Position number", y="Wavelength")
        self._plot_logic().set_units(plotindex, y="nm")
        self.calibrated.emit(wavelength)
        
    def get_calibration(self):
        _, wavelength = self.ni_laser_spectrometer_interfuse.get_calibration()
        return wavelength
        
