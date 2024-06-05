from PySide2 import QtCore, QtWidgets
import qudi.util.uic as uic
from qudi.core.connector import Connector
from qudi.core.module import GuiBase

class LaserScanningSettingsMainWindow(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Laser Scanning")
        self.layout = QtWidgets.QHBoxLayout()
        self.scanGroupBox = QtWidgets.QGroupBox("Scanner Settings")
        self.manualGroupBox = QtWidgets.QGroupBox("Manual control")
        self.scanFormLayout = QtWidgets.QFormLayout()
        self.manualLayout = QtWidgets.QFormLayout()
        self.scanMinSpinBox = QtWidgets.QDoubleSpinBox()
        self.scanMinSpinBox.setDecimals(3)
        self.scanMaxSpinBox = QtWidgets.QDoubleSpinBox()
        self.scanMaxSpinBox.setDecimals(3)
        self.sampleSizeSpinBox = QtWidgets.QSpinBox()
        self.exposureSpinBox = QtWidgets.QDoubleSpinBox()
        self.exposureSpinBox.setSuffix(" s")
        self.resetWavemeterButton = QtWidgets.QPushButton("Reset wavemeter")
        self.calibrateButton = QtWidgets.QPushButton("Calibrate grating")
        self.manualTensionSpinBox = QtWidgets.QDoubleSpinBox()
        self.manualTensionSpinBox.setSuffix(" V")
        self.manualTensionSpinBox.setDecimals(3)
        self.centralWavelengthComboBox = QtWidgets.QComboBox()
        self.scanFormLayout.addRow("Minimum piezo tension", self.scanMinSpinBox)
        self.scanFormLayout.addRow("Maximum piezo tension", self.scanMaxSpinBox)
        self.scanFormLayout.addRow("Number of points", self.sampleSizeSpinBox)
        self.scanFormLayout.addRow("Exposure", self.exposureSpinBox)
        self.manualLayout.addRow("", self.resetWavemeterButton)
        self.manualLayout.addRow("", self.calibrateButton)
        self.manualLayout.addRow("Central wavelength", self.centralWavelengthComboBox)
        self.manualLayout.addRow("Piezo tension", self.manualTensionSpinBox)
        self.manualGroupBox.setLayout(self.manualLayout)
        self.scanGroupBox.setLayout(self.scanFormLayout)
        self.layout.addWidget(self.scanGroupBox)
        self.layout.addWidget(self.manualGroupBox)
        self.setLayout(self.layout)

class LaserScanningSettingsGui(GuiBase):
    _scanning_settings_logic = Connector(interface="NILaserScanningSettingsLogic")
        
    def on_activate(self):
        self._mw = LaserScanningSettingsMainWindow()
        logic = self._scanning_settings_logic()
        scan_range = logic.scan_constraints()
        self._mw.scanMinSpinBox.setMinimum(scan_range[0])
        self._mw.scanMinSpinBox.setMaximum(scan_range[1])
        self._mw.scanMaxSpinBox.setMinimum(scan_range[0])
        self._mw.scanMaxSpinBox.setMaximum(scan_range[1])
        self._mw.manualTensionSpinBox.setMinimum(scan_range[0])
        self._mw.manualTensionSpinBox.setMaximum(scan_range[1])
        self._mw.manualTensionSpinBox.setSingleStep(0.001)
        available_wavelengths = logic.get_calibration()
        for w in available_wavelengths:
            self._mw.centralWavelengthComboBox.addItem(str(w))
        
        sample_size_constraints = logic.sample_size_constraints()
        self._mw.sampleSizeSpinBox.setMinimum(sample_size_constraints[0])
        self._mw.sampleSizeSpinBox.setMaximum(sample_size_constraints[1])
        
        self._mw.exposureSpinBox.setMinimum(0.001)
        self._mw.exposureSpinBox.setMaximum(10)
        
        self._mw.scanMinSpinBox.setValue(logic.scan_min)
        self._mw.scanMaxSpinBox.setValue(logic.scan_max)
        self._mw.sampleSizeSpinBox.setValue(logic.sample_size)
        self._mw.manualTensionSpinBox.setValue(logic.get_tension())
        self._mw.exposureSpinBox.setValue(logic.get_exposure())
        
        self._mw.scanMinSpinBox.valueChanged.connect(logic.set_scan_min)
        self._mw.scanMaxSpinBox.valueChanged.connect(logic.set_scan_max)
        self._mw.manualTensionSpinBox.valueChanged.connect(logic.set_tension)
        self._mw.sampleSizeSpinBox.valueChanged.connect(logic.set_sample_size)
        self._mw.exposureSpinBox.valueChanged.connect(logic.set_exposure)
        self._mw.centralWavelengthComboBox.currentTextChanged.connect(self.send_new_wavelength)
        self._mw.resetWavemeterButton.clicked.connect(logic.set_continuous_reading)
        self._mw.calibrateButton.clicked.connect(logic.calibrate)
        logic.calibrated.connect(self.update_available_central_wavelength)
        self._mw.show()
        
        
    def update_available_central_wavelength(self, wavelengths):
        self._mw.centralWavelengthComboBox.clear()
        for w in wavelengths:
            self._mw.centralWavelengthComboBox.addItem(str(w))
            
    def send_new_wavelength(self, wavelength_str):
        self._scanning_settings_logic().set_center_wavelength(int(wavelength_str))
            
    def on_deactivate(self):
        self._mw.scanMinSpinBox.valueChanged.disconnect()
        self._mw.scanMaxSpinBox.valueChanged.disconnect()
        self._mw.sampleSizeSpinBox.valueChanged.disconnect()
        self._mw.close()
    
    def show(self):
        """Make window visible and put it above all other windows. """
        self._mw.show()
        self._mw.activateWindow()
        self._mw.raise_()
