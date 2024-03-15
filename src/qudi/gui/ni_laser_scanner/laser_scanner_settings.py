from PySide2 import QtCore, QtWidgets
import qudi.util.uic as uic
from qudi.core.connector import Connector
from qudi.core.module import GuiBase

class LaserScanningSettingsMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Laser Scanning Settings")
        self.formLayout = QtWidgets.QFormLayout()
        self.scanMinLabel = QtWidgets.QLabel("Minimum piezo tension")
        self.scanMaxLabel = QtWidgets.QLabel("Maximum piezo tension")
        self.sampleSizeLabel = QtWidgets.QLabel("Number of points")
        self.scanMinSpinBox = QtWidgets.QDoubleSpinBox()
        self.scanMaxSpinBox = QtWidgets.QDoubleSpinBox()
        self.sampleSizeSpinBox = QtWidgets.QSpinBox()
        self.formLayout.addRow(self.scanMinLabel, self.scanMinSpinBox)
        self.formLayout.addRow(self.scanMaxLabel, self.scanMaxSpinBox)
        self.formLayout.addRow(self.sampleSizeLabel, self.sampleSizeSpinBox)
        self.setLayout(self.formLayout)

class LaserScanningSettingsGui(GuiBase):
    _scanning_settings_logic = Connector()
        
    def on_activate(self):
        self._mw = LaserScanningSettingsGui()
        self._mw.scanMinSpinBox.setDecimals(3)
        self._mw.scanMinSpinBox.setMinimum(-10)
        self._mw.scanMinSpinBox.setMaximum(10)
        self._mw.scanMaxSpinBox.setDecimals(3)
        self._mw.scanMaxSpinBox.setMinimum(-10)
        self._mw.scanMaxSpinBox.setMaximum(10)
        self._mw.sampleSizeSpinBox.setMinimum(1)
        self._mw.sampleSizeSpinBox.setMaximum(2**16)
        self._mw.scanMinSpinBox.setValue(self._scanning_settings_logic.scan_min)
        self._mw.scanMaxSpinBox.setValue(self._scanning_settings_logic.scan_max)
        self._mw.sampleSizeSpinBox.setValue(self._scanning_settings_logic.sample_size)
        self._mw.scanMinSpinBox.valueChanged.connect(self._scanning_settings_logic.sigChangeScanMin)
        self._mw.scanMaxSpinBox.valueChanged.connect(self._scanning_settings_logic.sigChangeScanMax)
        self._mw.sampleSizeSpinBox.valueChanged.connect(self._scanning_settings_logic.sigChangeSampleSize)

    def on_deactivate(self):
        self._mw.scanMinSpinBox.valueChanged.disconnect()
        self._mw.scanMaxSpinBox.valueChanged.disconnect()
        self._mw.sampleSizeSpinBox.valueChanged.disconnect()
