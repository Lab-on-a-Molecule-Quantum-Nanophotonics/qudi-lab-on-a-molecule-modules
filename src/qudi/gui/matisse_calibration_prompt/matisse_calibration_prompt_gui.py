__all__ = ['CalibrationGui']

from PySide2 import QtCore, QtWidgets
from PySide2 import QtGui
from typing import Union, Dict, Tuple

from qudi.core.statusvariable import StatusVar
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.module import GuiBase
from qudi.util.widgets.scientific_spinbox import ScienDSpinBox

class CalibrationMainWindow(QtWidgets.QMainWindow):
    """ Main Window for the Calibration Gui module """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('qudi: Matisse Calibration factor')
        main_layout = QtWidgets.QFormLayout()
        self.calibration_spinbox = ScienDSpinBox()
        self.calibration_spinbox.setSuffix("Hz/unit scan")
        main_layout.addRow("Calibration", self.calibration_spinbox)
        widget = QtWidgets.QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

class CalibrationGui(GuiBase):
    matisse_hw = Connector(interface="MatisseCommander")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw = None
        self.stop_update = False
    def on_activate(self):
        """ Initialisation of the GUI """
        self._mw = CalibrationMainWindow()
        self.stop_update = False
        self.update_display()
        self.show()
    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.stop_update = True
        self._mw.close()
    def show(self):
        """Make window visible and put it above all other windows.
        """
        self._mw.show()
        self._mw.activateWindow()
        self._mw.raise_()
    def update_hw(self, v):
        self.matisse_hw().set_setpoint("conversion factor", v)
    def update_display(self):
        if self._mw.calibration_spinbox.hasFocus():
            return
        self._mw.calibration_spinbox.blockSignals(True)
        self._mw.calibration_spinbox.setValue(self.matisse_hw().get_setpoint("conversion factor"))
        self._mw.calibration_spinbox.blockSignals(False)
        if self.stop_update:
            QtCore.QTimer.singleShot(1000, self.update_display)