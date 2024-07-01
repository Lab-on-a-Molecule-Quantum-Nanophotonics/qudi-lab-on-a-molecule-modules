from PySide2 import QtCore
from PySide2 import QtWidgets

class CateyeReportWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        main_layout = QtWidgets.QFormLayout()
        self.setLayout(main_layout)
        self.grating = QtWidgets.QLabel("")
        self.photodiode = QtWidgets.QLabel("")
        self.frequency = QtWidgets.QLabel("")
        self.state = QtWidgets.QLabel("")
        main_layout.addRow("Grating :", self.grating)
        main_layout.addRow("Photodiode :", self.photodiode)
        main_layout.addRow("Frequency :", self.frequency)
        main_layout.addRow("Scanner state :" ,self.state)
    def update(self, report):
        for k in ("grating", "photodiode", "frequency"):
            getattr(self, k).setText(report["values"][k] + " " + report["units"][k])
        self.state.setText(report["state"])
