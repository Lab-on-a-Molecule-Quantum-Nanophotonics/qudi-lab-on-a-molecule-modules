import os

import qudi.util.uic as uic
from PySide6 import QtWidgets


class MoleculeDetectionSettings(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'settings.ui')
        super().__init__(*args, **kwargs)
        uic.loadUi(ui_file, self)

