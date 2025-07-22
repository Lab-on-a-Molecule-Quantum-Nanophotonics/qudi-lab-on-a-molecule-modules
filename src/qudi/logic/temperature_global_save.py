from PySide2 import QtCore
import time
from datetime import datetime
from qudi.core.connector import Connector
from qudi.util.datastorage import DataStorageBase
from qudi.core.configoption import ConfigOption
from qudi.core.module import LogicBase

from qudi.interface.data_instream_interface import StreamingMode, SampleTiming

class ProcessValueMetadataSampling(LogicBase):
    _cryostation = Connector(name="process", interface="ProcessValueInterface")

    refresh_time = ConfigOption(name='refresh_time_ms', default=500)
    channels = ConfigOption(name="channels", default=[])
    metadata_name = ConfigOption(name="metadata_name")

    sigStart = QtCore.Signal()
    sigStop = QtCore.Signal()
    sigTimer = QtCore.Signal(int)
    _threaded = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)

    def update(self):
        metadata = {}
        cryostation = self._cryostation()
        units = cryostation.constraints.channel_units
        for channel in self.channels:
            metadata[channel] = dict(unit=units[channel], value=cryostation.get_process_value(channel))
        metadata["measurement date"] = datetime.now()
        DataStorageBase.add_global_metadata(self.metadata_name, metadata, overwrite=True)
        self.start()

    def on_activate(self):
        self._timer.timeout.connect(self.update, QtCore.Qt.QueuedConnection)
        self.sigTimer.connect(self._timer.start, QtCore.Qt.QueuedConnection)
        self.sigStop.connect(self._timer.stop, QtCore.Qt.QueuedConnection)
        self._timer.setInterval(self.refresh_time)
        self.start()

    def start(self):
        self.sigTimer.emit(self.refresh_time)

    def stop(self):
        self._timer.timeout.disconnect(self.update)
        self.sigStop.emit()
        self.sigStop.disconnect()
        self.sigTimer.disconnect()

    def on_deactivate(self):
        self.stop()

