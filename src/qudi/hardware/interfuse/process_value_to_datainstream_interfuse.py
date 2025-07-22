from PySide2.QtCore import QThread, QMutex, QWaitCondition
import numpy as np
import time
from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.interface.data_instream_interface import DataInStreamInterface, StreamingMode, SampleTiming, DataInStreamConstraints
from qudi.util.constraints import ScalarConstraint


class ProcessDataStream(DataInStreamInterface):
    _process = Connector(name="process", interface="ProcessValueInterface")
    _threaded = True
    _sample_rate = StatusVar(name="sample_rate", default=10)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._active_channels = []
        self._streaming_mode = StreamingMode.CONTINUOUS
        self._sample_rate = 0
        self._buffer = []
        self._timestamps = []
        self._mutex = QMutex()
        self._wait_condition = QWaitCondition()
        self._running = False
        self._start_time = None
        self._thread = None

    def on_activate(self):
        process = self._process()
        self._thread = self.AcquisitionThread(self)
        if not process:
            raise RuntimeError("ProcessValueInterface is not connected.")
        
        self._active_channels = list(process.constraints.process_channels)
        for channel in self._active_channels:
            process.set_activity_state(channel, True)
        
        self._buffer = []
        self._timestamps = []
        
    def on_deactivate(self):
        if self._thread and self._thread.isRunning():
            self.stop_stream()
        
        process = self._process()
        for channel in self._active_channels:
            process.set_activity_state(channel, False)

    def configure(self, active_channels, streaming_mode, channel_buffer_size, sample_rate):
        self._active_channels = list(active_channels)
        self._streaming_mode = StreamingMode.CONTINUOUS
        self._sample_rate = sample_rate
        self._buffer = []
        self._timestamps = []

    def start_stream(self):
        if not self._active_channels:
            raise RuntimeError("No active channels configured.")
        
        self._running = True
        self._start_time = time.perf_counter()
        self._thread.start()

    def stop_stream(self):
        if self._thread:
            self._running = False
            self._wait_condition.wakeAll()
            self._thread.wait()

    @property
    def constraints(self):
        process = self._process()
        return DataInStreamConstraints(
            channel_units=process.constraints.channel_units,
            sample_timing=SampleTiming.TIMESTAMP,
            streaming_modes=[StreamingMode.CONTINUOUS],
            data_type=float,
            channel_buffer_size=ScalarConstraint(default=1024, bounds=(1, 4096), increment=1),
            sample_rate=ScalarConstraint(default=self._sample_rate, bounds=(1, self._sample_rate), increment=1)
        )

    @property
    def available_samples(self):
        return len(self._buffer)

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def channel_buffer_size(self):
        return len(self._buffer)

    @property
    def streaming_mode(self):
        return StreamingMode.CONTINUOUS

    @property
    def active_channels(self):
        return self._active_channels

    def read_data(self, samples_per_channel=None):
        if not self._running:
            raise ValueError("Aquisition is not running.")
        if samples_per_channel is None:
            samples_per_channel = self.available_samples
        self._mutex.lock()
        while len(self._buffer) < samples_per_channel:
            self._wait_condition.wait(self._mutex)
        data = np.array(self._buffer[:samples_per_channel])
        timestamps = np.array(self._timestamps[:samples_per_channel])
        del self._buffer[:samples_per_channel]
        del self._timestamps[:samples_per_channel]
        self._mutex.unlock()
        return data, timestamps

    def read_data_into_buffer(self, data_buffer, samples_per_channel, timestamp_buffer=None):
        data, timestamps = self.read_data(samples_per_channel)
        data_buffer[:samples_per_channel] = data
        if timestamp_buffer is not None:
            timestamp_buffer[:samples_per_channel] = timestamps

    def read_available_data_into_buffer(self, data_buffer, timestamp_buffer=None):
        samples_available = len(self._buffer)
        self.read_data_into_buffer(data_buffer, samples_available, timestamp_buffer)
        return samples_available

    def read_single_point(self):
        data, timestamps = self.read_data(1)
        return data[0], timestamps[0]

    class AcquisitionThread(QThread):
        def __init__(self, parent):
            super().__init__(parent)
            self.parent = parent

        def run(self):
            process = self.parent._process()
            sample_interval = 1.0 / self.parent._sample_rate
            while self.parent._running:
                self.parent._mutex.lock()
                timestamp = time.perf_counter() - self.parent._start_time
                for channel in self.parent._active_channels:
                    value = process.get_process_value(channel)
                    self.parent._buffer.append(value)
                    self.parent._timestamps.append(timestamp)
                self.parent._wait_condition.wakeAll()
                self.parent._mutex.unlock()
                self.msleep(int(sample_interval * 1000))

