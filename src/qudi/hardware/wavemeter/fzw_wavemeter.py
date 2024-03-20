import serial
import queue
import struct
import time
from typing import Union, Optional, Tuple, Sequence

from PySide2 import QtCore

import numpy as np

from qudi.core.configoption import ConfigOption
# from qudi.interface.scanning_laser_interface import ScanningLaserInterface, ScanningState, ScanningLaserReturnError
from qudi.interface.data_instream_interface import DataInStreamInterface, DataInStreamConstraints, SampleTiming, StreamingMode, ScalarConstraint
from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputInterface, FiniteSamplingInputConstraints
from qudi.util.mutex import Mutex
from qudi.util.overload import OverloadedAttribute

class SerialHandler(QtCore.QObject):
    """Handler class for the socket that communicates with the MOGLabs software
    in a dedicated thread."""

    response_received = QtCore.Signal(object)

    def __init__(self, parentclass):
        super().__init__()
        # The settings are stored within the parent class
        self._parentclass = parentclass
        self.log = self._parentclass.log
        self.serial = serial.Serial()
        self.input_queue = queue.Queue()

    def connect_socket(self):
        self.log.info("Connecting to MOGLabs interface.")
        self.serial.port = self._parentclass._com_port
        self.serial.timeout = 0.1
        self.serial.open()

    def disconnect_socket(self):
        self.log.info("Disconnecting from MOGLabs interface.")
        self.serial.open()

    @QtCore.Slot(str)
    def send(self, value):
        self.input_queue.put(bytes(value, "utf8"))

    def run(self):
        self.connect_socket()
        value = None
        while True:
            if self._parentclass.module_state() == 'deactivated':
                break
            try:
                value = self.input_queue.get(block=False)
                self.serial.write(value.encode('utf8'))
            except queue.Empty:
                pass
            try:
                value = self.serial.readline()
                self.response_received.emit(value)
            except TimeoutError:
                pass
        self.disconnect_socket()

class MOGLabsFZW(DataInStreamInterface, FiniteSamplingInputInterface):
    """A class to control our MOGLabs laser to perform excitation spectroscopy.

    Example config:

    """
    _com_port = ConfigOption('com_port', 'COM6')
    refresh_rate = ConfigOption('refresh_rate', 10)

    sig_connect_cem = QtCore.Signal()
    sig_scan_done = QtCore.Signal()
    _sig_next_frame = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._constraints: Optional[DataInStreamConstraints] = None
        self.socket_thread = QtCore.QThread()
        self._socket_handler = SocketHandler(self)
        self._lock = Mutex()
        self._data_buffer: Optional[np.ndarray] = None
        self._timestamp_buffer: Optional[np.ndarray] = None
        self._current_buffer_position = 0
        self._time_offset = 0
        self._last_read = 0
        self._frame_size = 1

    # Qudi activation / deactivation
    def on_activate(self):
        """Activate module.
        """
        self.log.info("Starting the MOGLabs laser.")
        self._socket_handler.moveToThread(self.socket_thread)
        self.response_queue = queue.Queue()
        self.sig_connect_cem.connect(self._socket_handler.run)
        self._socket_handler.response_received.connect(self.on_receive)
        self.socket_thread.start()
        self.sig_connect_cem.emit()
        self._time_offset = 0

        self._constraints = DataInStreamConstraints(
            channel_units = {
                'wavelength': 'nm',
            },
            sample_timing=SampleTiming.TIMESTAMP,
            streaming_modes = [StreamingMode.CONTINUOUS],
            data_type=np.float64,
            channel_buffer_size=ScalarConstraint(default=2**16,
                                                 bounds=(2, (2**32)//10),
                                                 increment=1,
                                                 enforce_int=True),
            sample_rate=ScalarConstraint(default=1e3,bounds=(1e3,1e3),increment=0),
        )
        self._constraints_finite_sampling = FiniteSamplingInputConstraints(
            channel_units = {
                'wavelength': 'm',
            },
            frame_size_limits=(1,2**16),
            sample_rate_limits=(1,1e3),
        )


    def on_deactivate(self):
        """Deactivate module.
        """
        self.socket_thread.quit()
        self._socket_handler.response_received.disconnect()
        self.sig_connect_cem.disconnect()
        self.stop_stream()
        self._data_buffer = None
        self._timestamp_buffer = None

    # Internal communication facilities
    def on_receive(self, value):
        self.response_queue.put(value)

    def send_and_recv(self, value, timeout_s=1, check_ok=True):
        self._socket_handler.send(value)
        recv = None
        try:
            recv = self.response_queue.get(timeout=timeout_s)
        except queue.Empty:
            self.log.warning("Timeout while waiting for a response to %s", value)
        if check_ok:
            if recv is None:
                self.log.error(f"Command \"{value}\" did not return.")
            elif not recv.decode("utf8").startswith("OK"):
                self.log.error(f"Command \"{value}\" errored: \"{recv}\"")
                recv = None
        return recv

    def empty_input_queue(self):
        while not self.response_queue.empty():
            self.response_queue.get()

    def _set_continuous_acquisition(self):
        ret = self.send_and_recv("meas,extrig,off")
        if ret is None:
            return
        self.send_and_recv("meas,clear")

    def _set_oneshot_acquisition(self):
        ret = self.send_and_recv("meas,extrig,on")
        if ret is None:
            return
        self.send_and_recv("meas,clear")

    def _dump_data(self, buffer_data : Optional[np.ndarray] = None , buffer_timestamp : Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        self.empty_input_queue()
        current_time = time.time()# - self._time_offset
        self._socket_handler.send("meas,dump")
        recv = b''
        while True:
            try:
                recv += self.response_queue.get(timeout=10)
            except queue.Empty:
                self.log.warning("Timeout while waiting for a response to dump.")
                break
            if recv[-2:] == b'\r\n':
                break
        if len(recv) < 0:
            return np.empty(0),np.empty(0),0
        try:
            recv = eval(recv)
        except:
            self.log.warning("Could not parse the response.")
            return np.empty(0),np.empty(0),0

        size = struct.unpack("<I", recv[:4])[0]
        number_of_measurements = size//10
        if buffer_data is None:
            buffer_data = np.empty(number_of_measurements, dtype=np.float64)
        if buffer_timestamp is None:
            buffer_timestamp = np.empty(number_of_measurements, dtype=np.float64)
        dump_stop = min(number_of_measurements, len(buffer_timestamp))
        
        recv = recv[4:]

        dump_index = 0
        remaining_bytes = size
        d,_ = divmod(len(recv),10)
        for i in range(0,d*10,10):
            t,wavelength,_,_,_,_=struct.unpack("<HIbbbb", recv[i:(i+10)])
            remaining_bytes -= 10
            if dump_index < dump_stop:
                buffer_timestamp[dump_index] = t / 1000
                buffer_data[dump_index] = (wavelength * 1200.0 / (2**32 - 1)) * 1e-9
                dump_index += 1
        
        differences = np.diff(buffer_timestamp)
        differences[differences < 0] = 0.001 # time wrapped
        differences = np.hstack(([0], differences))
        buffer_timestamp = np.cumsum(differences)
        buffer_timestamp = buffer_timestamp - buffer_timestamp[-1]
        buffer_timestamp = current_time + buffer_timestamp
        return buffer_data,buffer_timestamp,dump_index+1

    # DataInStreamInterface
    constraints = OverloadedAttribute()
    @constraints.overload("DataInStreamInterface")
    @property
    def constraints(self) -> DataInStreamConstraints:
        if self._constraints is None:
            raise ValueError("Constraints have not yet been initialized.")
        return self._constraints

    def start_stream(self) -> None:
        """ Start the data acquisition/streaming """
        with self._lock:
            if self.module_state() == 'idle':
                self.module_state.lock()
                self._set_continuous_acquisition()
                self._time_offset = time.time()
            else:
                self.log.warning('Unable to start input stream. It is already running.')

    def stop_stream(self) -> None:
        """ Stop the data acquisition/streaming """
        with self._lock:
            if self.module_state() == 'locked':
                self.module_state.unlock()
            else:
                self.log.warning('Unable to stop wavemeter input stream as nothing is running.')

    def read_data_into_buffer(self,
                              data_buffer: np.ndarray,
                              samples_per_channel: int,
                              timestamp_buffer: Optional[np.ndarray] = None) -> None:
        """ Read data from the stream buffer into a 1D numpy array given as parameter.
        Samples of all channels are stored interleaved in contiguous memory.
        In case of a multidimensional buffer array, this buffer will be flattened before written
        into.
        The 1D data_buffer can be unraveled into channel and sample indexing with:

            data_buffer.reshape([<samples_per_channel>, <channel_count>])

        The data_buffer array must have the same data type as self.constraints.data_type.

        In case of SampleTiming.TIMESTAMP a 1D numpy.float64 timestamp_buffer array has to be
        provided to be filled with timestamps corresponding to the data_buffer array. It must be
        able to hold at least <samples_per_channel> items:

        If the number of available samples is too low, you will get zeros at the end.
        """
        with self._lock:
            if self.module_state() != 'locked':
                raise RuntimeError('Unable to read data. Stream is not running.')
            self._dump_data(data_buffer, timestamp_buffer)

    def read_available_data_into_buffer(self,
                                        data_buffer: np.ndarray,
                                        timestamp_buffer: Optional[np.ndarray] = None) -> int:
        """ Read data from the stream buffer into a 1D numpy array given as parameter.
        All samples for each channel are stored in consecutive blocks one after the other.
        The number of samples read per channel is returned and can be used to slice out valid data
        from the buffer arrays like:

            valid_data = data_buffer[:<channel_count> * <return_value>]
            valid_timestamps = timestamp_buffer[:<return_value>]

        See "read_data_into_buffer" documentation for more details.

        This method will read all currently available samples into buffer. If number of available
        samples exceeds buffer size, read only as many samples as fit into the buffer and drop the rest.
        """
        with self._lock:
            if self.module_state() != 'locked':
                raise RuntimeError('Unable to read data. Stream is not running.')
            if timestamp_buffer is None:
                raise RuntimeError(
                    'SampleTiming.TIMESTAMP mode requires a timestamp buffer array'
                )

            return self._dump_data(data_buffer, timestamp_buffer)[-1]

    def read_data(self,
                  samples_per_channel: Optional[int] = None
                  ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """ Read data from the stream buffer into a 1D numpy array and return it.
        All samples for each channel are stored in consecutive blocks one after the other.
        The returned data_buffer can be unraveled into channel samples with:

            data_buffer.reshape([<samples_per_channel>, <channel_count>])

        The numpy array data type is the one defined in self.constraints.data_type.

        In case of SampleTiming.TIMESTAMP a 1D numpy.float64 timestamp_buffer array will be
        returned as well with timestamps corresponding to the data_buffer array.

        If samples_per_channel is omitted all currently available samples are read from buffer.
        This method will not return until all requested samples have been read or a timeout occurs.
        """
        with self._lock:
            if self.module_state() != 'locked':
                raise RuntimeError('Unable to read data. Stream is not running.')

            buffer_data, buffer_timestamp, l = self._dump_data()
            if samples_per_channel is not None and l > samples_per_channel:
                buffer_data = buffer_data[:samples_per_channel]
                buffer_timestamp = buffer_timestamp[:samples_per_channel]
            return buffer_data, buffer_timestamp

    def read_single_point(self) -> Tuple[np.ndarray, Union[None, np.float64]]:
        """ This method will initiate a single sample read on each configured data channel.
        In general this sample may not be acquired simultaneous for all channels and timing in
        general can not be assured. Us this method if you want to have a non-timing-critical
        snapshot of your current data channel input.
        May not be available for all devices.
        The returned 1D numpy array will contain one sample for each channel.

        In case of SampleTiming.TIMESTAMP a single numpy.float64 timestamp value will be returned
        as well.
        """
        with self._lock:
            if self.module_state() != 'locked':
                raise RuntimeError('Unable to read data. Stream is not running.')
            self.send_and_recv("meas,softrig")
            f = self.send_and_recv("freq", check_ok=False)
            if f is None:
                return np.empty(0),None
            try:
                return np.array(float(f)),None 
            except ValueError:
                return np.empty(0), None

    @property
    def available_samples(self) -> int:
        # self.log.warning("Available samples reading is not available on the MOGLabs FZW.")
        return 1

    @property
    def sample_rate(self) -> float:
        # self.log.warning("Sample rate reading is not available on the MOGLabs FZW.")
        return 140

    @property
    def channel_buffer_size(self) -> int:
        return (2**32)/10

    @property 
    def streaming_mode(self) -> StreamingMode:
        return StreamingMode.CONTINUOUS

    @property
    def active_channels(self):
        return ['wavelength']

    def configure(self,
                  active_channels: Sequence[str],
                  streaming_mode: Union[StreamingMode, int],
                  channel_buffer_size: int,
                  sample_rate: float) -> None:
        """ Configure a data stream. See read-only properties for information on each parameter. """
        # self.log.warning("There is nothing to configure on the MOGLabs FZW!")

    # FiniteSamplingInputInterface
    @constraints.overload("FiniteSamplingInputInterface")
    @property
    def constraints(self):
        return self._constraints_finite_sampling

    @property 
    def frame_size(self):
        return self._frame_size

    @property
    def samples_in_buffer(self):
        return 1
    
    def set_active_channels(self, _):
        pass 

    def set_sample_rate(self, rate):
        pass

    def set_frame_size(self, size):
        self._frame_size = size

    def start_buffered_acquisition(self):
        self._set_oneshot_acquisition()

    def stop_buffered_acquisition(self):
        pass

    def get_buffered_samples(self, number_of_samples: Optional[int] =None):
        with self._lock:
            if number_of_samples is None:
                buffer_data, _, l = self._dump_data()
                return {"wavelength": buffer_data}
            elif number_of_samples > self.frame_size:
                raise ValueError(f"You are asking for too many samples ({number_of_samples} for a maximum of {self.frame_size}.")
            else:
                buffer_data, _, l = self._dump_data()
                while l < number_of_samples:
                    time.sleep(0.1)
                    added_buffer_data, added_buffer_timestamp, added_l = self._dump_data()
                    l += added_l
                    buffer_data = np.concatenate(buffer_data, added_buffer_data)
                return {"wavelength": buffer_data[:number_of_samples]}

    def acquire_frame(self, frame_size=None):
        old_frame_size = self.frame_size
        if frame_size is not None:
            self.set_frame_size(frame_size)
        self.start_buffered_acquisition()
        data = self.get_buffered_samples(self.frame_size)
        self.stop_buffered_acquisition()
        if frame_size is not None:
            self.set_frame_size(old_frame_size)
        return data
