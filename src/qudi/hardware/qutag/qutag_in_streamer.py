import ctypes
import numpy as np
import platform
from typing import Tuple, List, Optional, Sequence, Union
from qudi.core.configoption import ConfigOption
from qudi.util.mutex import Mutex
from qudi.util.constraints import ScalarConstraint
from qudi.interface.data_instream_interface import (
    DataInStreamInterface,
    DataInStreamConstraints,
)
from qudi.interface.data_instream_interface import StreamingMode, SampleTiming
from qudi.interface.sampled_finite_state_interface import (
    SampledFiniteStateInterface,
    transition_to,
    transition_from,
    state,
)


class QuTagInStreamer(DataInStreamInterface):
    _device_id = ConfigOption("device_id", default=-1, missing="info")
    _tdcbase_path = ConfigOption("tdcbase_path", default="tdcbase.dll", missing="warn")
    _max_channel_samples_buffer = ConfigOption(
        name="max_channel_samples_buffer",
        default=1024**2,
        missing="info",
        constructor=lambda x: max(int(round(x)), 1024**2),
    )
    _coincidence_window = ConfigOption(
        "coincidence_window",
        default=2000,
        missing="info",
        constructor=lambda x: max(int(round(int)), 2000000000),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tdclib = None
        self._number_of_channels = 0
        self._data_lock = Mutex()
        self.__sample_rate = -1.0
        self.__buffer_size = -1
        self.__buffer = np.array()
        self.__read_buffer = np.zeros(int(59), dtype=np.int32)
        self.__stored_watchdog_delay = 0.0
        self.__buffer_start = 0
        self.__buffer_stop = -1
        self.__streaming_mode = None
        self.__active_channels = []

    def on_activate(self):
        if platform.system() == "Windows":
            self.tdclib = ctypes.windll.LoadLibrary(self._tdcbase_path)
        if platform.system() == "Linux":
            self.tdclib = ctypes.cdll.LoadLibrary(self._tdcbase_path)
        if self.tdclib is None:
            raise Error("Could not load QuTag library.")
        self.tdclib.TDC_getVersion.argtypes = None
        self.tdclib.TDC_getVersion.restype = ctypes.c_double
        self.tdclib.TDC_perror.argtypes = [ctypes.c_int32]
        self.tdclib.TDC_perror.restype = ctypes.c_char_p
        self.tdclib.TDC_init.argtypes = [ctypes.c_int32]
        self.tdclib.TDC_init.restype = ctypes.c_int32
        self.tdclib.TDC_deInit.argtypes = None
        self.tdclib.TDC_deInit.restype = ctypes.c_int32
        self.tdclib.TDC_setCoincidenceWindow.argtypes = [ctypes.c_int32]
        self.tdclib.TDC_setCoincidenceWindow.restype = ctypes.c_int32
        self.tdclib.TDC_setExposureTime.argtypes = [ctypes.c_int32]
        self.tdclib.TDC_setExposureTime.restype = ctypes.c_int32
        self.tdclib.TDC_getCoincCounters.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.tdclib.TDC_getCoincCounters.restype = ctypes.c_int32
        rc = self.tdclib.TDC_init(self._device_id)
        self.perror("TDC_init", rc)
        self._number_of_channels = self.tdclib.TDC_getChannelCount()
        self._constraints = DataInStreamConstraints(
            channel_units={str(i): "counts/s" for i in range(self._number_of_channels)},
            sample_timing=SampleTiming.CONSTANT,
            streaming_modes=[
                StreamingMode.CONTINUOUS
            ],  # TODO: Implement FINITE streaming mode
            data_type=np.float64,
            channel_buffer_size=ScalarConstraint(
                default=1024**2,
                bounds=(2, self._max_channel_samples_buffer),
                increment=1,
                enforce_int=True,
            ),
            sample_rate=ScalarConstraint(
                default=10.0,
                bounds=(
                    1000/(2**16-1),
                    1000,
                ),
                increment=1,
                enforce_int=False,
            ),
        )
        self.enable_watchdog()
        self.start_watchdog()

    def perror(self, environ, returnCode):
        if returnCode != 0:
            msg = self.tdclib.TDC_perror(returnCode)
            self.log.error("Error in %s: %s" % (environ, msg))

    def on_deactivate(self):
        rc = self.tdclib.TDC_deInit()
        self.perror("TDC_deInit", rc)
        self.stop_watchdog()
        self.disable_watchdog()

    # SampledFiniteStateInterface
    @state
    @transition_to(("start", "prepare_sampling"))
    def idle(self):
        """idling mode: do not read the current count rates."""
        pass

    @state
    @transition_to(("start", "sampling"))
    @transition_to(("stop", "idle"))
    def prepare_sampling(self):
        """Prepare sampling mode by setting exposure time and preparing buffers."""
        rc = self.tdclib.TDC_setCoincidenceWindow(self._coincidence_window)
        self.perror("TDC_setCoincidenceWindow", rc)
        if rc != 0:
            self.stop_stream()
        exposure = int(round(1000/self._sample_rate))
        rc = self.tdclib.TDC_setExposureTime(exposure)
        self.perror("TDC_setExposureTime", rc)
        if rc != 0:
            self.stop_stream()
        self.__stored_watchdog_delay = self._watchdog_delay
        self._watchdog_delay = 0.8 / self.sample_rate
        rc = qutag.tdclib.TDC_getCoincCounters(self.__read_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.byref(updates))
        qutag.perror("TDC_getCoincCounters", rc)
        self.watchdog_event("start")

    @state
    @transition_to(("stop", "stop_sampling"))
    def sampling(self):
        """Read the current count rates and store it in the current buffer."""
        updates = ctypes.c_int32()
        rc = qutag.tdclib.TDC_getCoincCounters(self.__read_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.byref(updates))
        qutag.perror("TDC_getCoincCounters", rc)
        with self._data_lock:
            available_space = self.__buffer_start + self.__buffer_size * 59 - self.__buffer_stop
            number_of_values = updates.value*59
            if available_space <= number_of_values:
                self.log.error("Buffer overflow. Please increase buffer size.")
                self.stop_stream()
                return
            if number_of_values > 59: # fill the missed samples with NaN
                number_of_nan = (updates.value - 1) * 59
                space_at_the_end = self.__buffer_size * 59 - self.__buffer_stop
                if space_at_the_end >= number_of_nan:
                    self.__buffer[self.__buffer_stop:self.__buffer_stop+number_of_nan] = np.nan
                    self.__buffer_stop = self.__buffer_stop+number_of_nan
                else:
                    self.__buffer[self.__buffer_stop:] = np.nan
                    to_be_stored_at_the_start = (number_of_nan - space_at_the_end)
                    self.__buffer[:to_be_stored_at_the_start] = np.nan
                    self.__buffer_stop = to_be_stored_at_the_start
                number_of_values -= number_of_nan
            if number_of_values > 0:
                space_at_the_end = self.__buffer_size * 59 - self.__buffer_stop
                if space_at_the_end >= 59:
                    self.__buffer[self.__buffer_stop:self.__buffer_stop+59] = self.__read_buffer
                    self.__buffer_stop += 59
                else:
                    self.__buffer[self.__buffer_stop:] = self.__read_buffer[:space_at_the_end]
                    to_be_stored_at_the_start = (59 - space_at_the_end)
                    self.__buffer[:to_be_stored_at_the_start] = self.__read_buffer[:to_be_stored_at_the_start]
                    self.__buffer_stop = to_be_stored_at_the_start

    @state
    @transition_to(("stop", "idle"))
    def stop_sampling(self):
        """Transition back to idle mode."""
        self._watchdog_delay = self.__stored_watchdog_delay
        self.watchdog_event("stop")

    @property
    def constraints(self) -> DataInStreamConstraints:
        """Read-only property returning the constraints on the settings for this data streamer."""
        return self._constraints

    @property
    def available_samples(self) -> int:
        """Read-only property to return the currently available number of samples per channel ready
        to read from buffer.
        It must be ensured that each channel can provide at least the number of samples returned
        by this property.
        """
        with self._data_lock:
            return self.__buffer_stop - self.__buffer_start + 1

    @property
    def channel_buffer_size(self) -> int:
        """Read-only property returning the currently set buffer size in samples per channel.
        The total buffer size in bytes can be estimated by:
            <buffer_size> * <channel_count> * numpy.nbytes[<data_type>]
        """
        return self.__buffer_size

    @property
    def streaming_mode(self) -> StreamingMode:
        """Read-only property returning the currently configured StreamingMode Enum"""
        return self.__streaming_mode

    @property
    def active_channels(self) -> List[str]:
        """Read-only property returning the currently configured active channel names"""
        return list(self.__active_channels)

    def configure(
        self,
        active_channels: Sequence[str],
        streaming_mode: Union[StreamingMode, int],
        channel_buffer_size: int,
        sample_rate: float,
    ) -> None:
        """Configure a data stream. See read-only properties for information on each parameter."""
        if self.module_state() == "locked":
            raise RuntimeError(
"Unable to configure data stream while it is already running"
            )
        streaming_mode = StreamingMode(streaming_mode)
        channel_buffer_size = int(round(channel_buffer_size))
        if any(ch not in self._constraints.channel_units for ch in active_channels):
            raise ValueError(
                f"Invalid channel to stream from encountered {tuple(active_channels)}. \n"
                f"Valid channels are: {tuple(self._constraints.channel_units)}"
            )
        if (
            streaming_mode not in self._constraints.streaming_modes
            or streaming_mode == StreamingMode.INVALID
        ):
            raise ValueError(
                f'Invalid streaming mode "{streaming_mode}" encountered.\n'
                f"Valid modes are: {self._constraints.streaming_modes}."
            )
        self._constraints.channel_buffer_size.check(channel_buffer_size)
        self._constraints.sample_rate.check(sample_rate)

        self.__active_channels = tuple(active_channels)
        self.__streaming_mode = streaming_mode
        self.__buffer_size = channel_buffer_size
        self.__sample_rate = sample_rate
        with self._data_lock:
            self.__buffer_start = 0
            self.__buffer_stop = 0
            self.__read_buffer = np.zeros(int(59), dtype=np.int32)
            self.__buffer = np.empty(
                self.__buffer_size * 59, dtype=self._constraints.data_type
                )

    def start_stream(self) -> None:
        """ Start the data acquisition/streaming """
        if self.module_state() == 'locked':
            self.log.warning('Unable to start input stream. It is already running.')
        else:
            self.module_state.lock()
            try:
                self.watchdog_event("start")
            except:
                self.module_state.unlock()
                raise

    def stop_stream(self) -> None:
        """ Stop the data acquisition/streaming """
        try:
            self.watchdog_event("stop")
        finally:
            if self.module_state() == 'locked':
                self.module_state.unlock()

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

        This function is blocking until the required number of samples has been acquired.
        """
        pass
