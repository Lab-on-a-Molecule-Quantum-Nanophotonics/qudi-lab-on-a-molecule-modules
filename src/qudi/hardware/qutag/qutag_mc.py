import ctypes
import itertools
import platform
import time
from collections.abc import Sequence

import numpy as np
from qudi.core.configoption import ConfigOption
from qudi.util.constraints import ScalarConstraint
from qudi.util.mutex import Mutex
from qudi.util.overload import OverloadedAttribute

from qudi.interface.data_instream_interface import (
    DataInStreamConstraints,
    DataInStreamInterface,
    SampleTiming,
    StreamingMode,
)
from qudi.interface.sampled_finite_state_interface import SampledFiniteStateInterface, initial, state, transition_to
from qudi.interface.start_stop_interface import (
    StartStopHardwareState,
    StartStopHistogram,
    StartStopHistogramConfiguration,
    StartStopInterface,
)

AVAILABLE_HISTOGRAMS = tuple([f"{i}/{j}" for (i,j) in itertools.product(range(3), range(3))])
def histogram_pair_from_name(name:str) -> tuple[int, int]:
    splitted = name.split("/")
    if len(splitted) != 2:
        raise ValueError(f"Could not extract inputs from histogram name '{name}'")
    start_name = splitted[0]
    stop_name = splitted[1]
    start = int(start_name)
    stop = int(stop_name)
    if not (0 < start < 33):
        raise ValueError(f"Start channel {start} is not in range 1...32.")
    if not (0 < stop < 33):
        raise ValueError(f"Stop channel {stop} is not in range 1...32.")
    return (start, stop)

class QuTagMC(DataInStreamInterface, SampledFiniteStateInterface, StartStopInterface):
    """
    Use a QuTAG device as a digital streamer. 
    Tested with QuTAG-MC.

    Example configuration for copy-paste:
    ```yaml
    qutag_mc:
        module.Class: qutag.qutag_mc.QuTagMC
        options:
            device_id: -1 # Take any device available
            tdcbase_path: "C:/Users/Lab66_spectrometer/Desktop/QUTAG-MC-WIN64QT5-v1.0.15/QUTAG-MC-WIN64QT5-v1.0.15/userlib/lib64/tdcbase.dll" # path to the DLL
            max_channel_samples_buffer: 2048 # default
            coincidence_window: 2000 # default
    ```
    """
    _device_id = ConfigOption("device_id", default=-1, missing="info")
    _tdcbase_path = ConfigOption("tdcbase_path", default="tdcbase.dll", missing="warn")
    _max_channel_samples_buffer = ConfigOption(
        name="max_channel_samples_buffer",
        default=1024**2,
        missing="info",
        constructor=lambda x: max(int(round(x)), 1024**2),
    )
    _coincidence_window = ConfigOption(
        "coincidence_window", default=2000, missing="info", constructor=lambda x: max(int(round(int(x))), 2000000000)
    )
    _NUMBER_OF_COINCIDENCE_CHANNELS = 59

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tdclib = None
        self._number_of_channels = 0
        self._data_lock = Mutex()
        self.__sample_rate = -1.0
        self.__buffer_size = -1
        self.__buffer = np.array([])
        self.__read_buffer = np.zeros(int(self._NUMBER_OF_COINCIDENCE_CHANNELS), dtype=np.int32)
        self.__stored_watchdog_delay = self.watchdog_delay
        self.__buffer_start = 0
        self.__buffer_stop = 0
        self.__streaming_mode = None
        self.__active_channels = []
        self.__coincidence_channels_names = [str(i) for i in range(33)]
        self.__coincidence_channels_names[0] = "Start"
        self.__coincidence_channels_names.extend(
            [
                "1/2",
                "1/3",
                "2/3",
                "1/4",
                "2/4",
                "3/4",
                "1/5",
                "2/5",
                "3/5",
                "4/5",
                "1/2/3",
                "1/2/4",
                "1/3/4",
                "2/3/4",
                "1/2/5",
                "1/3/5",
                "2/3/5",
                "1/4/5",
                "2/4/5",
                "3/4/5",
                "1/2/3/4",
                "1/2/3/5",
                "1/2/4/5",
                "1/3/4/5",
                "2/3/4/5",
                "1/2/3/4/5",
            ]
        )
        self.__inverted_coincidence_channels_names = {ch: i for i, ch in enumerate(self.__coincidence_channels_names)}
        self._enabled_histograms = {histogram:False for histogram in AVAILABLE_HISTOGRAMS}
        self._histogram_configurations = {
                histogram: StartStopHistogramConfiguration(1.0, 50.0, 30.0)
                for histogram in AVAILABLE_HISTOGRAMS
                }
        self._start_stop_status = StartStopHardwareState.UNCONFIGURED
        self._histogram_data: dict[str, StartStopHistogram | None] = {
                histogram: None
                for histogram in AVAILABLE_HISTOGRAMS
                }

    def on_activate(self):
        if platform.system() == "Windows":
            self.tdclib = ctypes.windll.LoadLibrary(self._tdcbase_path)
        if platform.system() == "Linux":
            self.tdclib = ctypes.cdll.LoadLibrary(self._tdcbase_path)
        if self.tdclib is None:
            raise ValueError("Could not load QuTag library.")
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
        self.tdclib.TDC_getCoincCounters.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
        self.tdclib.TDC_getCoincCounters.restype = ctypes.c_int32
        self.tdclib.TDC_addHistogram.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self.tdclib.TDC_addHistogram.restype = ctypes.c_int32
        self.tdclib.TDC_enableStartStop.argtypes = [ctypes.c_int32]
        self.tdclib.TDC_enableStartStop.restype = ctypes.c_int32
        self.tdclib.TDC_getHistogram.argtypes = [
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, 
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int64),
                ]
        self.tdclib.TDC_getHistogram.restype = ctypes.c_int32
        self.tdclib.TDC_getHistogramParams.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32)]
        self.tdclib.TDC_getHistogramParams.restype = ctypes.c_int32
        self.tdclib.TDC_setHistogramParams.argtypes = [ctypes.c_int32,ctypes.c_int32]
        self.tdclib.TDC_setHistogramParams.restype = ctypes.c_int32
        self.tdclib.TDC_getTimebase.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.tdclib.TDC_getTimebase.restype = ctypes.c_int32
        self.tdclib.TDC_freezeBuffers.argtypes = [ctypes.c_int32]
        self.tdclib.TDC_freezeBuffers.restype = ctypes.c_int32
        rc = self.tdclib.TDC_init(self._device_id)
        self.perror("TDC_init", rc)
        if rc != 0:
            raise ValueError("Could not connect to Qutag device.")
        self._number_of_channels = self.tdclib.TDC_getChannelCount()
        self._constraints = DataInStreamConstraints(
            channel_units={ch: "counts/s" for ch in self.__coincidence_channels_names},
            sample_timing=SampleTiming.CONSTANT,
            streaming_modes=[StreamingMode.CONTINUOUS],  # TODO: Implement FINITE streaming mode
            data_type=np.float64,
            channel_buffer_size=ScalarConstraint(
                default=1024**2, bounds=(2, self._max_channel_samples_buffer), increment=1, enforce_int=True
            ),
            sample_rate=ScalarConstraint(
                default=10.0, bounds=(1000 / (2**16 - 1), 1000), increment=1, enforce_int=False
            ),
        )
        self.configure(
            active_channels=list(self._constraints.channel_units),
            streaming_mode=StreamingMode.CONTINUOUS,
            channel_buffer_size=int(self._constraints.channel_buffer_size.default),
            sample_rate=self._constraints.sample_rate.default,
        )
        self.enable_watchdog()
        self.start_watchdog()
        self._start_stop_status = StartStopHardwareState.UNCONFIGURED

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
    @initial
    @transition_to(("start", "prepare_sampling"))
    @transition_to(("stop", "idle"))
    def idle(self):
        """idling mode: do not read the current count rates."""

    @state
    @transition_to(("start", "sampling"))
    @transition_to(("stop", "idle"))
    def prepare_sampling(self):
        """Prepare sampling mode by setting exposure time and preparing buffers."""
        rc = self.tdclib.TDC_setCoincidenceWindow(self._coincidence_window)
        self.perror("TDC_setCoincidenceWindow", rc)
        if rc != 0:
            self.stop_stream()
        exposure = int(round(1000 / self.__sample_rate))
        rc = self.tdclib.TDC_setExposureTime(exposure)
        self.perror("TDC_setExposureTime", rc)
        if rc != 0:
            self.stop_stream()
        self.__stored_watchdog_delay = self.watchdog_delay
        self.watchdog_delay = 0.8 / self.sample_rate
        updates = ctypes.c_int32()
        rc = self.tdclib.TDC_getCoincCounters(
            self.__read_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.byref(updates)
        )
        self.perror("TDC_getCoincCounters", rc)
        self.watchdog_event("start")

    @state
    @transition_to(("stop", "idle"))
    def sampling(self):
        """Read the current count rates and store it in the current buffer."""
        updates = ctypes.c_int32()
        rc = self.tdclib.TDC_getCoincCounters(
            self.__read_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.byref(updates)
        )
        self.perror("TDC_getCoincCounters", rc)
        if updates.value <= 0:
            return
        with self._data_lock:
            available_space = (
                self.__buffer_start + self.__buffer_size - self.__buffer_stop
            )
            number_of_values = updates.value * self._NUMBER_OF_COINCIDENCE_CHANNELS
            if available_space <= number_of_values:
                self.log.error(f"Buffer overflow. Please increase buffer size. Available space is {available_space}, number of values read is {number_of_values}.")
                self.stop_stream()
                return
            if number_of_values > self._NUMBER_OF_COINCIDENCE_CHANNELS:  # fill the missed samples with NaN
                number_of_nan = (updates.value - 1) * self._NUMBER_OF_COINCIDENCE_CHANNELS
                space_at_the_end = self.__buffer_size - self.__buffer_stop
                if space_at_the_end >= number_of_nan:
                    self.__buffer[self.__buffer_stop : self.__buffer_stop + number_of_nan] = np.nan
                    self.__buffer_stop = (self.__buffer_stop + number_of_nan) % self.__buffer_size
                else:
                    self.__buffer[self.__buffer_stop :] = np.nan
                    to_be_stored_at_the_start = number_of_nan - space_at_the_end
                    self.__buffer[:to_be_stored_at_the_start] = np.nan
                    self.__buffer_stop = to_be_stored_at_the_start
                number_of_values -= number_of_nan
            if number_of_values > 0:
                space_at_the_end = self.__buffer_size - self.__buffer_stop
                if space_at_the_end >= self._NUMBER_OF_COINCIDENCE_CHANNELS:
                    self.__buffer[self.__buffer_stop : self.__buffer_stop + self._NUMBER_OF_COINCIDENCE_CHANNELS] = (
                        self.__read_buffer
                    )
                    self.__buffer_stop = (self.__buffer_stop + self._NUMBER_OF_COINCIDENCE_CHANNELS) % self.__buffer_size
                else:
                    self.__buffer[self.__buffer_stop :] = self.__read_buffer[:space_at_the_end]
                    to_be_stored_at_the_start = self._NUMBER_OF_COINCIDENCE_CHANNELS - space_at_the_end
                    self.__buffer[:to_be_stored_at_the_start] = self.__read_buffer[:to_be_stored_at_the_start]
                    self.__buffer_stop = to_be_stored_at_the_start

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
            stop = self.__buffer_stop
            start = self.__buffer_start
            if stop >= start:
                return (stop - start) // self._NUMBER_OF_COINCIDENCE_CHANNELS
            else:
                number_to_the_end = self.__buffer_size - self.__buffer_start
                number_from_start = self.__buffer_stop
                return (number_to_the_end + number_from_start) // self._NUMBER_OF_COINCIDENCE_CHANNELS

    @property
    def sample_rate(self) -> float:
        """ Read-only property returning the currently set sample rate in Hz.
        For SampleTiming.CONSTANT this is the sample rate of the hardware, for any other timing mode
        this property represents only a hint to the actual hardware timebase and can not be
        considered accurate.
        """
        return self.__sample_rate

    @property
    def channel_buffer_size(self) -> int:
        """Read-only property returning the currently set buffer size in samples per channel.
        The total buffer size in bytes can be estimated by:
            <buffer_size> * <channel_count> * numpy.nbytes[<data_type>]
        """
        return self.__buffer_size // self._NUMBER_OF_COINCIDENCE_CHANNELS

    @property
    def streaming_mode(self) -> StreamingMode:
        """Read-only property returning the currently configured StreamingMode Enum"""
        return StreamingMode(self.__streaming_mode)

    @property
    def active_channels(self) -> list[str]:
        """Read-only property returning the currently configured active channel names"""
        return list(self.__active_channels)

    configure = OverloadedAttribute()

    @configure.overload("DataInStreamInterface")
    def configure(
        self,
        active_channels: Sequence[str],
        streaming_mode: StreamingMode | int,
        channel_buffer_size: int,
        sample_rate: float,
    ) -> None:
        """Configure a data stream. See read-only properties for information on each parameter."""
        if self.module_state() == "locked":
            raise RuntimeError("Unable to configure data stream while it is already running")
        streaming_mode = StreamingMode(streaming_mode)
        channel_buffer_size = int(round(channel_buffer_size))
        if any(ch not in self._constraints.channel_units for ch in active_channels):
            raise ValueError(
                f"Invalid channel to stream from encountered {tuple(active_channels)}. \n"
                f"Valid channels are: {tuple(self._constraints.channel_units)}"
            )
        if streaming_mode not in self._constraints.streaming_modes or streaming_mode == StreamingMode.INVALID:
            raise ValueError(
                f'Invalid streaming mode "{streaming_mode}" encountered.\n'
                f"Valid modes are: {self._constraints.streaming_modes}."
            )
        self._constraints.channel_buffer_size.check(channel_buffer_size)
        self._constraints.sample_rate.check(sample_rate)

        self.__active_channels = tuple(active_channels)
        self.__streaming_mode = streaming_mode
        self.__buffer_size = channel_buffer_size * self._NUMBER_OF_COINCIDENCE_CHANNELS
        self.__sample_rate = sample_rate
        with self._data_lock:
            self.__buffer_start = 0
            self.__buffer_stop = 0
            self.__read_buffer = np.zeros(int(self._NUMBER_OF_COINCIDENCE_CHANNELS), dtype=np.int32)
            self.__buffer = np.empty(
                self.__buffer_size,  dtype=self._constraints.data_type
            )

    def start_stream(self) -> None:
        """Start the data acquisition/streaming"""
        if self.module_state() == 'locked':
            self.log.warning('Unable to start input stream. It is already running.')
        else:
            self.module_state.lock()
            try:
                self.watchdog_event("start")
            except:
                if self._start_stop_status not in (StartStopHardwareState.RUNNING, StartStopHardwareState.PAUSED): 
                    self.module_state.unlock()
                raise

    def stop_stream(self) -> None:
        """Stop the data acquisition/streaming"""
        try:
            self.watchdog_delay = self.__stored_watchdog_delay
            self.watchdog_event("stop")
        finally:
            if self.module_state() == 'locked' and self._start_stop_status not in (StartStopHardwareState.RUNNING, StartStopHardwareState.PAUSED): 
                self.module_state.unlock()

    def read_data_into_buffer(
        self, data_buffer: np.ndarray, samples_per_channel: int, timestamp_buffer: np.ndarray | None = None
    ) -> None:
        """Read data from the stream buffer into a 1D numpy array given as parameter.
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
        channel_count = len(self.__active_channels)
        if samples_per_channel is None:
            samples_per_channel = len(data_buffer) // channel_count
            to_be_read = len(data_buffer)
        else:
            to_be_read = samples_per_channel * channel_count
        if self.available_samples <= samples_per_channel and self.module_state() != 'locked':
            raise RuntimeError('Unable to read data. Device is not running.')
        if not isinstance(data_buffer, np.ndarray) or data_buffer.dtype != self._constraints.data_type:
            raise TypeError(f'data_buffer must be numpy.ndarray with dtype {self._constraints.data_type}')
        selected_channels = [self.__inverted_coincidence_channels_names[ch] for ch in self.__active_channels]
        data_buffer_index = 0
        try:
            while samples_per_channel > 0 and self.module_state() == 'locked':
                number_of_waits_remaining = 3
                while number_of_waits_remaining > 0 and self.available_samples <= 0:
                    time.sleep(self.watchdog_delay)
                    number_of_waits_remaining -= 1
                if number_of_waits_remaining <= 0:
                    self.log.error("Timeout while reading data. Stopping streamer.")
                    self.stop_stream()
                    break
                samples_to_be_read = to_be_read * channel_count
                with self._data_lock:
                    stop = self.__buffer_stop
                    start = self.__buffer_start
                    valid_samples_from_the_start = 0
                    if stop >= start:
                        consumed_buffer_indices = stop - start
                        valid_samples_to_the_end = consumed_buffer_indices // self._NUMBER_OF_COINCIDENCE_CHANNELS
                        valid_samples_to_the_end = min(valid_samples_to_the_end, samples_to_be_read)
                    else:
                        number_to_the_end = self.__buffer_size - self.__buffer_start
                        valid_samples_to_the_end = number_to_the_end // self._NUMBER_OF_COINCIDENCE_CHANNELS
                        number_from_start = self.__buffer_stop
                        valid_samples_from_the_start = number_from_start // self._NUMBER_OF_COINCIDENCE_CHANNELS
                        if valid_samples_to_the_end >= samples_to_be_read:
                            valid_samples_to_the_end = samples_to_be_read
                            valid_samples_from_the_start = 0
                        elif (valid_samples_to_the_end + valid_samples_from_the_start) > samples_to_be_read:
                            valid_samples_from_the_start = samples_to_be_read - valid_samples_to_the_end
                    buffer_last_index_to_the_end = (
                        start + valid_samples_to_the_end * self._NUMBER_OF_COINCIDENCE_CHANNELS
                    )
                    buffer_last_index_from_the_start = (
                        valid_samples_from_the_start * self._NUMBER_OF_COINCIDENCE_CHANNELS
                    )
                    if valid_samples_to_the_end > 0:
                        data_buffer_extent = valid_samples_to_the_end * channel_count
                        data_buffer[data_buffer_index : (data_buffer_index + data_buffer_extent)] = (
                            self.__buffer[start:buffer_last_index_to_the_end]
                            .reshape((valid_samples_to_the_end, self._NUMBER_OF_COINCIDENCE_CHANNELS))[
                                :, selected_channels
                            ]
                            .reshape((data_buffer_extent,))
                        ) * self.sample_rate
                        data_buffer_index += data_buffer_extent
                    if valid_samples_from_the_start > 0:
                        data_buffer_extent = valid_samples_from_the_start * channel_count
                        data_buffer[data_buffer_index : (data_buffer_index + data_buffer_extent)] = (
                            self.__buffer[:buffer_last_index_from_the_start]
                            .reshape((valid_samples_from_the_start, self._NUMBER_OF_COINCIDENCE_CHANNELS))[
                                :, selected_channels
                            ]
                            .reshape((data_buffer_extent,))
                        ) * self.sample_rate
                        data_buffer_index += data_buffer_extent
                    consumed_samples = valid_samples_from_the_start + valid_samples_to_the_end
                    self.__buffer_start = (
                        self.__buffer_start + consumed_samples * self._NUMBER_OF_COINCIDENCE_CHANNELS
                    ) % self.__buffer_size
                samples_per_channel = samples_per_channel - consumed_samples

        except Exception as e:
            self.log.exception(f'Getting samples from streamer failed, got exception {e}. Stopping streamer.')
            self.stop_stream()

    def read_available_data_into_buffer(
        self, data_buffer: np.ndarray, timestamp_buffer: np.ndarray | None = None
    ) -> int:
        """Read data from the stream buffer into a 1D numpy array given as parameter.
        All samples for each channel are stored in consecutive blocks one after the other.
        The number of samples read per channel is returned and can be used to slice out valid data
        from the buffer arrays like:

            valid_data = data_buffer[:<channel_count> * <return_value>]
            valid_timestamps = timestamp_buffer[:<return_value>]

        See "read_data_into_buffer" documentation for more details.

        This method will read all currently available samples into buffer. If number of available
        samples exceeds buffer size, read only as many samples as fit into the buffer.
        """
        channel_count = len(self.__active_channels)
        samples_per_channel = min(self.available_samples, data_buffer.size // channel_count)
        self.read_data_into_buffer(
            data_buffer=data_buffer, samples_per_channel=samples_per_channel, timestamp_buffer=timestamp_buffer
        )
        return samples_per_channel

    def read_data(self, samples_per_channel: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Read data from the stream buffer into a 1D numpy array and return it.
        All samples for each channel are stored in consecutive blocks one after the other.
        The returned data_buffer can be unraveled into channel samples with:

            data_buffer.reshape([<channel_count>, number_of_samples])

        The numpy array data type is the one defined in self.constraints.data_type.

        In case of SampleTiming.TIMESTAMP a 1D numpy.float64 timestamp_buffer array will be
        returned as well with timestamps corresponding to the data_buffer array.

        If number_of_samples is omitted all currently available samples are read from buffer.
        This method will not return until all requested samples have been read or a timeout occurs.
        """
        if samples_per_channel is None:
            samples_per_channel = self.available_samples
        channel_count = len(self.__active_channels)
        data_buffer = np.empty(samples_per_channel * channel_count, dtype=self._constraints.data_type)
        self.read_data_into_buffer(data_buffer=data_buffer, samples_per_channel=samples_per_channel)
        return data_buffer, None

    def read_single_point(self) -> tuple[np.ndarray, None | np.float64]:
        """This method will initiate a single sample read on each configured data channel.
        In general this sample may not be acquired simultaneous for all channels and timing in
        general can not be assured. Us this method if you want to have a non-timing-critical
        snapshot of your current data channel input.
        May not be available for all devices.
        The returned 1D numpy array will contain one sample for each channel.

        In case of SampleTiming.TIMESTAMP a single numpy.float64 timestamp value will be returned
        as well.
        """
        if self.module_state() != 'locked':
            raise RuntimeError('Unable to read data. Device is not running.')

        data_buffer = np.empty(len(self.__active_channels), dtype=self._constraints.data_type)
        self.read_data_into_buffer(data_buffer=data_buffer, samples_per_channel=1)
        return data_buffer, None

    # StartStopInterface
    @configure.overload("StartStopInterface")
    def configure(self, channels: dict[str, StartStopHistogramConfiguration]):
        StartStopInterface.configure(self, channels)
    @property
    def channels(self):
        return AVAILABLE_HISTOGRAMS
    def is_channel_enabled(self, channel: str) -> bool:
        return self._enabled_histograms[channel]
    def enable_channel(self, channel:str, enabled: bool):
        if self.status not in (StartStopHardwareState.IDLE, StartStopHardwareState.UNCONFIGURED):
            self.log.warning("Histogram enabling is only evaluated when acquisition start. Your setting will only take effect on next run.")
        self._enabled_histograms[channel] = enabled
    def channel_configuration(self, channel: str) -> StartStopHistogramConfiguration:
        return self._histogram_configurations[channel]
    def configure_channel(self, channel:str, config: StartStopHistogramConfiguration) -> None:
        self._histogram_configurations[channel] = config
    @property
    def status(self) -> StartStopHardwareState:
        return self._start_stop_status
    def start_acquisition(self):
        if self.tdclib is None:
            raise ValueError("TDC library not initialized.")
        if self._start_stop_status not in (StartStopHardwareState.UNCONFIGURED, StartStopHardwareState.IDLE, StartStopHardwareState.ERROR):
            self.log.error(f"Could not start histogram acquisition from state {self._start_stop_status}.")
            return
        self._histogram_data = {histogram: None for histogram in self.channels}
        enabled_histograms = tuple(histogram for (histogram, enabled) in self._enabled_histograms.items() if enabled)
        if len(enabled_histograms) <= 0:
            return
        current_configuration = self.enabled_configuration
        min_bin_width_ns = min([ configuration.bin_width_ns for configuration in current_configuration.values()])
        max_duration_ns = max([ configuration.histogram_length_ns for configuration in current_configuration.values()])
        err = self.tdclib.TDC_enableStartStop(False)
        self.perror("TDC_enableStartStop", err)
        for possible_histogram in self.channels:
            start,stop = histogram_pair_from_name(possible_histogram)
            present = possible_histogram in enabled_histograms
            err = self.tdclib.TDC_addHistogram(start, stop, present)
            self.perror("TDC_addHistogram", err)
        timebase_s = ctypes.c_double()
        err = self.tdclib.TDC_getTimebase(ctypes.byref(timebase_s))
        self.perror("TDC_getTimebase", err)
        bin_width = int(min_bin_width_ns/(timebase_s.value*1e9))
        actual_bin_width_ns = timebase_s.value * 1e9 * bin_width
        bin_count = int(max_duration_ns / actual_bin_width_ns)
        err = self.tdclib.TDC_setHistogramParams(bin_width, bin_count)
        self.perror("TDC_setHistogramParams", err)
        err = self.tdclib.TDC_freezeBuffers(False)
        self.perror("TDC_freezeBuffers", err)
        err = self.tdclib.TDC_enableStartStop(True)
        self.perror("TDC_enableStartStop", err)
        err = self.tdclib.TDC_enableStartStop(True)
        self.perror("TDC_enableStartStop", err)
        if not err:
            self._start_stop_status = StartStopHardwareState.RUNNING
            self.module_state.lock()
        else:
            self._start_stop_status = StartStopHardwareState.ERROR
        return self._start_stop_status

    def stop_acquisition(self):
        new_status = self.pause_acquisition()
        if new_status is not StartStopHardwareState.ERROR:
            self._start_stop_status = StartStopHardwareState.IDLE
        return self._start_stop_status


    def pause_acquisition(self):
        if self.tdclib is None:
            raise ValueError("TDC library not initialized.")
        if self._start_stop_status not in (StartStopHardwareState.RUNNING, StartStopHardwareState.PAUSED, StartStopHardwareState.ERROR):
            self.log.error(f"Could not pause acquisition from state {self._start_stop_status}.")
            return
        err = self.tdclib.TDC_freezeBuffers(True)
        self.perror("TDC_freezeBuffers", err)
        if not err:
            self._histogram_data = {
                    histogram:self._read_hardware_data(histogram)
                    for histogram in self.channels
                    if self.is_channel_enabled(histogram)
                    }
            self._start_stop_status = StartStopHardwareState.PAUSED
        else:
            self._start_stop_status = StartStopHardwareState.ERROR
        return self._start_stop_status

    def continue_acquisition(self):
        if self.tdclib is None:
            raise ValueError("TDC library not initialized.")
        if self._start_stop_status not in (StartStopHardwareState.PAUSED,):
            self.log.error(f"Could not restart acquisition from state {self._start_stop_status}.")
            return
        err = self.tdclib.TDC_freezeBuffers(False)
        self.perror("TDC_freezeBuffers", err)
        if not err:
            self._start_stop_status = StartStopHardwareState.RUNNING
        else:
            self._start_stop_status = StartStopHardwareState.ERROR
        return self._start_stop_status
        
    def _read_hardware_data(self, channel):
        if self.tdclib is None:
            raise ValueError("TDC library not initialized.")
        start,stop = histogram_pair_from_name(channel)
        bin_count = ctypes.c_int32()
        bin_width = ctypes.c_int32()
        err = self.tdclib.TDC_getHistogramParams(ctypes.byref(bin_width), ctypes.byref(bin_count))
        self.perror("TDC_setHistogramParams", err)
        timebase_s = ctypes.c_double()
        err = self.tdclib.TDC_getTimebase(ctypes.byref(timebase_s))
        self.perror("TDC_getTimebase", err)
        data = np.zeros(bin_count.value, dtype=np.int32)
        count = ctypes.c_int32()
        exposure_time = ctypes.c_int64()
        self.tdclib.TDC_getHistogram(
                start, stop, False,
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 
                ctypes.byref(count), None, None, None, None, ctypes.byref(exposure_time)
            )
        return StartStopHistogram(
                histogram=data,
                total_count=count.value,
                bin_width_ns=timebase_s.value*1e9*bin_width.value,
                exposure_time_s=exposure_time.value*timebase_s.value*1e9
            )

    def histogram_data(self, channel):
        if self._start_stop_status is StartStopHardwareState.RUNNING:
            return self._read_hardware_data(channel)
        else:
            return self._histogram_data[channel]
