import time
from enum import Enum

from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import Mutex
from qudi.interface.excitation_scanner_interface import ExcitationScannerInterface, ExcitationScannerConstraints, ExcitationScanControlVariable, ExcitationScanDataFormat
from qudi.interface.sampled_finite_state_interface import SampledFiniteStateInterface, transition_to, transition_from, state
from qudi.util.network import netobtain

from PySide2 import QtCore
from fysom import Fysom
import numpy as np
from numpy.polynomial import polynomial


class MatisseScanMode(Enum):
    INCREASE_VOLTAGE_STOP_NEITHER = 0
    DECREASE_VOLTAGE_STOP_NEITHER = 1
    INCREASE_VOLTAGE_STOP_LOW = 2
    DECREASE_VOLTAGE_STOP_LOW = 3
    INCREASE_VOLTAGE_STOP_UP = 4
    DECREASE_VOLTAGE_STOP_UP = 5
    INCREASE_VOLTAGE_STOP_EITHER = 6
    DECREASE_VOLTAGE_STOP_EITHER = 7

class RemoteMatisseScanner(ExcitationScannerInterface, SampledFiniteStateInterface):
    """
    A ExcitationScannerInterface implementation to use a Matisse laser.

    Copy and paste example configuration:
    ```yaml
    matisse_scanner:
        module.Class: 'interfuse.remote_matisse_scanner.RemoteMatisseScanner'
        connect:
            input: ni_finite_sampling_input
            matisse: matisse
            matisse_sw: matisse
            wavemeter: wavemeter
        config:
            input_channels:
                - 'pfi0'
                - 'ai0'
            chunk_size: 10 # default
            max_scan_speed: 0.01 # default
    ```
    """
    _finite_sampling_input = Connector(name='input', interface='FiniteSamplingInputInterface')
    "Connector to the finite sampling input. Typically your NI card sampling an APD or a photodiode."
    _matisse = Connector(name='matisse', interface='ProcessControlInterface')
    "A connector to the remote matisse commander proxy."
    _matisse_sw = Connector(name='matisse_sw', interface='SwitchInterface')
    "A connector to the remote matisse commander proxy using another interface to control scanning."
    _wavemeter = Connector(name='wavemeter', interface='DataInStreamInterface')
    "A connector to the remote wavemeter of the matisse."

    _input_channels = ConfigOption(name="input_channels")
    "The channels of the data that are scanned."
    _chunk_size = ConfigOption(name="chunk_size", default=10)
    "Internal control for buffering incoming data."
    _max_scan_speed = ConfigOption(name="max_scan_speed", default=0.01)
    "Limitation on the maximum scan speed."

    _scan_data = StatusVar(name="scan_data", default=np.empty((0,4)))
    "Internal data cache."
    _exposure_time = StatusVar(name="exposure_time", default=1e-2)
    "How long one frequency is exposed."
    _sleep_time_before_scan = StatusVar(name="sleep_time_before_scan", default=60)
    "How long do we wait at maximum before a scan. This is to avoid getting stuck waiting for the laser."
    _n_repeat = StatusVar(name="n_repeat", default=1)
    "How many times do we scan the frequency range."
    _idle_value = StatusVar(name="idle_value", default=0.4)
    "What should be the value while idling."
    _stitching_enabled = StatusVar(name="stitching_enabled", default=False)
    "Should we stitch several scan range to scan more frequencies?"
    _stitching_first_frequency = StatusVar(name="stitching_first_frequency", default=0.0)
    "First central frequency used when stitching."
    _stitching_last_frequency = StatusVar(name="stitching_last_frequency", default=0.0)
    "Last central frequency used when stitching."
    _last_known_scan_parameters = StatusVar(name="last_known_scan_parameters", default={})
    "Caching dict to use when we cannot query the laser."

    _scanning_states = {"prepare_scan", "prepare_step", "wait_ready", "go_to_position", "record_scan_step", "prepare_stitch"}
    "States that are considering as scanning when reporting."
    _initial_state = "prepare_idle"

    _stitch_column_number = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_lock = Mutex()
        self._constraints = ExcitationScannerConstraints((0,0),(0,0),(0,0),{})
        self._waiting_start = time.perf_counter()
        self._repeat_no = 0
        self._data_row_index = 0
        self._conversion_offset = 0.0
        self._scan_start_time = 0
        self._step_start_time = 0
        self._last_center_frequency = 0.0
        self._stitch_no = 0
        self._last_prepared_stitch = -1
        self._stitch_buffer_row_index = 0
        self._scan_up = True

    # Internal data querying utilities
    def _query_laser(self, name, bypass=False):
        active = netobtain(self._matisse().get_activity_state(name))
        if not active and not bypass:
            return self._last_known_scan_parameters.get(name, 0.0)
        v = netobtain(self._matisse().get_setpoint(name))
        self._last_known_scan_parameters[name] = v
        return v
    def _set_laser(self, name, v):
        active = netobtain(self._matisse().get_activity_state(name))
        if not active:
            return 
        self._matisse().set_setpoint(name, v)
    @property
    def _scan_mini(self):
        return self._query_laser("scan lower limit")
    @_scan_mini.setter
    def _scan_mini(self, v):
        self._set_laser("scan lower limit", v)
    @property
    def _scan_maxi(self):
        return self._query_laser("scan upper limit")
    @_scan_maxi.setter
    def _scan_maxi(self, v):
        self._set_laser("scan upper limit", v)
    @property
    def _conversion_factor(self):
        return self._query_laser("conversion factor")
    @_conversion_factor.setter
    def _conversion_factor(self, v):
        self._set_laser("conversion factor", v)
    @property
    def _scan_speed(self):
        return self._query_laser("scan rising speed")
    @_scan_speed.setter
    def _scan_speed(self, value):
        self._set_laser("scan rising speed", value)
    @property
    def _fall_speed(self):
        return self._query_laser("scan falling speed")
    @_fall_speed.setter
    def _fall_speed(self, value):
        self._set_laser("scan falling speed", value)
    @property
    def _scan_value(self):
        return self._query_laser("scan value", bypass=True)
    @_scan_value.setter
    def _scan_value(self, v):
        self._set_laser("scan value", v)
    @property
    def _scanning(self):
        v = self._matisse_sw().get_state("Scan Status")
        return netobtain(v) == "RUN"
    def _set_central_frequency(self, v):
        self._last_center_frequency = v
        self._matisse().set_setpoint("go to target", v)
        self._matisse_sw().set_state("Go to Position", "Running")
    @property
    def _go_to_position_procedure_running(self):
        v = self._matisse_sw().get_state("Go to Position")
        return netobtain(v) == "Running"

    @property 
    def _number_of_samples_per_frame(self):
        return round((self._scan_maxi - self._scan_mini) / self._scan_speed / self._exposure_time)
    @property
    def _number_of_wavemeter_point_per_frame(self):
        return round(self._number_of_samples_per_frame * self._exposure_time * self._wavemeter().sample_rate * 1.5)

    def _init_stitch_buffer(self):
        n = self._number_of_samples_per_frame
        stitch_buffer = np.zeros((n*self._n_repeat, 4 + len(self._input_channels)))
        stitch_buffer[:,self.step_number_column_number] = np.repeat(range(self._n_repeat), n)
        stitch_buffer[:,self.time_column_number] = range(self._n_repeat*n)
        stitch_buffer[:,self._stitch_column_number] = self._stitch_no
        freq = np.linspace(start=self._scan_mini, stop=self._scan_maxi, num=n)*self._conversion_factor + self._conversion_offset
        stitch_buffer[:, self.frequency_column_number] = np.tile(freq, self._n_repeat)
        return stitch_buffer
        
    # SampledFiniteStateInterface
    @state
    @transition_to(("start_idle", "idle"))
    @transition_from(("interrupt", ["prepare_scan", "prepare_step", "wait_ready", "go_to_position", "record_scan_step"]))
    def prepare_idle(self):
        """Prepare idling mode. We stop all acquisitions and stop scanning.
        In case of failure, a warning is issued and we proceed to the idle mode.
        """
        try:
            if self._finite_sampling_input().module_state() == 'locked':
                self._finite_sampling_input().stop_buffered_acquisition()
            if self._matisse_sw().get_state("Scan Status") == "RUN":
                self._matisse_sw().set_state("Scan Status", "STOP")
            if self._go_to_position_procedure_running:
                self._matisse_sw().set_state("Go to Position", "Idle")
        except Exception as e:
            self.log.warn(f"Could not prepare idling: {e}")
        self.watchdog_event("start_idle")
    @state
    @transition_to(("start", "prepare_scan"))
    def idle(self):
        """Idling mode, we wait for a new scan to be triggered, and update the 
        idling value if needed.
        """
        if self._scan_value != self._idle_value:
            self._scan_value = self._idle_value
    @state
    @transition_to(("next", "prepare_stitch"))
    def prepare_scan(self):
        """Prepare a scan, we initialize the internal buffers and status variables.
        """
        n = self._number_of_samples_per_frame
        self.log.debug(f"Preparing scan from {self._scan_mini} to {self._scan_maxi} with {n} points.")
        self._scan_data = np.empty((0, 4 + len(self._input_channels)))

        self._stitch_no = 0
        self._last_prepared_stitch = -1
        self._repeat_no = 0
        self._data_row_index = 0
        try:
            self._finite_sampling_input().set_sample_rate(1/self._exposure_time)
            self._finite_sampling_input().set_frame_size(n)

            self._finite_sampling_input().set_active_channels(self._input_channels)
        except Exception as e:
            self.log.warn(f"Could not prepare the scan: {e}")
            self.watchdog_event("interrupt")
        self._scan_start_time = time.perf_counter()
        self.log.debug("Scan prepared.")
        self.watchdog_event("next")
    @state
    @transition_to(("next", "prepare_step"))
    @transition_to(("go to position", "go_to_position"))
    @transition_to(("end", "prepare_idle"))
    def prepare_stitch(self):
        no_stitch_all_done = not self._stitching_enabled and self._repeat_no >= self._n_repeat
        if no_stitch_all_done or self._last_center_frequency >= self._stitching_last_frequency:
            self.watchdog_event("end")
            self.log.info("All stitches done done.")
        elif self._stitching_enabled and self._last_prepared_stitch != self._stitch_no:
            if self._last_prepared_stitch == -1:
                next_central_frequency = self._stitching_first_frequency
            else:
                next_central_frequency = np.quantile(self.sampled_frequencies, 0.8)
                self._stitch_no += 1
            self._set_central_frequency(next_central_frequency)
            self._waiting_start = time.perf_counter()
            self.watchdog_event("go to position")
        else:
            stitch_buffer = self._init_stitch_buffer()
            self._repeat_no = 0
            self._data_row_index = 0
            with self._data_lock:
                self._stitch_buffer_row_index = self._scan_data.shape[0]
                self._scan_data = np.vstack((self._scan_data, stitch_buffer))
            self.log.info("Stitch prepared.")
            self.watchdog_event("next")
    @state
    @transition_to(("next", "prepare_stitch"))
    def go_to_position(self):
        """Wait for the laser to reach its new central frequency."""
        # We wait at least 1s to be sure.
        if time.perf_counter() - self._waiting_start > 1 and not self._go_to_position_procedure_running:
            self._last_prepared_stitch = self._stitch_no
            self.watchdog_event("next")
    @state
    @transition_to(("next", "wait_ready"))
    @transition_to(("end", "prepare_stitch"))
    def prepare_step(self):
        """Prepare a specific scan step. We first check if we have already run 
        all requested steps, in which cas we go back to preparing idling. Otherwise,
        we initialize the instruments, and move the scanner to its first position.
        """
        if self._repeat_no >= self._n_repeat:
            perm = np.argsort(self.scanner_positions)
            poly = polynomial.polyfit(self.scanner_positions[perm], self.sampled_frequencies[perm], deg=1)
            self.log.debug(f"Fitted polynomial {poly}.")
            if not any(np.isnan(poly)):
                self._conversion_offset, self._conversion_factor = poly
            self._update_conversion()
            self.watchdog_event("end")
            self.log.info("All steps done done.")
            return
        delta_up = self._scan_maxi - self._scan_value
        delta_down = self._scan_mini - self._scan_value
        self._scan_up = np.abs(delta_up) > np.abs(delta_down)
        n = self._number_of_samples_per_frame
        offset = n * self._repeat_no + self._stitch_buffer_row_index
        freq = np.linspace(start=self._scan_mini, stop=self._scan_maxi, num=n)*self._conversion_factor + self._conversion_offset
        if not self._scan_up:
            freq = freq[::-1]
        self._scan_data[offset:(offset+n), self.frequency_column_number] = freq
        try:
            if self._wavemeter().module_state() == 'locked':
                self._wavemeter().stop_stream()
            self._wavemeter().configure(
                active_channels=None,
                streaming_mode=None,
                channel_buffer_size = max(self._number_of_wavemeter_point_per_frame, self._wavemeter().channel_buffer_size),
                sample_rate=None
            )
            if self._finite_sampling_input().module_state() == 'locked':
                self._finite_sampling_input().stop_buffered_acquisition()
            scanning_mode = None
            if self._scan_up:
                if self._scan_value < self._scan_mini:
                    scanning_mode = MatisseScanMode.INCREASE_VOLTAGE_STOP_LOW
                elif self._scan_value > self._scan_mini:
                    scanning_mode = MatisseScanMode.DECREASE_VOLTAGE_STOP_LOW
            else:
                if self._scan_value < self._scan_maxi:
                    scanning_mode = MatisseScanMode.INCREASE_VOLTAGE_STOP_UP
                elif self._scan_value > self._scan_maxi:
                    scanning_mode = MatisseScanMode.DECREASE_VOLTAGE_STOP_UP
            if scanning_mode is not None:
                self.log.debug(f"Moving to first position {scanning_mode}, scan up {self._scan_up}, delta up {delta_up}, delta down {delta_down}")
                self._matisse().set_setpoint("scan mode", scanning_mode.value)
                self._matisse_sw().set_state("Scan Status", "RUN")
            self.log.debug("Step prepared, starting wait.")
            self._waiting_start = time.perf_counter()
            self.watchdog_event("next")
        except Exception as e:
            self.log.warn(f"Could not prepare the step: {e}")
            self.watchdog_event("interrupt")
    @state
    @transition_to(("next", "record_scan_step"))
    def wait_ready(self):
        """Wait for the scanner to reach its first position. There is a time limit
        after which we give up to avoid getting stuck.
        """
        if time.perf_counter() - self._waiting_start > self._sleep_time_before_scan:
                self.log.warn("Time before scan exceeded, giving up.")
                self.watchdog_event("interrupt")
        elif not self._scanning:
            self.log.debug("Ready.")
            try:
                if self._scan_up:
                    self._matisse().set_setpoint("scan mode", MatisseScanMode.INCREASE_VOLTAGE_STOP_UP.value)
                else:
                    self._matisse().set_setpoint("scan mode", MatisseScanMode.DECREASE_VOLTAGE_STOP_LOW.value)
                self._matisse_sw().set_state("Scan Status", "RUN")
                self._finite_sampling_input().start_buffered_acquisition()
                self.watchdog_event("next")
                self._wavemeter().start_stream()
                self._step_start_time = time.perf_counter()
            except Exception as e:
                self.log.warn(f"Could not start the step: {e}")
                self.watchdog_event("interrupt")
    @state
    @transition_to(("next", "prepare_step"))
    def record_scan_step(self):
        """Actual data recording. We collect the available data. When all data
        have been collected, we go back to preparing a step.
        """
        samples_missing = self._number_of_samples_per_frame * (self._repeat_no+1) - self._data_row_index
        if samples_missing <= 0:
            self.sampled_frequencies, _ = netobtain(self._wavemeter().read_data())
            self.sampled_frequencies = netobtain(self.sampled_frequencies)
            self._wavemeter().stop_stream()
            if self._scan_up:
                start = self._scan_mini
                stop = self._scan_maxi
            else:
                start = self._scan_maxi
                stop = self._scan_mini
            self.scanner_positions = np.linspace(start=start, stop=stop, num=len(self.sampled_frequencies))
            n = self._number_of_samples_per_frame
            offset = n * self._repeat_no + self._stitch_buffer_row_index
            perm = np.argsort(self.scanner_positions)
            interp = np.interp(
                np.linspace(start=start, stop=stop, num=n),
                self.scanner_positions[perm],
                self.sampled_frequencies[perm]
            )
            self.log.debug(f"Preparing interpolation. interp={interp.shape}, scndata {self._scan_data[offset:(offset+n),self.frequency_column_number].shape}")
            self._scan_data[offset:(offset+n),self.frequency_column_number] = interp
            self._scan_data[offset:(offset+n),self.time_column_number] = (self._step_start_time - self._scan_start_time) + np.arange(n)*self._exposure_time
            self._repeat_no += 1
            self.log.debug("Step done.")
            self.watchdog_event("next")
        elif self._finite_sampling_input().samples_in_buffer < min(self._chunk_size, samples_missing):
            pass
        else:
            i = self._data_row_index + self._stitch_buffer_row_index
            new_data_length = 0
            with self._data_lock:
                new_data = self._finite_sampling_input().get_buffered_samples()
                for (chnum, ch) in enumerate(self._input_channels):
                    self._scan_data[i:i+len(new_data[ch]),3+chnum] = new_data[ch]
                    new_data_length = len(new_data[ch])
                self._data_row_index += new_data_length
    @state
    @transition_from(("stop", "*"))
    @transition_to(("start_idle", "prepare_idle"))
    def stopped(self):
        """Stop all operations and wait.
        """
        self.log.debug("stopped")
        if self._finite_sampling_input().module_state() == 'locked':
            self._finite_sampling_input().stop_buffered_acquisition()

    # Activation/De-activation
    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        scan_limits = (0.0, 0.7)
        self._constraints = ExcitationScannerConstraints(
            exposure_limits=(1e-4,1),
            repeat_limits=(1,1000),
            idle_value_limits=(0.0, 0.7),
            control_variables_list=[
                ExcitationScanControlVariable("Conversion factor", (0.0, 1e17), float, "Hz"),
                ExcitationScanControlVariable("Conversion offset", (0.0, 1e17), float, "Hz"),
                ExcitationScanControlVariable("Minimum scan", scan_limits, float, ""),
                ExcitationScanControlVariable("Maximum scan", scan_limits, float, ""),
                ExcitationScanControlVariable("Minimum frequency", (0.0, 1e17), float, "Hz"),
                ExcitationScanControlVariable("Maximum frequency", (0.0, 1e17), float, "Hz"),
                ExcitationScanControlVariable("Frequency step", (0.0, 1e17), float, "Hz"),
                ExcitationScanControlVariable("Sleep before scan", (0.0, 3000.0), float, "s"),
                ExcitationScanControlVariable("Idle active", (False, True), bool, ""),
                ExcitationScanControlVariable("Idle value", scan_limits, float, ""),
                ExcitationScanControlVariable("Idle frequency", scan_limits, float, "Hz"),
                ExcitationScanControlVariable("Stitch scans", (False, True), bool, ""),
                ExcitationScanControlVariable("Stitching first central frequency", (0.0, 1e17), float, "Hz"),
                ExcitationScanControlVariable("Stitching last central frequency", (0.0, 1e17), float, "Hz"),
            ]
        )
        number_of_columns = 4 + len(self._input_channels)
        if len(self._scan_data.shape) != 2 or (len(self._scan_data.shape) == 2 and self._scan_data.shape[1] != number_of_columns):
            self.log.debug("Wrong shape of stored data, discarding.")
            self._scan_data = np.empty((0, number_of_columns))
        self.enable_watchdog()
        self.start_watchdog()
        self._wavemeter().start_stream()
        time.sleep(1)
        self._conversion_offset = float(netobtain(self._wavemeter().read_single_point())[0][0])
        self.log.debug(self._conversion_offset)
        self._wavemeter().stop_stream()
        self._update_conversion()

    def on_deactivate(self):
        self.watchdog_event("stop")
        self.disable_watchdog()
    def get_frame_size(self):
        return self._number_of_samples_per_frame
    @property
    def scan_running(self) -> bool:
        "Return True if a scan can be launched."
        return self.watchdog_state in self._scanning_states
    @property
    def state_display(self) -> str:
        return self.watchdog_state.replace("_", " ")
    def start_scan(self) -> None:
        "Start scanning in a non_blocking way."
        if not self.scan_running:
            self.watchdog_event("start")
    def stop_scan(self) -> None:
        "Stop scanning in a non_blocking way."
        if self.scan_running:
            self.watchdog_event("interrupt")
    @property
    def constraints(self) -> ExcitationScannerConstraints:
        "Get the list of control variables for the scanner."
        return self._constraints
    def _update_conversion(self):
        freq_mini = self.get_control("Minimum frequency")
        freq_maxi = self.get_control("Maximum frequency")
        freq_step = self.get_control("Frequency step")
        idle_value = self.get_control("Idle frequency")
        self._scan_mini = max(0.0, (freq_mini-self._conversion_offset)/self._conversion_factor)
        self._scan_maxi = min(0.7, (freq_maxi-self._conversion_offset)/self._conversion_factor)
        self._idle_value = min(0.7, max(0.0, (idle_value-self._conversion_offset)/self._conversion_factor))
        #val_scan = freq_step/self._conversion_factor
        #self._scan_speed = min(val_scan/self._exposure_time, self._max_scan_speed)
        #self._fall_speed = min(10*val_scan/self._exposure_time, self._max_scan_speed)
        lims = (self._conversion_offset, 0.7*self._conversion_factor + self._conversion_offset)
        self._constraints.set_limits("Minimum frequency", *lims)
        self._constraints.set_limits("Maximum frequency", *lims)
        self._constraints.set_limits("Frequency step", 0, lims[1]-self._conversion_offset)
    def set_control(self, variable: str, value) -> None:
        "Set a control variable value."
        if not self.constraints.variable_in_range(variable, value):
            raise ValueError(f"Cannot set {variable}={value}")
        if variable == "Conversion factor":
            self._conversion_factor = value
            self._update_conversion()
        elif variable == "Conversion offset":
            self._conversion_offset = value
            self._update_conversion()
        elif variable == "Minimum scan":
            self._scan_mini = value
        elif variable == "Maximum scan":
            self._scan_maxi = value
        elif variable == "Minimum frequency":
            self.set_control("Minimum scan", (value-self._conversion_offset)/self._conversion_factor)
        elif variable == "Maximum frequency":
            self.set_control("Maximum scan", (value-self._conversion_offset)/self._conversion_factor)
        elif variable == "Frequency step":
            val_scan = value/self._conversion_factor
            val_scan = min(val_scan/self._exposure_time, self._max_scan_speed)
            self._scan_speed = val_scan
            self._fall_speed = val_scan
        elif variable == "Sleep before scan":
            self._sleep_time_before_scan = value
        elif variable == "Idle active":
            self._matisse().set_activity_state("scan value", value)
        elif variable == "Idle value":
            self.log.debug(f"Setting idle value to {value}")
            self._idle_value = value
        elif variable == "Idle active":
            self._matisse().set_activity_state("scan value", value)
        elif variable == "Stitch scans":
            self._stitching_enabled = value
        elif variable == "Stitching first central frequency":
            self._stitching_first_frequency = value
        elif variable == "Stitching last central frequency":
            self._stitching_last_frequency = value
    def get_control(self, variable: str):
        "Get a control variable value."
        if variable == "Conversion factor":
            return self._conversion_factor
        elif variable == "Conversion offset":
            return self._conversion_offset
        elif variable == "Minimum scan":
            return self._scan_mini
        elif variable == "Maximum scan":
            return self._scan_maxi
        elif variable == "Minimum frequency":
            return self._scan_mini*self._conversion_factor + self._conversion_offset
        elif variable == "Maximum frequency":
            return self._scan_maxi*self._conversion_factor + self._conversion_offset
        elif variable == "Frequency step":
            return self._scan_speed*self._exposure_time*self._conversion_factor
        elif variable == "Sleep before scan":
            return self._sleep_time_before_scan
        elif variable == "Idle active":
            v = self._matisse().get_activity_state("scan value")
            return netobtain(v)
        elif variable == "Idle value":
            return self._idle_value
        elif variable == "Idle frequency":
            return self._idle_value*self._conversion_factor + self._conversion_offset
        elif variable == "Stitch scans":
            return self._stitching_enabled
        elif variable == "Stitching first central frequency":
            return self._stitching_first_frequency
        elif variable == "Stitching last central frequency":
            return self._stitching_last_frequency
        else:
            raise ValueError(f"Unknown variable {variable}")
    def get_current_data(self) -> np.ndarray:
        "Return current scan data."
        return self._scan_data
    def set_exposure_time(self, time:float) -> None:
        "Set exposure time for one data point."
        if not self.constraints.exposure_in_range(time):
            raise ValueError(f"Unable to set exposure to {time}")
        self._exposure_time = time

    def set_repeat_number(self, n:int) -> None:
        "Set number of repetition of each segment of the scan."
        if not self.constraints.repeat_in_range(n):
            raise ValueError(f"Unable to set repeat to {n}")
        self._n_repeat = n
    def get_exposure_time(self) -> float:
        "Get exposure time for one data point."
        return self._exposure_time
    def get_repeat_number(self) -> int:
        "Get number of repetition of each segment of the scan."
        return self._n_repeat
    def get_idle_value(self) -> float:
        return self._idle_value * self._conversion_factor + self._conversion_offset
    def set_idle_value(self, n):
        tension = (n-self._conversion_offset) / self._conversion_factor
        if not self.constraints.idle_value_in_range(tension):
            tension=0.0
        self._idle_value = tension
    @property
    def data_format(self) -> ExcitationScanDataFormat:
        "Return the data format used in this implementation of the interface."
        units = self._finite_sampling_input().constraints.channel_units
        return ExcitationScanDataFormat(
                frequency_column_number=0,
                step_number_column_number=1,
                time_column_number=2,
                data_column_number=[i+4 for i in range(len(self._input_channels))],
                data_column_unit=["Hz", "", "s", ""] + [units[ch] for ch in self._input_channels],
                data_column_names=["Frequency", "Step number", "Time", "Stitch number"] + list(self._input_channels)
            )


