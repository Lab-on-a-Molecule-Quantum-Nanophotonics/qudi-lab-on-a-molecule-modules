import time

from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import Mutex
from qudi.util.enums import SamplingOutputMode
from qudi.interface.excitation_scanner_interface import ExcitationScannerInterface, ExcitationScannerConstraints, ExcitationScanControlVariable, ExcitationScanDataFormat
from qudi.interface.sampled_finite_state_interface import SampledFiniteStateInterface, transition_to, transition_from, state, initial

import numpy as np

class FiniteSamplingScanningExcitationInterfuse(ExcitationScannerInterface, SampledFiniteStateInterface):
    """
    An ExcitationScannerInterface to use a FiniteSamplingIOInterface, and an
    analog output, as a scanning excitation module. Typical use case is to 
    have a laser whose frequency is set by the analog output of a NI card, and 
    measure the corresponding count rate on an APD.

    Copy and paste example configuration:
    ```yaml
    finite_sampling_excitation:
        module.Class: 'interfuse.finite_sampling_scanning_excitation.FiniteSamplingScanningExcitationInterfuse'
        connect:
            scan_hardware: ni_finite_sampling_io
            analog_output: ni_analog_output # to control the idle value of the laser
            external_process: cryostation # optional
            external_setpoint: cryostation # optional
        config:
            maximum_tension: 10 # V (default)
            minimum_tension: -10 # V (default)
            minimum_tension_step: 1e-4 # V (default)
            output_channel: 'ao1' # required
            input_channel: 'pfi1' # required
            chunk_size: 10 # default
            process_channel: "" # required if an external process is connected
            setpoint_channel: "" # required if an external setpoint is connected
    ```
    """
    _finite_sampling_io = Connector(name='scan_hardware', interface='FiniteSamplingIOInterface')
    _ao = Connector(name='analog_output', interface='ProcessSetpointInterface')
    _ext_process = Connector(name='external_process', interface='ProcessValueInterface', optional=True)
    _ext_setpoint = Connector(name='external_setpoint', interface='ProcessSetpointInterface', optional=True)

    _maximum_tension = ConfigOption(name="maximum_tension", default=10)
    _minimum_tension = ConfigOption(name="minimum_tension", default=-10)
    _minimum_tension_step = ConfigOption(name="minimum_tension", default=1e-4)
    _output_channel = ConfigOption(name="output_channel")
    _input_channel = ConfigOption(name="input_channel")
    _chunk_size = ConfigOption(name="chunk_size", default=10)
    _process_channel = ConfigOption(name="process_channel", default=None)
    _setpoint_channel = ConfigOption(name="setpoint_channel", default=None)

    _scan_data = StatusVar(name="scan_data", default=np.empty((0,3)))
    _conversion_factor = StatusVar(name="conversion_factor", default=13.1378e9)
    _scan_mini = StatusVar(name="scan_mini", default=-10)
    _scan_maxi = StatusVar(name="scan_maxi", default=10)
    _exposure_time = StatusVar(name="exposure_time", default=1e-2)
    _scan_step_tension = StatusVar(name="scan_step_tension", default=1e-3)
    _sleep_time_before_scan = StatusVar(name="sleep_time_before_scan", default=5)
    _n_repeat = StatusVar(name="n_repeat", default=1)
    _idle_value = StatusVar(name="idle_value", default=0.0)
    _setpoint_first = StatusVar(name="setpoint_first", default=0.0)
    _setpoint_last = StatusVar(name="setpoint_last",  default=0.0)
    _repeat_per_setpoint = StatusVar(name="repeat_per_setpoint", default=1)
    _external_enabled = StatusVar(name="external_enabled", default=False)
    _sleep_time_external_stabilization = StatusVar(name="sleep_time_external_stabilization", default=60)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scanning_states = {"prepare_scan", "prepare_step", "wait_ready", "record_scan_step"}
        self._data_lock = Mutex()
        self._constraints = ExcitationScannerConstraints((0,0),(0,0),(0,0),{})
        self._waiting_start = time.perf_counter()
        self._repeat_no = 0
        self._data_row_index = 0
        self._scan_start_time = 0
        self._step_start_time = 0
        self._setpoint_no = 0
        self._process_begin = 0.0
        self._process_end = 0.0
        self._setpoints = []
        self._sleep_duration = 0.0

    # Internal utilities
    @property 
    def _number_of_samples_per_frame(self):
        return round((self._scan_maxi - self._scan_mini) / self._scan_step_tension)

    # Activation/De-activation
    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        variables = [
                ExcitationScanControlVariable("Conversion factor", (0.0, 1e11), float, "Hz/V"),
                ExcitationScanControlVariable("Minimum tension", (self._minimum_tension, self._maximum_tension), float, "V"),
                ExcitationScanControlVariable("Maximum tension", (self._minimum_tension, self._maximum_tension), float, "V"),
                ExcitationScanControlVariable("Tension step", (self._minimum_tension_step,self._maximum_tension-self._minimum_tension), float, "V"),
                ExcitationScanControlVariable("Minimum frequency", (-10e12, 10e12), float, "Hz"),
                ExcitationScanControlVariable("Maximum frequency", (-10e12, 10e12), float, "Hz"),
                ExcitationScanControlVariable("Frequency step", (0.0, 10e12), float, "Hz"),
                ExcitationScanControlVariable("Sleep before scan", (0.0, 60.0), float, "s"),
                ExcitationScanControlVariable("Idle tension", (self._minimum_tension, self._maximum_tension), float, "V"),
                ExcitationScanControlVariable("Idle frequency", (-10e12, 10e12), float, "Hz"),
                ExcitationScanControlVariable("External enabled", (False, True), bool, ""),
        ]
        if self._ext_setpoint() is not None:
            limits = self._ext_setpoint().constraints.channel_limits[self._setpoint_channel]
            variables.append(ExcitationScanControlVariable("First external setpoint", limits, float, self._ext_setpoint().constraints.channel_units[self._setpoint_channel]),)
            variables.append(ExcitationScanControlVariable("Last external setpoint", limits, float, self._ext_setpoint().constraints.channel_units[self._setpoint_channel]),)
            variables.append(ExcitationScanControlVariable("External setpoint step", (0, abs(limits[0]-limits[1])), float, self._ext_setpoint().constraints.channel_units[self._setpoint_channel]),)
            variables.append(ExcitationScanControlVariable("External setpoint repeat per step", (1, 100), int, ""))
            variables.append(ExcitationScanControlVariable("External setpoint sleep to stabilize", (0, 3000), int, "s"))
        self._constraints = ExcitationScannerConstraints(
            exposure_limits=(1e-4,1),
            repeat_limits=(1,10000),
            idle_value_limits=(self._minimum_tension*self._conversion_factor, self._maximum_tension*self._conversion_factor),
            control_variables_list=variables)
        self.enable_watchdog()
        self.start_watchdog()

    def on_deactivate(self):
        self.watchdog_event("stop")
        self.disable_watchdog()

    # SampledFiniteStateInterface
    @state
    @initial
    @transition_to(("start_idle", "idle"))
    @transition_from(("interrupt", ["prepare_scan", "prepare_step", "wait_ready", "record_scan_step"]))
    def prepare_idle(self):
        "Prepare the hardware for idling: stop acquisition and enable analog output hardware."
        if self._finite_sampling_io().is_running:
            self._finite_sampling_io().stop_buffered_frame()
            self._ao().set_activity_state(self._output_channel, True)
            self._ao().set_setpoint(self._output_channel, self._idle_value)
        self.watchdog_event("start_idle")
    @state
    @transition_to(("start", "prepare_scan"))
    def idle(self):
        "Idling between scans, setting the analog output value if requested."
        if not self._ao().get_activity_state(self._output_channel):
            self._ao().set_activity_state(self._output_channel, True)
        if self._ao().get_setpoint(self._output_channel) != self._idle_value:
            self._ao().set_setpoint(self._output_channel, self._idle_value)
    @state
    @transition_to(("next", "prepare_step"))
    def prepare_scan(self):
        """Prepare a scan, we initialize the internal buffers and status variables.
        """
        n = self._number_of_samples_per_frame
        self.log.debug(f"Preparing scan from {self._scan_mini} to {self._scan_maxi} with {n} points.")
        ncols = 4
        if self._external_enabled and self._ext_setpoint() is not None:
            ncols += 1
        if self._external_enabled and self._ext_process() is not None:
            ncols += 1
        with self._data_lock:
            self._scan_data = np.zeros((n*self._n_repeat, ncols))
            self._scan_data[:,self.frequency_column_number] = np.tile(np.linspace(start=self._scan_mini, stop=self._scan_maxi, num=n), self._n_repeat)*self._conversion_factor
            self._scan_data[:,self.step_number_column_number] = np.repeat(range(self._n_repeat), n)
        self._setpoint_no = 0
        if self._external_enabled and self._ext_setpoint() is not None:
            self._setpoints = np.linspace(start=self._setpoint_first, stop=self._setpoint_last, num = self._n_repeat // self._repeat_per_setpoint)
            self._ext_setpoint().set_activity_state(self._setpoint_channel, True)
        if self._external_enabled and self._ext_process() is not None:
            self._ext_process().set_activity_state(self._process_channel, True)
        self._repeat_no = 0
        self._data_row_index = 0
        self._finite_sampling_io().set_sample_rate(1/self._exposure_time)
        self._finite_sampling_io().set_active_channels(
            input_channels=(self._input_channel,),
            output_channels=(self._output_channel,)
        )
        self._finite_sampling_io().set_output_mode(SamplingOutputMode.JUMP_LIST)
        self._finite_sampling_io().set_frame_data({self._output_channel:np.linspace(start=self._scan_mini, stop=self._scan_maxi, num=n)})
        self._scan_start_time = time.perf_counter()
        self.log.debug("Scan prepared.")
        self.watchdog_event("next")
    @state
    @transition_to(("next", "wait_ready"))
    @transition_to(("end", "prepare_idle"))
    def prepare_step(self):
        """Prepare a specific scan step. We first check if we have already run 
        all requested steps, in which cas we go back to preparing idling. Otherwise,
        we initialize the instruments, and move the scanner to its first position.
        """
        if self._repeat_no >= self._n_repeat:
            self.watchdog_event("end")
            self.log.info("Scan done.")
        else:
            self.log.debug(f"Preparing step {self._repeat_no}")
            if self._finite_sampling_io().is_running:
                self._finite_sampling_io().stop_buffered_frame()
            n = self._number_of_samples_per_frame
            self._ao().set_activity_state(self._output_channel, True)
            self._ao().set_setpoint(self._output_channel, self._scan_mini)
            self._sleep_duration = self._sleep_time_before_scan
            if self._external_enabled and self._ext_setpoint() is not None and self._repeat_no % self._repeat_per_setpoint == 0:
                self.log.debug(f"Setting external setpoint.")
                self._ext_setpoint().set_setpoint(self._setpoint_channel, self._setpoints[self._setpoint_no])
                self._sleep_duration = self._sleep_time_external_stabilization
            self.log.debug("Step prepared, starting wait.")
            self._waiting_start = time.perf_counter()
            self.watchdog_event("next")
    @state
    @transition_to(("next", "record_scan_step"))
    def wait_ready(self):
        """Wait a bit for the laser to reach its first value."""
        if time.perf_counter() - self._waiting_start > self._sleep_duration:
            self.log.debug("Ready.")
            self._ao().set_activity_state(self._output_channel, False)
            if self._external_enabled and self._ext_process() is not None:
                self._process_begin = self._ext_process().get_process_value(self._process_channel)
            self._finite_sampling_io().start_buffered_frame()
            self._step_start_time = time.perf_counter()
            self.watchdog_event("next")
    @state
    @transition_to(("next", "prepare_step"))
    def record_scan_step(self):
        """Actual data recording. We collect the available data. When all data
        have been collected, we go back to preparing a step.
        """
        samples_missing = self._number_of_samples_per_frame * (self._repeat_no+1) - self._data_row_index
        if samples_missing <= 0: 
            n = self._number_of_samples_per_frame
            offset = n * self._repeat_no
            ncols = 4
            if self._external_enabled and self._ext_setpoint() is not None:
                ncols += 1
                self._scan_data[offset:(offset+n),4] = self._ext_setpoint().get_setpoint(self._setpoint_channel)
                if self._repeat_no % self._repeat_per_setpoint == self._repeat_per_setpoint-1:
                    self._setpoint_no = min(len(self._setpoints)-1, self._setpoint_no+1)
            if self._external_enabled and self._ext_process() is not None:
                ncols += 1
                self._process_end = self._ext_process().get_process_value(self._process_channel)
                self._scan_data[offset:(offset+n),ncols-1] = np.linspace(start=self._process_begin, stop=self._process_end, num=n)
            self._scan_data[offset:(offset+n),self.time_column_number] = (self._step_start_time - self._scan_start_time) + np.arange(n)*self._exposure_time
            self._repeat_no += 1
            self.log.debug("Step done.")
            self.watchdog_event("next")
        elif self._finite_sampling_io().samples_in_buffer < min(self._chunk_size, samples_missing):
            pass
        else:
            new_data = self._finite_sampling_io().get_buffered_samples()[self._input_channel]
            i = self._data_row_index
            with self._data_lock:
                self._scan_data[i:i+len(new_data),1] = new_data
                self._data_row_index += len(new_data)
    @state
    @transition_from(("stop", "*"))
    @transition_to(("start_idle", "prepare_idle"))
    def stopped(self):
        """Stop all operations and wait.
        """
        self.log.debug("stopped")
        if self._finite_sampling_io().is_running:
            self._finite_sampling_io().stop_buffered_frame()

    # ExcitationScannerInterface
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
    def set_control(self, variable: str, value) -> None:
        "Set a control variable value."
        if not self.constraints.variable_in_range(variable, value):
            raise ValueError(f"Cannot set {variable}={value}")
        if variable == "Conversion factor":
            freq_mini = self.get_control("Minimum frequency")
            freq_maxi = self.get_control("Maximum frequency")
            freq_step = self.get_control("Frequency step")
            idle_value = self.get_control("Idle frequency")
            self._conversion_factor = value
            self._scan_mini = max(self._minimum_tension, freq_mini/self._conversion_factor)
            self._scan_maxi = min(self._maximum_tension, freq_maxi/self._conversion_factor)
            self._scan_step_tension = max(self._minimum_tension_step, freq_step/self._conversion_factor)
            self._idle_value = min(self._maximum_tension, max(self._minimum_tension, idle_value/self._conversion_factor))
            lims = (self._minimum_tension/self._conversion_factor, self._maximum_tension/self._conversion_factor)
            self._constraints.set_limits("Minimum frequency", *lims)
            self._constraints.set_limits("Maximum frequency", *lims)
            self._constraints.set_limits("Frequency step", 0, lims[1])
        elif variable == "Minimum tension":
            self._scan_mini = value
        elif variable == "Maximum tension":
            self._scan_maxi = value
        elif variable == "Tension step":
            self._scan_step_tension = value
        elif variable == "Minimum frequency":
            self.set_control("Minimum tension", value/self._conversion_factor)
        elif variable == "Maximum frequency":
            self.set_control("Maximum tension", value/self._conversion_factor)
        elif variable == "Frequency step":
            self.set_control("Tension step", value/self._conversion_factor)
        elif variable == "Sleep before scan":
            self._sleep_time_before_scan = value
        elif variable == "Idle tension":
            self.log.debug(f"Setting idle tension to {value}")
            self._idle_value = value
        elif variable == "Idle frequency":
            self.log.debug(f"Setting idle frequency to {value}")
            self.set_control("Idle tension", value/self._conversion_factor)
        elif variable == "External setpoint repeat per step":
            self._repeat_per_setpoint = max(int(value), 1)
        elif variable == "External setpoint step":
            number_of_setpoints = abs(self._setpoint_first - self._setpoint_last) / value
            repeat_per_setpoint = self._n_repeat // number_of_setpoints
            self.set_control("External setpoint repeat per step", repeat_per_setpoint)
        elif variable == "Last external setpoint":
            self._setpoint_last = value
        elif variable == "First external setpoint":
            self._setpoint_first = value
        elif variable == "External enabled":
            self._external_enabled = value
        elif variable == "External setpoint sleep to stabilize":
            self._sleep_time_external_stabilization = value
            
    def get_control(self, variable: str):
        "Get a control variable value."
        if variable == "Conversion factor":
            return self._conversion_factor
        elif variable == "Minimum tension":
            return self._scan_mini
        elif variable == "Maximum tension":
            return self._scan_maxi
        elif variable == "Tension step":
            return self._scan_step_tension
        elif variable == "Minimum frequency":
            return self._scan_mini*self._conversion_factor
        elif variable == "Maximum frequency":
            return self._scan_maxi*self._conversion_factor
        elif variable == "Frequency step":
            return self._scan_step_tension*self._conversion_factor
        elif variable == "Sleep before scan":
            return self._sleep_time_before_scan
        elif variable == "Idle tension":
            return self._idle_value
        elif variable == "Idle frequency":
            return self._idle_value*self._conversion_factor
        elif variable == "External setpoint repeat per step":
            return self._repeat_per_setpoint
        elif variable == "External setpoint step":
            repeat_per_setpoint = self.get_control("External setpoint repeat per step")
            number_of_setpoints = max(self._n_repeat // repeat_per_setpoint, 1)
            return abs(self._setpoint_first - self._setpoint_last) / number_of_setpoints
        elif variable == "Last external setpoint":
            return self._setpoint_last
        elif variable == "First external setpoint":
            return self._setpoint_first
        elif variable == "External enabled":
            return self._external_enabled
        elif variable == "External setpoint sleep to stabilize":
            return self._sleep_time_external_stabilization
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
        return self._idle_value * self._conversion_factor
    def set_idle_value(self, n):
        tension = n / self._conversion_factor
        if not self.constraints.idle_value_in_range(tension):
            raise ValueError(f"Unable to set idle value to {n}")
        self._idle_value = tension
    @property
    def data_format(self) -> ExcitationScanDataFormat:
        "Return the data format used in this implementation of the interface."
        units = self._finite_sampling_io().constraints.input_channel_units
        data_column_number = [1]
        data_column_unit=["Hz", units[self._input_channel], "", "s"]
        data_column_names=["Frequency", self._input_channel, "Step number", "Time"] 
        if self._external_enabled and self._ext_setpoint() is not None:
            data_column_number.append(max(data_column_number)+1)
            data_column_unit.append(self._ext_setpoint().constraints.channel_units[self._setpoint_channel])
            data_column_names.append(self._setpoint_channel)
        if self._external_enabled and self._ext_process() is not None:
            data_column_number.append(max(data_column_number)+1)
            data_column_unit.append(self._ext_process().constraints.channel_units[self._process_channel])
            data_column_names.append(self._process_channel)
        return ExcitationScanDataFormat(
                frequency_column_number=0,
                step_number_column_number=2,
                time_column_number=3,
                data_column_number=data_column_number,
                data_column_unit=data_column_unit,
                data_column_names=data_column_names 
            )

