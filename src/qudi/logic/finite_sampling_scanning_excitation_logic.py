import time

from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import Mutex
from qudi.core.module import LogicBase
from qudi.interface.excitation_scanner_interface import ExcitationScannerInterface, ExcitationScannerConstraints

from PySide2 import QtCore
from fysom import Fysom
import numpy as np

class FiniteSamplingScanningExcitationLogic(LogicBase, ExcitationScannerInterface):
    _finite_sampling_io = Connector(name='scan_hardware', interface='FiniteSamplingIOInterface')
    _ao = Connector(name='analog_output', interface='ProcessSetpointInterface')

    _maximum_tension = ConfigOption(name="maximum_tension", default=10)
    _minimum_tension = ConfigOption(name="minimum_tension", default=-10)
    _output_channel = ConfigOption(name="output_channel")
    _input_channel = ConfigOption(name="input_channel")
    _chunk_size = ConfigOption(name="chunk_size", default=10)
    _watchdog_delay = ConfigOption(name="watchdog_delay", default=0.2)

    _scan_data = StatusVar(name="scan_data", default=np.empty((0,3)))
    _conversion_factor = StatusVar(name="conversion_factor", default=13.4e6)
    _scan_mini = StatusVar(name="scan_mini", default=-10)
    _scan_maxi = StatusVar(name="scan_maxi", default=10)
    _exposure_time = StatusVar(name="exposure_time", default=1e-2)
    _scan_step_frequency = StatusVar(name="scan_step_frequency", default=10e6)
    _sleep_time_before_scan = StatusVar(name="sleep_time_before_scan", default=2)
    _n_repeat = StatusVar(name="n_repeat", default=1)
    _idle_value = StatusVar(name="idle_value", default=0.0)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._watchdog_state = Fysom({
            "initial": "stopped",
            "events": [
                {"name":"start_idle", "src":"prepare_idle", "dst":"idle"},
                {"name":"start_idle", "src":"stopped", "dst":"prepare_idle"},

                {"name":"start_scan", "src":"idle", "dst":"prepare_scan"},
                {"name":"start_prepare_step", "src":"prepare_scan", "dst":"prepare_step"},
                {"name":"start_wait_first_value", "src":"prepare_step", "dst":"wait_ready"},
                {"name":"start_scan_step", "src":"wait_ready", "dst":"record_scan_step"},
                {"name":"step_done", "src":"record_scan_step", "dst":"prepare_step"},
                {"name":"end_scan", "src":"prepare_step", "dst":"prepare_idle"},

                {"name":"interrupt_scan", "src":["prepare_scan","prepare_step","wait_ready","record_scan_step"], "dst":"prepare_idle"},

                {"name":"stop_watchdog", "src":"*", "dst":"stopped"},
            ],
            # "callbacks":{
            #     "on_start_scan": self._on_start_scan,
            #     "on_prepare_idle": self._on_stop_scan,
            #     "on_step_done": self._on_step_done,
            # }
        })
        self._scanning_states = {"prepare_scan", "prepare_step", "wait_ready", "record_scan_step"}
        self._watchdog_lock = Mutex()
        self._data_lock = Mutex()
        self._constraints = ExcitationScannerConstraints((0,0),(0,0),(0,0),[],[],[],[])
        self._waiting_start = time.perf_counter()
        self._repeat_no = 0
        self._data_row_index = 0
        self._watchdog_timer = QtCore.QTimer(parent=self)

    # Internal utilities
    @property
    def watchdog_state(self):
        with self._watchdog_lock:
            return self._watchdog_state.current
    def watchdog_event(self, event):
        with self._watchdog_lock:
            self._watchdog_state.trigger(event)
    @property 
    def _number_of_samples_per_frame(self):
        step_tension = self._scan_step_frequency / self._conversion_factor
        return round((self._scan_maxi - self._scan_mini) / step_tension)
    def _watchdog(self):
        time_start = time.perf_counter()
        watchdog_state = self.watchdog_state
        if watchdog_state == "prepare_idle": 
            if self._finite_sampling_io().is_running:
                self._finite_sampling_io().stop_buffered_frame()
            self._ao().set_activity_state(self._output_channel, True)
            self._ao().set_setpoint(self._output_channel, self._idle_value)
            self.watchdog_event("start_idle")
        elif watchdog_state == "idle": 
            if not self._ao().get_activity_state(self._output_channel):
                self._ao().set_activity_state(self._output_channel, True)
            if self._ao().get_setpoint(self._output_channel) != self._idle_value:
                self._ao().set_setpoint(self._output_channel, self._idle_value)
        elif watchdog_state == "prepare_scan": 
            with self._data_lock:
                n = self._number_of_samples_per_frame
                self._scan_data = np.zeros((n*self._n_repeat, 3))
                self._scan_data[:,0] = np.tile(np.linspace(start=self._scan_mini, stop=self._scan_maxi, num=n), self._n_repeat)
                self._scan_data[:,2] = np.repeat(range(self._n_repeat), n)
                self._repeat_no = 0
                self._data_row_index = 0
                self._finite_sampling_io().set_sample_rate(1/self._exposure_time)
                self._ni_finite_sampling_io().set_active_channels(
                    input_channels=(self._input_channel,),
                    output_channels=(self._output_channel,)
                    #output_channels = (self._ni_channel_mapping[ax] for ax in axes)
                )

                self._ni_finite_sampling_io().set_output_mode(SamplingOutputMode.JUMP_LIST)
                self._ni_finite_sampling_io().set_frame_data({self._output_channel:np.linspace(start=self._scan_mini, stop=self._scan_maxi, num=n)})
                self.watchdog_event("start_prepare_step")
        elif watchdog_state == "prepare_step": 
            if self._repeat_no > self._n_repeat:
                self.watchdog_event("end_scan")
                self.log.info("Scan done.")
            else:
                self._ao().set_activity_state(self._output_channel, True)
                self._ao().set_setpoint(self._output_channel, self._scan_mini)
                self._waiting_start = time.perf_counter()
                self.watchdog_event("start_wait_first_value")
        elif watchdog_state == "wait_ready": 
            if time_start - self._waiting_start > self.sleep_time_before_scan:
                self._ao().set_activity_state(self._output_channel, False)
                self._finite_sampling_io().start_buffered_frame()
                self.watchdog_event("start_scan_step")
        elif watchdog_state == "record_scan_step": 
            if self._data_row_index >= self._number_of_samples_per_frame:
                self.watchdog_event("step_done")
            elif self._finite_sampling_io().samples_in_buffer < self._chunk_size:
                pass
            else:
                new_data = self._finite_sampling_io().get_buffered_samples()
                i = self._data_row_index
                with self._data_lock:
                    self._scan_data[i:i+len(new_data),1] = new_data
        elif watchdog_state == "stopped": 
            if self._finite_sampling_io().is_running:
                self._finite_sampling_io().stop_buffered_frame()
        time_end = time.perf_counter()
        time_overhead = time_end-time_start
        new_time = max(0, self._watchdog_delay - time_overhead)
        self._watchdog_timer.start(new_time*1000)

    # Activation/De-activation
    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._constraints = ExcitationScannerConstraints(
            exposure_limits=(1e-4,1),
            repeat_limits=(1,100),
            idle_value_limits=(self._minimum_tension*self._conversion_factor, self._maximum_tension*self._conversion_factor),
            control_variables=("Conversion factor", "Minimum tension", "Maximum tension", "Frequency step", "Sleep before scan"),
            control_variable_limits=((0.0, 1e10), (self._minimum_tension, self.maximum_tension), (100e3,1e9), (0.0, 10.0)),
            control_variable_types=(float, float, float, float, float),
            control_variable_units=("Hz/V", "V", "V", "Hz", "s")
        )
        self._watchdog_timer.setSingleShot(True)
        self._watchdog_timer.timeout.connect(self._watchdog, QtCore.Qt.QueuedConnection)

    def on_deactivate(self):
        pass
    @property
    def scan_running(self) -> bool:
        "Return True if a scan can be launched."
        return self.watchdog_state in self._scanning_states
    def start_scan(self) -> None:
        "Start scanning in a non_blocking way."
        if not self.scan_running:
            self.watchdog_event("start_scan")
    def stop_scan(self) -> None:
        "Stop scanning in a non_blocking way."
        if self.scan_running:
            self.watchdog_event("interrupt_scan")
    @property
    def constraints(self) -> ExcitationScannerConstraints:
        "Get the list of control variables for the scanner."
        return self._constraints
    def set_control(self, variable: str, value) -> None:
        "Set a control variable value."
        "", "Minimum tension", "Maximum tension", "Frequency step", "Sleep before scan"
        if not self.constraints.variable_in_range(variable, value):
            raise ValueError(f"Cannot set {variable}={value}")
        if variable == "Conversion factor":
            self._conversion_factor = value
        elif variable == "Minimum tension":
            self._minimum_tension = value
        elif variable == "Maximum tension":
            self._maximum_tension = value
        elif variable == "Frequency step":
            self._scan_step_frequency = value
        elif variable == "Sleep before scan":
            self._sleep_time_before_scan = value
    def get_control(self, variable: str):
        "Get a control variable value."
        if variable == "Conversion factor":
            return self._conversion_factor
        elif variable == "Minimum tension":
            return self._minimum_tension
        elif variable == "Maximum tension":
            return self._maximum_tension
        elif variable == "Frequency step":
            return self._scan_step_frequency
        elif variable == "Sleep before scan":
            return self._sleep_time_before_scan
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

