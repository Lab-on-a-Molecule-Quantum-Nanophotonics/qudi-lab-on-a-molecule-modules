import time

from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import Mutex
from qudi.util.enums import SamplingOutputMode
from qudi.interface.excitation_scanner_interface import ExcitationScannerInterface, ExcitationScannerConstraints

from PySide2 import QtCore
from fysom import Fysom
import numpy as np

class FiniteSamplingScanningExcitationInterfuse(ExcitationScannerInterface):
    _finite_sampling_io = Connector(name='sampling_io', interface='FiniteSamplingIOInterface')
    _ldd_switches = Connector(name="ldd_switches", interface="SwitchInterface")
    _ldd_control = Connector(name="ldd_control", interface="ProcessControlInterface")
    _cem_control = Connector(name="cem_control", interface="ProcessControlInterface")
    _fzw_sampling = Connector(name="fzw_sampling", interface="FiniteSamplingInputInterface")

    _input_channel = ConfigOption(name="input_channel", missing="error")
    _output_channel = ConfigOption(name="output_channel", missing="error")
    _chunk_size = ConfigOption(name="chunk_size", default=10)
    _watchdog_delay = ConfigOption(name="watchdog_delay", default=0.2)

    _scan_data = StatusVar(name="scan_data", default=np.empty((0,3)))
    _exposure_time = StatusVar(name="exposure_time", default=1e-2)
    _n_repeat = StatusVar(name="n_repeat", default=1)
    _idle_value = StatusVar(name="idle_value", default=0.0)
    _bias = StatusVar(name='bias', default=33.0)
    _offset = StatusVar(name='offset', default=0.0)
    _span = StatusVar(name='span', default=1)
    _frequency = StatusVar(name='frequency', default=5)
    _interpolate_frequencies = StatusVar(name='interpolate_frequencies', default=True)
    _idle_scan = StatusVar(name='idle_scan', default=False)
    
    _threaded = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._watchdog_state = Fysom({
            "initial": "stopped",
            "events": [
                {"name":"start_idle", "src":"prepare_idle", "dst":"idle"},
                {"name":"start_idle", "src":"stopped", "dst":"prepare_idle"},
                {"name":"start_scan", "src":("idle", "idle_scan"), "dst":"prepare_scan"},
                {"name":"start_prepare_step", "src":"prepare_scan", "dst":"prepare_step"},
                {"name":"start_wait_ready", "src":"prepare_step", "dst":"wait_ready"},
                {"name":"start_scan_step", "src":"wait_ready", "dst":"record_scan_step"},
                {"name":"step_done", "src":"record_scan_step", "dst":"prepare_step"},
                {"name":"end_scan", "src":"prepare_step", "dst":"prepare_idle"},

                {"name":"interrupt_scan", "src":["prepare_scan","prepare_step","wait_ready","record_scan_step"], "dst":"prepare_idle"},
                
                {"name":"start_idle_scan", "src":"idle", "dst":"prepare_idle_scan"},
                {"name":"start_idle_scan", "src":"prepare_idle_scan", "dst":"idle_scan"},
                {"name":"start_idle", "src":"idle_scan", "dst":"prepare_idle"},

                {"name":"stop_watchdog", "src":"*", "dst":"stopped"},
            ],
        })
        self._scanning_states = {"prepare_scan", "prepare_step", "record_scan_step"}
        self._watchdog_lock = Mutex()
        self._data_lock = Mutex()
        self._constraints = ExcitationScannerConstraints((0,0),(0,0),(0,0),[],[],[],[])
        self._waiting_start = time.perf_counter()
        self._idle_scan_start = time.perf_counter()
        self._repeat_no = 0
        self._data_row_index = 0
        self._frequency_row_index = 0
        self._watchdog_timer = QtCore.QTimer(parent=self)
        self._scan_data = np.zeros((0, 3))
        self._measurement_time = np.zeros(0)
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
        return round(1/ (self._exposure_time * self._frequency))
    @property 
    def _number_of_frequencies_per_frame(self):
        return round(1/ (0.1 * self._frequency))
    def _prepare_ramp(self):
        n = self._number_of_samples_per_frame
        ramp_values = np.linspace(start=self._offset-self._span/2, stop=self._offset+self._span/2, num=n)
        self._finite_sampling_io().set_active_channels((self._input_channel,), (self._output_channel,))
        self._finite_sampling_io().set_sample_rate(1/self._exposure_time)
        self._finite_sampling_io().set_output_mode(SamplingOutputMode.JUMP_LIST)
        self._finite_sampling_io().set_frame_data({self._output_channel: ramp_values})
        self._fzw_sampling().set_sample_rate(1/0.1)
        self._fzw_sampling().set_frame_size(n)
    def _start_ramp(self, start_fzw=True):
        if self._finite_sampling_io().samples_in_buffer > 0:
            self._finite_sampling_io().get_buffered_samples()
        self._finite_sampling_io().start_buffered_frame()
        if start_fzw:
            if self._fzw_sampling().samples_in_buffer > 0:
                self._fzw_sampling().stop_buffered_acquisition()
            self._fzw_sampling().start_buffered_acquisition()
    def _stop_ramp(self):
        self._fzw_sampling().stop_buffered_acquisition()
        self._finite_sampling_io().stop_buffered_frame()
    def _watchdog(self):
        try:
            time_start = time.perf_counter()
            watchdog_state = self.watchdog_state
            if watchdog_state == "prepare_idle": 
                self._ldd_switches().set_state("HV,MOD", "EXT")
                self._stop_ramp()
                self.watchdog_event("start_idle")
            elif watchdog_state == "prepare_idle_scan":
                self._idle_scan_start = time.perf_counter()
                self._stop_ramp()
                self._prepare_ramp()
                self._start_ramp(start_fzw=False)
                self.watchdog_event("start_idle_scan")
            elif watchdog_state == "idle_scan":
                if not self._idle_scan:
                    self.watchdog_event("start_idle")
                elif time_start - self._idle_scan_start >= 1/self._frequency:
                    self._idle_scan_start = time.perf_counter()
                    self._stop_ramp()
                    self._prepare_ramp()
                    self._start_ramp(start_fzw=False)
            elif watchdog_state == "idle": 
                if self._idle_scan:
                    self.watchdog_event("start_idle_scan")
            elif watchdog_state == "prepare_scan": 
                n = self._number_of_samples_per_frame
                with self._data_lock:
                    self._scan_data = np.zeros((n*self._n_repeat, 3))
                    self._scan_data[:,2] = np.repeat(range(self._n_repeat), n)
                    self._measurement_time = np.zeros(n*self._n_repeat)
                    self._frequency_buffer = np.zeros(self._number_of_frequencies_per_frame*self._n_repeat)
                self._repeat_no = 0
                self._data_row_index = 0
                self._frequency_row_index = 0
                self._stop_ramp()
                self._prepare_ramp()
                self.log.debug("Scan prepared.")
                self.watchdog_event("start_prepare_step")
            elif watchdog_state == "prepare_step": 
                if self._repeat_no >= self._n_repeat:
                    if self._interpolate_frequencies:
                        self.log.debug("interpolating frequencies.")
                        n = self._scan_data.shape[0]
                        measurements_times = np.linspace(start=0, stop=(n-1)*self._exposure_time, num=n)
                        frequency_times = np.linspace(start=0, stop=(n-1)*self._exposure_time, num=self._frequency_buffer.shape[0])
                        self._scan_data[:,0] = np.interp(measurements_times, 
                                frequency_times, 
                                self._frequency_buffer
                            )
                    self.watchdog_event("end_scan")
                    self.log.info("Scan done.")
                else:
                    self._ldd_control().set_setpoint("bias", self._bias)
                    self.log.debug(f"Step {self._repeat_no} prepared.")
                    self._waiting_start = time.perf_counter()
                    self.watchdog_event("start_wait_ready")
            elif watchdog_state == "wait_ready":
                # if time_start - self._waiting_start > 1:
                #     self._ldd_control().set_setpoint("span", self._span)
                #     self._ldd_control().set_setpoint("offset", self._offset)
                #     self._finite_sampling_io().start_buffered_acquisition()
                #     self._fzw_sampling().start_buffered_acquisition()
                #     self.log.debug("Ready to start acquisition.")
                self._start_ramp(start_fzw=True)
                self.watchdog_event("start_scan_step")
            elif watchdog_state == "record_scan_step": 
                samples_missing_data = self._number_of_samples_per_frame * (self._repeat_no+1) - self._data_row_index
                samples_missing_frequency = self._number_of_frequencies_per_frame * (self._repeat_no+1) - self._frequency_row_index
                if samples_missing_data <= 0 and samples_missing_frequency <= 0:
                    self._repeat_no += 1
                    self._stop_ramp()
                    self.log.debug("Step done.")
                    self.watchdog_event("step_done")
                else:
                    if samples_missing_data > 0 and self._finite_sampling_io().samples_in_buffer >= min(self._chunk_size, samples_missing_data):
                        new_data = self._finite_sampling_io().get_buffered_samples()[self._input_channel]
                        i = self._data_row_index
                        with self._data_lock:
                            self._scan_data[i:i+len(new_data),1] = new_data
                            self._data_row_index += len(new_data)
                    if samples_missing_frequency > 0 and self._fzw_sampling().samples_in_buffer >= min(1, samples_missing_frequency):
                        new_data = self._fzw_sampling().get_buffered_samples()
                        new_frequencies = new_data["frequency"]
                        new_timestamps = new_data["timestamp"]
                        i = self._frequency_row_index
                        data_size = min(self._frequency_buffer.shape[0]-i, len(new_frequencies))
                        new_frequencies = new_frequencies[:data_size]
                        new_timestamps = new_timestamps[:data_size]
                        with self._data_lock:
                            self._frequency_buffer[i:i+len(new_frequencies)] = new_frequencies
                            #self._measurement_time[i:i+len(new_frequencies)] = new_timestamps
                            self._frequency_row_index += len(new_frequencies)
            elif watchdog_state == "stopped": 
                self.log.debug("stopped")
                self._finite_sampling_io().stop_buffered_frame()
                self._fzw_sampling.stop_buffered_acquisition()
            time_end = time.perf_counter()
            time_overhead = time_end-time_start
            new_time = max(0, self._watchdog_delay - time_overhead)
            self._watchdog_timer.start(new_time*1000)
        except:
            self.log.exception("")
    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        cem_constraints = self._cem_control().constraints
        cem_limits = cem_constraints.channel_limits
        cem_units = cem_constraints.channel_units
        ldd_constraints = self._ldd_control().constraints
        ldd_limits = ldd_constraints.channel_limits
        ldd_units = ldd_constraints.channel_units
        self._constraints = ExcitationScannerConstraints(
            exposure_limits=(1e-4,1),
            repeat_limits=(1,2**32-1),
            idle_value_limits=(0.0, 400e12),
            control_variables=("grating", "current", "offset", "span", "bias", "frequency", "interpolate_frequencies", "idle_scan"),
            control_variable_limits=(cem_limits["grating"], ldd_limits["current"], (-10, 10), (0,20), ldd_limits["bias"], ldd_limits["frequency"], (False, True), (False, True)),
            control_variable_types=(int, float, float, float, float, float, bool, bool),
            control_variable_units=(cem_units["grating"], ldd_units["current"], 'V', 'V', ldd_units["bias"], ldd_units["frequency"], None, None)
        )
        self._ldd_control().set_setpoint("bias", self._bias)
        self.watchdog_event("start_idle")
        self._watchdog_timer.setSingleShot(True)
        self._watchdog_timer.timeout.connect(self._watchdog, QtCore.Qt.QueuedConnection)
        self._watchdog_timer.start(self._watchdog_delay)

    def on_deactivate(self):
        self.watchdog_event("stop_watchdog")
        time.sleep(3*self._watchdog_delay)
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
        if not self.constraints.variable_in_range(variable, value):
            raise ValueError(f"Cannot set {variable}={value}")
        if variable == "grating":
            self._cem_control().set_setpoint("grating", value)
        elif variable == "current":
            self._ldd_control().set_setpoint("current", value)
        elif variable == "offset":
            if self.watchdog_state != 'idle' or self._idle_scan:
                self._ldd_control().set_setpoint("offset", value)
            self._offset = value
        elif variable == "span":
            if self.watchdog_state != 'idle' or self._idle_scan:
                self._ldd_control().set_setpoint("span", value)
            self._span = value
        elif variable == "bias":
            if self.watchdog_state != 'idle' or self._idle_scan:
                self._ldd_control().set_setpoint("bias", value)
            self._bias = value
        elif variable == "frequency":
            if self.watchdog_state != 'idle' or self._idle_scan:
                self._ldd_control().set_setpoint("frequency", value)
            self._frequency = value
        elif variable == "interpolate_frequencies":
            self._interpolate_frequencies = bool(value)
        elif variable == 'idle_scan':
            self._idle_scan = bool(value)
                
        
    def get_control(self, variable: str):
        "Get a control variable value."
        if variable == "grating":
            return self._cem_control().get_setpoint("grating")
        elif variable == "current":
            return self._ldd_control().get_setpoint("current")
        elif variable == "offset":
            return self._offset
        elif variable == "span":
            return self._span
        elif variable == "bias":
            if self.watchdog_state != 'idle' or self._idle_scan:
                return self._ldd_control().get_setpoint("bias")
            else:
                return self._bias
        elif variable == "frequency":
            return self._frequency
        elif variable == "interpolate_frequencies":
            return self._interpolate_frequencies
        elif variable == 'idle_scan':
            return self._idle_scan
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
    @property
    def _frequency_span(self):
        return np.min(frequencies),np.max(freqencies)
    def get_idle_value(self) -> float:
        if len(self._scan_data) <= 0:
            return 0.0
        roi = self._scan_data[:,2] == self._scan_data[0,2]
        frequencies = self._scan_data[roi,0]
        return np.interp(self._idle_value, 
                         np.linspace(start=max(self._offset-self._span/2,0.0), stop=min(self._offset+self._span/2, 1.0), num=len(frequencies)), 
                         frequencies
                         )
    def set_idle_value(self, v):
        if len(self._scan_data) <= 0:
            return 0.0
        roi = self._scan_data[:,2] == self._scan_data[0,2]
        frequencies = self._scan_data[roi,0]
        self._idle_value =  np.interp(v, 
                                      frequencies,
                                      np.linspace(start=max(self._offset-self._span/2,0.0), 
                                                  stop=min(self._offset+self._span/2, 1.0), 
                                                  num=len(frequencies)
                                      ))

