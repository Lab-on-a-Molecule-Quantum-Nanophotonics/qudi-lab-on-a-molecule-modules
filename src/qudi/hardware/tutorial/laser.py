import time
import numpy as np
from qudi.interface.excitation_scanner_interface import ExcitationScannerInterface, ExcitationScannerConstraints, ExcitationScanControlVariable, ExcitationScanDataFormat
from qudi.interface.sampled_finite_state_interface import SampledFiniteStateInterface, transition_to, transition_from, state, initial
from qudi.core.statusvariable import StatusVar

from .physical_model import PhysicalModel

class ExcitationScannerTutorial(ExcitationScannerInterface, SampledFiniteStateInterface):
    """
    An excitation laser that is meant to be used for the tutorial.

    Copy and paste configuration example:
    ```yaml
    excitation_scanner_hardware:
        module.Class: 'tutorial.laser.ExcitationScannerTutorial'
        options:
            watchdog_delay: 0.2 # default
    ```
    """
    _exposure_time = StatusVar(default=0.001)
    _repeat_no = StatusVar(default=1)
    _start_frequency = StatusVar(default=-200e9)
    _stop_frequency = StatusVar(default=200e9)
    _step_frequency = StatusVar(default=50e6)
    _buffer = StatusVar(default=np.zeros((0, 4)))
    _repeat = StatusVar(default=np.zeros(0))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._constraints = ExcitationScannerConstraints(
            (1e-5,1), (1,1000), (-200e9,200e9),[
                ExcitationScanControlVariable("Start frequency", (-200e9, 200e9), float, "Hz"),
                ExcitationScanControlVariable("Stop frequency", (-200e9, 200e9), float, "Hz"),
                ExcitationScanControlVariable("Step frequency", (-200e9, 200e9), float, "Hz"),
        ])
        self._timer_start = time.perf_counter()
        self._scan_running = False
        self._current_step = 0
        self._buffer_position = 0
        self._datapoint_per_scan = 0
        self._idle_value = 0.0
        self.model = PhysicalModel(self._idle_value)

    def on_activate(self):
        self.enable_watchdog()
        self.start_watchdog()
    def on_deactivate(self):
        self.disable_watchdog()
    @property
    def scan_running(self):
        return self._scan_running
    @property 
    def state_display(self):
        return self.watchdog_state
    def start_scan(self):
        self._scan_running = True
    def stop_scan(self):
        self._scan_running = False
    @property 
    def constraints(self):
        return self._constraints
    def set_control(self, variable: str, value) -> None:
        "Set a control variable value."
        if not self.constraints.variable_in_range(variable, value):
            raise ValueError(f"Cannot set {variable}={value}")
        if variable == "Start frequency": 
            self._start_frequency = value
        elif variable == "Stop frequency": 
            self._stop_frequency = value
        elif variable == "Step frequency": 
            self._step_frequency = value
    def get_control(self, variable: str):
        "Get a control variable value."
        if variable == "Start frequency": 
            return self._start_frequency
        elif variable == "Stop frequency": 
            return self._stop_frequency
        elif variable == "Step frequency": 
            return self._step_frequency
        else:
            raise ValueError(f"Unknown variable {variable}")
    def get_current_data(self) -> np.ndarray:
        "Return current scan data."
        return self._buffer[:self._buffer_position, :]
    def set_exposure_time(self, time:float) -> None:
        "Set exposure time for one data point."
        self._exposure_time = time
    def set_repeat_number(self, n:int) -> None:
        "Set number of repetition of each segment of the scan."
        self._repeat_no = n
    def set_idle_value(self, n:float) -> None:
        "Set idle value."
        self._idle_value = n
        self.model.idle = n
    def get_exposure_time(self) -> float:
        "Get exposure time for one data point."
        return self._exposure_time
    def get_repeat_number(self) -> int:
        "Get number of repetition of each segment of the scan."
        return 1
    def get_idle_value(self) -> float:
        "Get idle value."
        return self._idle_value
    @property
    def data_format(self) -> ExcitationScanDataFormat:
        "Return the data format used in this implementation of the interface."
        return ExcitationScanDataFormat(
                frequency_column_number=0,
                step_number_column_number=1,
                time_column_number=2,
                data_column_number=[3],
                data_column_unit=["Hz", "", "s", "c"],
                data_column_names=["Frequency", "Step number", "Time", "Count"] 
            )

    # SampledFiniteStateInterface
    @state
    @initial
    @transition_to(("start_idle", "idle"))
    @transition_from(("interrupt_scan", ["prepare_scan", "prepare_step", "wait_ready", "record_scan_step"]))
    def prepare_idle(self):
        self.log.info("Preparing idle.")
        self._scan_running = False
        self.watchdog_event("start_idle")
    @state
    @transition_to(("start_scan", "prepare_scan"))
    def idle(self):
        if self._scan_running:
            self.watchdog_event("start_scan")
    @state
    @transition_to(("start_prepare_step", "prepare_step"))
    def prepare_scan(self):
        self.log.info("Preparing scan.")
        self._current_step = 0
        self._scan_start_time = time.perf_counter()
        self._buffer_position = 0
        # Obviously, in a real setting you'd acquire data in record_scan_step, and
        # not make them up here ;)
        self._datapoint_per_scan = max(int(abs(self._stop_frequency - self._start_frequency) / self._step_frequency), 1)
        freq = np.linspace(self._start_frequency, self._stop_frequency, self._datapoint_per_scan)
        self._buffer = np.zeros((self._datapoint_per_scan*self._repeat_no, 4))
        self._buffer[:, self.frequency_column_number] = np.tile(freq, self._repeat_no)
        self._buffer[:, self.step_number_column_number] = np.repeat(
                range(self._repeat_no), 
                self._datapoint_per_scan
        )
        self._buffer[:, self.time_column_number] = np.linspace(
                start=0, 
                stop=self._repeat_no*self._exposure_time*self._datapoint_per_scan, 
                num=self._repeat_no*self._datapoint_per_scan
            )
        self._buffer[:, 3] = self.model(self._buffer[:, self.frequency_column_number])
        self.watchdog_event("start_prepare_step")
    @state
    @transition_to(("start_wait_ready", "wait_ready"))
    @transition_to(("scan_done", "prepare_idle"))
    def prepare_step(self):
        self.log.info(f"Preparing step {self._current_step + 1}/{self._repeat_no}.")
        self._current_step += 1
        if self._current_step <= self._repeat_no:
            self._timer_start = time.perf_counter()
            self.watchdog_event("start_wait_ready")
        else:
            self.log.info("Scan done!")
            self.watchdog_event("scan_done")
    @state
    @transition_to(("start_scan_step", "record_scan_step"))
    def wait_ready(self):
        if time.perf_counter() - self._timer_start > 0.5:
            self.log.info("Sucessfully wasted 0.5s of your time waiting to be ready!")
            self._timer_start = time.perf_counter()
            self.watchdog_event("start_scan_step")
    @state
    @transition_to(("step_done", "prepare_step"))
    def record_scan_step(self):
        time_elapsed = time.perf_counter() - self._timer_start
        n = int(time_elapsed / self._exposure_time)
        self._buffer_position = min(self._current_step * (self._datapoint_per_scan-1), self._buffer_position + n)
        freq = self._buffer[self._buffer_position, self.frequency_column_number]
        self.model.idle = freq
        self.log.info(f"n={n} time_elapsed={time_elapsed}")
        if n >= self._datapoint_per_scan:
            self.log.info("Step done!")
            self.watchdog_event("step_done")



