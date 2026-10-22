from collections.abc import Iterable
import numpy as np

from PySide6 import QtCore
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.core.connector import Connector
from qudi.core.module import LogicBase
from qudi.util.mutex import RecursiveMutex
from qudi.util.datafitting import FitConfigurationsModel, FitContainer

from qudi.interface.start_stop_interface import (
    InputConfiguration,
    StartStopHardwareState,
    StartStopHistogram,
    StartStopHistogramConfiguration,
    StartStopInterface,
)


class StartStopLogic(LogicBase):
    _watchdog_interval_ms = ConfigOption(default=1000)
    _hardware = Connector(name="start_stop", interface=StartStopInterface)
    _beep = ConfigOption(name="beep", default=True)

    _fit_region = StatusVar(name='fit_region', default=[0, 1])
    _fit_config = StatusVar(name='fit_config', default=dict())
    _fit_histogram = StatusVar(default='')
    _notes = StatusVar(name='notes', default="")

    sigNewData = QtCore.Signal(dict)
    sigNewInputsConfiguration = QtCore.Signal(dict)
    sigNewHistogramsConfiguration = QtCore.Signal(dict)
    sigNewStatus = QtCore.Signal(StartStopHardwareState)
    sigAcquisitionRunning = QtCore.Signal()
    sigAcquisitionPaused = QtCore.Signal()
    sigAcquisitionStopped = QtCore.Signal()
    sigFitUpdated = QtCore.Signal(str, object)
    sigFitRegionUpdated = QtCore.Signal(tuple)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._watchdog_active = True
        self._data: dict[str, StartStopHistogram] = {}
        self._lock = RecursiveMutex()
        self._current_status: StartStopHardwareState = StartStopHardwareState.UNCONFIGURED
        self._fit_config_model: FitConfigurationsModel | None = None
        self._fit_container: FitContainer | None = None
        self._fit_results = None
        self._fit_method: str = ''

    def on_activate(self):
        self._watchdog_active = True
        self._data = self._hardware().data
        self._fit_config_model = FitConfigurationsModel(parent=self)
        self._fit_config_model.load_configs(self._fit_config)
        self._fit_container = FitContainer(parent=self, config_model=self._fit_config_model)
        self.fit_region = self._fit_region
        self.start_watchdog()

    def on_deactivate(self):
        self._watchdog_active = False
        if self._fit_config_model is not None:
            self._fit_config = self._fit_config_model.dump_configs()

    @property
    def current_status(self):
        return self._current_status
    @current_status.setter
    def current_status(self, status: StartStopHardwareState):
        self._current_status = status
        self.sigNewStatus.emit(status)

    def start_watchdog(self):
        QtCore.QMetaObject.invokeMethod(self,
                                        '_query_data',
                                        QtCore.Qt.ConnectionType.QueuedConnection)

    @property
    def data(self) -> dict[str, StartStopHistogram]:
        with self._lock:
            return self._data

    @QtCore.Slot()
    def _query_data(self):
        status = self._hardware().status
        if status == StartStopHardwareState.RUNNING:
            with self._lock:
                self._data = self._hardware().data
                self.sigNewData.emit(self._data)
        if status != self.current_status:
            if status == StartStopHardwareState.RUNNING:
                self.sigAcquisitionRunning.emit()
            elif status == StartStopHardwareState.PAUSED:
                self.sigAcquisitionPaused.emit()
            else:
                self.sigAcquisitionStopped.emit()
            self.current_status = status
        if self._watchdog_active:
            QtCore.QTimer.singleShot(self._watchdog_interval_ms, self._query_data)
    
    @property
    def configuration(self) -> dict[str, StartStopHistogramConfiguration]:
        return self._hardware().configuration

    @configuration.setter
    def configuration(self, config:dict[str, StartStopHistogramConfiguration]):
        self._hardware().configure(config)
        self.sigNewHistogramsConfiguration.emit(self._hardware().configuration)

    @property
    def enabled_configuration(self) -> dict[str, StartStopHistogramConfiguration]:
        return self._hardware().enabled_configuration

    @property
    def channels(self) -> Iterable[str]:
        return self._hardware().channels

    @QtCore.Slot()
    def acquire(self):
        if self._hardware().status == StartStopHardwareState.PAUSED:
            self._hardware().continue_acquisition()
        else:
            self._hardware().start_acquisition()

    @QtCore.Slot()
    def pause(self):
        self._hardware().pause_acquisition()

    @QtCore.Slot()
    def stop(self):
        self._hardware().stop_acquisition()

    @property
    def inputs(self) -> Iterable[str]:
        return self._hardware().inputs

    @property
    def input_configurations(self) -> dict[str, InputConfiguration]:
        return self._hardware().input_configurations()

    def configure_inputs(self, configs: dict[str, InputConfiguration]):
        self._hardware().configure_inputs(configs)
        self.sigNewInputsConfiguration.emit(self._hardware().input_configurations())

    ################
    # Fitting things

    @property
    def fit_config_model(self):
        return self._fit_config_model

    @property
    def fit_container(self):
        return self._fit_container

    def do_fit(self, fit_method):
        if fit_method == 'No Fit':
            self.sigFitUpdated.emit('No Fit', None)
            return 'No Fit', None
        if self._fit_histogram not in self._data:
            self.log.error('No data to fit.')
            self.sigFitUpdated.emit('No Fit', None)
            return 'No Fit', None
        data = self._data[self._fit_histogram]
        self.fit_region = self._fit_region
        times = np.linspace(start=0, stop=data.bin_width_ns*data.histogram.shape[0], num=data.histogram.shape[0])
        times = times - times[data.zero]

        roi = (self.fit_region[0] <= times) & (times <= self.fit_region[1])
        
        self.log.debug("Roi built")
        if not roi.any():
            self.log.error('Fit region limited the data to less than two points. Fit not possible.')
            self.sigFitUpdated.emit('No Fit', None)
            return 'No Fit', None
        self.log.debug("Roi selected")

        times = times[roi]
        counts = data.histogram[roi]
        self.log.debug("Data selected")

        try:
            self._fit_method, self._fit_results = self._fit_container.fit_data(fit_method, times, counts)
        except:
            self.log.exception(f'Data fitting failed:\n{traceback.format_exc()}')
            self.sigFitUpdated.emit('No Fit', None)
            return 'No Fit', None

        self.sigFitUpdated.emit(self._fit_method, self._fit_results)
        self.log.debug("done")
        return self._fit_method, self._fit_results

    @property
    def fit_results(self):
        return self._fit_results

    @property
    def fit_method(self):
        return self._fit_method

    @property
    def fit_region(self) -> tuple[float, float]:
        return self._fit_region

    @fit_region.setter
    def fit_region(self, fit_region: tuple[float, float]):
        fit_region = fit_region if fit_region[0] <= fit_region[1] else (fit_region[1], fit_region[0])
        if self._fit_histogram not in self._data:
            self._fit_region = fit_region
            return
        data = self._data[self._fit_histogram]
        times = np.linspace(start=0, stop=data.bin_width_ns*data.histogram.shape[0], num=data.histogram.shape[0])
        times = times - times[data.zero]
        new_region = (max(min(times), fit_region[0]), min(max(times), fit_region[1]))
        self._fit_region = new_region
        self.sigFitRegionUpdated.emit(self._fit_region)
