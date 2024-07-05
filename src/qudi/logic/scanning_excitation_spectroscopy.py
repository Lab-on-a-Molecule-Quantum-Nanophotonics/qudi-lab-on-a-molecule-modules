from PySide2 import QtCore
import numpy as np
import functools
import operator
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
from qudi.core.module import LogicBase

from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.datastorage import TextDataStorage
from qudi.util.datafitting import FitContainer, FitConfigurationsModel
from qudi.util.mutex import Mutex

class ScanningExcitationLogic(LogicBase):
    # TODO: add a watchdog for the scanner that updates the data and updates the state
    _scan_logic = Connector(name='scan_logic', interface='ExcitationScannerInterface')

    _spectrum = StatusVar(name='spectrum', default=[None, None, None])
    _fit_region = StatusVar(name='fit_region', default=[0, 1])
    _fit_config = StatusVar(name='fit_config', default=dict())

    _sig_get_spectrum = QtCore.Signal(bool)
    
    sig_data_updated = QtCore.Signal()
    sig_state_updated = QtCore.Signal()
    sig_fit_updated = QtCore.Signal(str, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._fit_config_model = None
        self._fit_container = None

        self._logic_id = None

        # locking for thread safety
        self._lock = Mutex()

        self._stop_acquisition = False
        self._acquisition_running = False
        self._fit_results = None
        self._fit_method = ''

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._fit_config_model = FitConfigurationsModel(parent=self)
        self._fit_config_model.load_configs(self._fit_config)
        self._fit_container = FitContainer(parent=self, config_model=self._fit_config_model)
        self.fit_region = self._fit_region

        self._sig_get_spectrum.connect(self.get_spectrum, QtCore.Qt.QueuedConnection)
    
    def on_deactivate(self):
        """ Reverse steps of activation
        """
        self._sig_get_spectrum.disconnect()
        self._fit_config = self._fit_config_model.dump_configs()

    def stop(self):
        self._stop_acquisition = True

    def run_get_spectrum(self, reset=True):
        self._sig_get_spectrum.emit(reset)

    def get_spectrum(self, reset=True):
        self._stop_acquisition = False
        if reset:
            self._spectrum = [None,None,None]
        self.sig_state_updated.emit()

        self._scan_logic().start_scan()
        
        return self.spectrum

    @property
    def acquisition_running(self):
        return self._scan_logic().scan_running

    @property
    def spectrum(self):
        if self._spectrum[0] is None:
            return None
        return np.copy(self._spectrum[0])

    def get_spectrum_at_x(self, x):
        if self.frequency is None or self.spectrum is None:
            return -1
        return np.interp(x, self.frequency, self.spectrum)

    @property
    def frequency(self):
        if self._spectrum[1] is None:
            return None
        return np.copy(self._spectrum[1])
        
    @property
    def step_number(self):
        if self._spectrum[2] is None:
            return None
        return np.copy(self._spectrum[2])

    @property
    def repetitions(self):
        return self._scan_logic().get_repeat_number()


    def save_spectrum_data(self, background=False, name_tag='', root_dir=None, parameter=None):
        """ Saves the current spectrum data to a file.

        @param bool background: Whether this is a background spectrum (dark field) or not.
        @param string name_tag: postfix name tag for saved filename.
        @param string root_dir: overwrite the file position in necessary
        @param dict parameter: additional parameters to add to the saved file
        """

        timestamp = datetime.now()

        # write experimental parameters
        parameters = {'repetitions': self.repetitions,
                      'exposure' : self.exposure_time,
                      'control_variables' : self._scan_logic().control_dict
                      }
                
        # TODO: report the scanning parameters here.
        if self.fit_method != 'No Fit' and self.fit_results is not None:
            parameters['fit_method'] = self.fit_method
            parameters['fit_results'] = self.fit_results.params
            parameters['fit_region'] = self.fit_region
        if parameter:
            parameters.update(parameter)

        if self.x_data is None:
            self.log.error('No data to save.')
            return

        data = [self.x_data * 1e-12, ]
        header = ['Frequency (THz)', ]
        
        # prepare the data
        if self.spectrum is None:
            self.log.error('No spectrum to save.')
            return
        data.append(self.spectrum)
        file_label = 'spectrum' + name_tag
        
        header.append('Signal')
        
        data.append(self.step_number)
        header.append('Step_Number')

        # save the date to file
        ds = TextDataStorage(root_dir=self.module_default_data_dir if root_dir is None else root_dir)

        file_path, _, _ = ds.save_data(np.array(data).T,
                                       column_headers=header,
                                       metadata=parameters,
                                       nametag=file_label,
                                       timestamp=timestamp,
                                       column_dtypes=[float] * len(header))

        # save the figure into a file
        figure, ax1 = plt.subplots()
        rescale_factor, prefix = self._get_si_scaling(np.max(data[1]))

        ax1.plot(data[0],
                 data[1] / rescale_factor,
                 linestyle=':',
                 linewidth=0.5
                 )

        if self.fit_method != 'No Fit' and self.fit_results is not None:
            if self._axis_type_frequency:
                x_data = self.fit_results.high_res_best_fit[0] * 1e-12
            else:
                x_data = self.fit_results.high_res_best_fit[0] * 1e9

            ax1.plot(x_data,
                     self.fit_results.high_res_best_fit[1] / rescale_factor,
                     linestyle=':',
                     linewidth=0.5
                     )

        ax1.set_xlabel(header[0])
        ax1.set_ylabel('Intensity ({} count)'.format(prefix))
        figure.tight_layout()

        ds.save_thumbnail(figure, file_path=file_path.rsplit('.', 1)[0])

        self.log.debug(f'Spectrum saved to:{file_path}')

    @staticmethod
    def _get_si_scaling(number):

        prefix = ['', 'k', 'M', 'G', 'T', 'P']
        prefix_index = 0
        rescale_factor = 1

        # Rescale spectrum data with SI prefix
        while number / rescale_factor > 1000:
            rescale_factor = rescale_factor * 1000
            prefix_index = prefix_index + 1

        intensity_prefix = prefix[prefix_index]
        return rescale_factor, intensity_prefix

    @property
    def exposure_time(self):
        return self.scan_logic().get_exposure_time()

    @exposure_time.setter
    def exposure_time(self, value):
        self.scan_logic().set_exposure_time(value)
        self.sig_state_updated.emit()

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
            self.sig_fit_updated.emit('No Fit', None)
            return 'No Fit', None

        self.fit_region = self._fit_region
        if self.x_data is None or self.spectrum is None:
            self.log.error('No data to fit.')
            self.sig_fit_updated.emit('No Fit', None)
            return 'No Fit', None

        start = len(self.x_data) - np.searchsorted(self.x_data, self._fit_region[1], 'left')
        end = len(self.x_data) - np.searchsorted(self.x_data, self._fit_region[0], 'right')

        if end - start < 2:
            self.log.error('Fit region limited the data to less than two points. Fit not possible.')
            self.sig_fit_updated.emit('No Fit', None)
            return 'No Fit', None

        x_data = self.x_data[start:end]
        y_data = self.spectrum[start:end]

        try:
            self._fit_method, self._fit_results = self._fit_container.fit_data(fit_method, x_data, y_data)
        except:
            self.log.exception(f'Data fitting failed:\n{traceback.format_exc()}')
            self.sig_fit_updated.emit('No Fit', None)
            return 'No Fit', None

        self.sig_fit_updated.emit(self._fit_method, self._fit_results)
        return self._fit_method, self._fit_results

    @property
    def fit_results(self):
        return self._fit_results

    @property
    def fit_method(self):
        return self._fit_method

    @property
    def fit_region(self):
        return self._fit_region

    @fit_region.setter
    def fit_region(self, fit_region):
        assert len(fit_region) == 2, f'fit_region has to be of length 2 but was {type(fit_region)}'

        if self.x_data is None:
            return
        fit_region = fit_region if fit_region[0] <= fit_region[1] else (fit_region[1], fit_region[0])
        new_region = (max(min(self.x_data), fit_region[0]), min(max(self.x_data), fit_region[1]))
        self._fit_region = new_region
        self.sig_state_updated.emit()
