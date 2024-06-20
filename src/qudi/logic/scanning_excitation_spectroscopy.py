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

class SpectrometerLogic(LogicBase):
    _scan_logic = Connector(name='scan_logic', interface='ScanningProbeLogic')
    _wavelength_channel = ConfigOption(name="wavelength_channel", default="wavelength")
    _count_channel = ConfigOption(name="count_channel", default="APD")

    _spectrum = StatusVar(name='spectrum', default=[None, None])
    _background = StatusVar(name='background', default=None)
    _wavelength = StatusVar(name='wavelength', default=None)
    _constant_acquisition = StatusVar(name='constant_acquisition', default=False)
    _differential_spectrum = StatusVar(name='differential_spectrum', default=False)
    _background_correction = StatusVar(name='background_correction', default=False)
    _fit_region = StatusVar(name='fit_region', default=[0, 1])
    _axis_type_frequency = StatusVar(name='axis_type_frequency', default=False)
    max_repetitions = StatusVar(name='max_repetitions', default=0)
    _fit_config = StatusVar(name='fit_config', default=dict())

    _sig_get_spectrum = QtCore.Signal(bool, bool, bool)
    
    sig_data_updated = QtCore.Signal()
    sig_state_updated = QtCore.Signal()
    sig_fit_updated = QtCore.Signal(str, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.refractive_index_air = 1.00028823
        self.speed_of_light = 2.99792458e8 / self.refractive_index_air
        self._fit_config_model = None
        self._fit_container = None

        self._logic_id = None

        # locking for thread safety
        self._lock = Mutex()

        self._spectrum = None
        self._currently_running_spectrum = None
        self._wavelength = None
        self._stop_acquisition = False
        self._acquisition_running = False
        self._fit_results = None
        self._fit_method = ''
        self._scan_settings = {
            'range': {},
            'resolution': {},
            'frequency': None
        }

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._logic_id = self._scan_logic().module_uuid
        self._scan_logic().sigScanStateChanged.connect(self._update_scan_state)

        self._fit_config_model = FitConfigurationsModel(parent=self)
        self._fit_config_model.load_configs(self._fit_config)
        self._fit_container = FitContainer(parent=self, config_model=self._fit_config_model)
        self.fit_region = self._fit_region

        self._sig_get_spectrum.connect(self.get_spectrum, QtCore.Qt.QueuedConnection)
    
    def on_deactivate(self):
        """ Reverse steps of activation
        """
        self._scan_logic().sigScanStateChanged.disconnect(self._update_scan_state)
        self._sig_get_spectrum.disconnect()
        self._fit_config = self._fit_config_model.dump_configs()

    def stop(self):
        self._stop_acquisition = True

    def run_get_spectrum(self, constant_acquisition=None, differential_spectrum=None, reset=True):
        if constant_acquisition is not None:
            self.constant_acquisition = bool(constant_acquisition)
        if differential_spectrum is not None:
            self.log.warning("Differential spectra are not available in scqnning excitation spectroscopy.")
        self._sig_get_spectrum.emit(self._constant_acquisition, False, reset)

    def get_spectrum(self, constant_acquisition=None, differential_spectrum=None, reset=True):
        if constant_acquisition is not None:
            self.constant_acquisition = bool(constant_acquisition)
        self._stop_acquisition = False

        if reset:
            self._spectrum = None
            self._currently_running_spectrum = None
            self._wavelength = None
            self._repetitions_spectrum = 0

        self._acquisition_running = True
        self.sig_state_updated.emit()

        number_of_datapoints = functools.reduce(operator.mul, self._scan_logic().scan_resolution.values())
        self._currently_running_spectrum = np.zeros((2, number_of_datapoints))
        self._scan_logic().toggle_scan(True, self._scan_logic().scanner_axes, self.module_uuid)
        
        return self.spectrum

    def run_get_background(self, constant_acquisition=None, reset=True):
        self.log.warning("Background acquisition is not available for scanning excitation spectroscopy.")

    @property
    def acquisition_running(self):
        return self._acquisition_running

    @property
    def spectrum(self):
        if self._acquisition_running:
            return np.copy(self._currently_running_spectrum[1,:])
        if self._spectrum is None:
            return None
        return np.copy(self._spectrum)
        #if self._repetitions_spectrum != 0:
        #    data /= self._repetitions_spectrum
        return data

    def get_spectrum_at_x(self, x):
        if self.x_data is None or self.spectrum is None:
            return -1
        if self.axis_type_frequency:
            return np.interp(x, self.x_data[::-1], self.spectrum[::-1])
        else:
            return np.interp(x, self.x_data, self.spectrum)

    @property
    def background(self):
        return np.zeros(np.shape(self.spectrum))

    @property
    def x_data(self):
        if self._acquisition_running:
            w = self._currently_running_spectrum[0,:]
        else:
            w = self._wavelength
        if self._axis_type_frequency:
            if w is not None:
                return self.speed_of_light / w
        else:
            return w

    @property
    def repetitions(self):
        return self._repetitions_spectrum

    @property
    def background_correction(self):
        return False

    @background_correction.setter
    def background_correction(self, value):
        self._background_correction = False
        self.sig_state_updated.emit()
        self.sig_data_updated.emit()

    @property
    def constant_acquisition(self):
        return self._constant_acquisition

    @constant_acquisition.setter
    def constant_acquisition(self, value):
        self._constant_acquisition = bool(value)
        self.sig_state_updated.emit()

    @property
    def differential_spectrum_available(self):
        return False

    @property
    def differential_spectrum(self):
        return False

    @differential_spectrum.setter
    def differential_spectrum(self, value):
        self.sig_state_updated.emit()

    def save_spectrum_data(self, background=False, name_tag='', root_dir=None, parameter=None):
        """ Saves the current spectrum data to a file.

        @param bool background: Whether this is a background spectrum (dark field) or not.
        @param string name_tag: postfix name tag for saved filename.
        @param string root_dir: overwrite the file position in necessary
        @param dict parameter: additional parameters to add to the saved file
        """

        timestamp = datetime.now()

        # write experimental parameters
        parameters = {'acquisition repetitions': self.repetitions,
                      'differential_spectrum'  : self.differential_spectrum,
                      'background_correction'  : self.background_correction,
                      'constant_acquisition'   : self.constant_acquisition,
                      'scan frequency'         : self._scan_settings["frequency"]
                      }
        for ax in self._scan_settings["range"].keys():
            parameters[f"scan axis {ax} range"] = self._scan_settings["range"][ax]
            parameters[f"scan axis {ax} resolution"] = self._scan_settings["resolution"][ax]
        
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

        if self._axis_type_frequency:
            data = [self.x_data * 1e-12, ]
            header = ['Frequency (THz)', ]
        else:
            data = [self.x_data * 1e9, ]
            header = ['Wavelength (nm)', ]

        # prepare the data
        if not background:
            if self.spectrum is None:
                self.log.error('No spectrum to save.')
                return
            data.append(self.spectrum)
            file_label = 'spectrum' + name_tag
        else:
            self.log.warning("Trying to save the background, which is not available in scanning excitation spectroscopy.")
            if self.background is None or self.spectrum is None:
                self.log.error('No background to save.')
                return
            data.append(self.background)
            file_label = 'background' + name_tag

        header.append('Signal')

        if not background:
            # if background correction was on, also save the data without correction
            if self._background_correction:
                self._background_correction = False
                data.append(self.spectrum)
                self._background_correction = True
                header.append('Signal raw')

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
        ax1.set_ylabel('Intensity ({} arb. u.)'.format(prefix))
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
    def axis_type_frequency(self):
        return self._axis_type_frequency

    @axis_type_frequency.setter
    def axis_type_frequency(self, value):
        self._axis_type_frequency = bool(value)
        self._fit_method = 'No Fit'
        self._fit_results = None
        self.fit_region = (0, 1e20)
        self.sig_data_updated.emit()

    @property
    def exposure_time(self):
        return 1/list(self._scan_logic().scan_frequency.values())[0]

    @exposure_time.setter
    def exposure_time(self, value):
        d = {k:float(value) for k in self._scan_logic().scan_frequency.keys()}
        self._scan_logic().set_scan_frequency(d)
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

        if self._axis_type_frequency:
            start = len(self.x_data) - np.searchsorted(self.x_data[::-1], self._fit_region[1], 'left')
            end = len(self.x_data) - np.searchsorted(self.x_data[::-1], self._fit_region[0], 'right')
        else:
            start = np.searchsorted(self.x_data, self._fit_region[0], 'left')
            end = np.searchsorted(self.x_data, self._fit_region[1], 'right')

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

    # Handling of new scan data 
    def _update_scan_state(self, running, data, caller_id):
        # Only treat spectra for scans started by the 
        if caller_id not in (self._logic_id, self.module_uuid):
            #self.log.debug(f"update_scan_state called with caller_id {caller_id}, not in {self._logic_id} {self.module_uuid}")
            return
        with self._lock:
            self._scan_settings = {
                'range': {ax: data.scan_range[i] for i, ax in enumerate(data.scan_axes)},
                'resolution': {ax: data.scan_resolution[i] for i, ax in enumerate(data.scan_axes)},
                'frequency': data.scan_frequency
            }
            wavelengths = data.data[self._wavelength_channel]
            counts = data.data[self._count_channel]
            indices = np.isfinite(wavelengths)
            wavelengths = wavelengths[indices]
            counts = counts[indices]
            sorted_indices = np.argsort(wavelengths)
            wavelengths = wavelengths[sorted_indices]
            counts = counts[sorted_indices]
            self._currently_running_spectrum[0,:len(wavelengths)] = wavelengths * 1e-9
            self._currently_running_spectrum[1,:len(counts)] = counts
            self.sig_data_updated.emit()
            if not running: # When the scan finishes
                self.log.debug("Scan is over.")
                self._repetitions_spectrum += 1
                self._spectrum = self._currently_running_spectrum[1,:]
                self._wavelength = self._currently_running_spectrum[0,:]
                if self._constant_acquisition and not self._stop_acquisition \
                        and (not self.max_repetitions or self._repetitions_spectrum < self.max_repetitions):
                    self.log.debug("Starting a new spectrum.")
                    self.run_get_spectrum(reset=False)
                else:
                    self._acquisition_running = False
                    self.fit_region = self._fit_region
                    self.log.debug("Sending signal state update to notify end of scan.")
                    self.sig_state_updated.emit()