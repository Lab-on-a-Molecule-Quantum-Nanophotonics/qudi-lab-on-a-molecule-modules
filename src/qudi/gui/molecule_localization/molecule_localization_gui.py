from docutils.nodes import caption
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import Slot
import os

from qudi.core.connector import Connector
import qudi.util.uic as uic
from PySide6 import QtCore, QtGui, QtWidgets
from qudi.core.module import GuiBase
from qudi.logic.molecule_detection_logic import MoleculeDetectionLogic, MoleculeDetectionSettings

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'main_window.ui')
        super().__init__(*args, **kwargs)
        uic.loadUi(ui_file, self)
        self.settings.helpNoiseStatisticsToolButton.setDefaultAction(self.actionHelp_Noise_Statistics)
        self.settings.helpPolygammaInterpolationToolButton.setDefaultAction(self.actionHelp_Polygamma_Interpolation)
        self.settings.helpPSFToolButton.setDefaultAction(self.actionHelp_PSF)
        self.settings.helpMultithreadingtoolButton.setDefaultAction(self.actionHelp_Multithreading)
        self.settings.helpAlgorithmToolButton.setDefaultAction(self.actionHelp_Algorithm)
        self.settings.helpSuperlocalizationToolButton.setDefaultAction(self.actionHelp_Superlocalization)
        self.settings.runNoiseStatisticsToolButton.setDefaultAction(self.actionRun_noise_statistics)
        self.settings.estimatePSFFromSampledFramesToolButton.setDefaultAction(self.actionEstimate_Point_Spread_Function)
        self.settings.runMoleculeDetectionAlgorithmToolButton.setDefaultAction(self.actionRun_molecule_detection_algorithm)
        self.settings.superlocalizeCurrentMoleculeToolButton.setDefaultAction(self.actionSuperlocalize_current_molecule)
        self.settings.superlocalizeAllMoleculesToolButton.setDefaultAction(self.actionSuperlocalize_all_molecules)
        self.estimatePSFFromSampledFramesToolButton.setDefaultAction(self.actionEstimate_Point_Spread_Function)
        self.settings.browsePushButton.clicked.connect(self.actionSelect_tiff.trigger)

HELP_TEXTS = {
        "noise statistics" : """The main algorithm assumes some properties on the noise statistics of the signal, in particular the theta parameter of the gamma distribution of noise (comes from the EMCC gain) and the amplitude of molecule signal. We estimate it by sampling some frames from the signal and fitting the distribution of pixel values.""",
        "polygamma interpolation": """The formula for optimal filtering of a signal from an EMCCD camera involves evaluation of polygamma functions. To make it faster, we interpolate the values, and you can select here how this is done.""",
        "PSF": """To choose the correct dimension and properties of the filter, the algorithm needs to know the PSF. It can be extracted by fitting the data present in the signal.""",
        "multithreading": """The algorithm runs in parallel to avoid loading the whole video at once. These settings allow you to fine tune this behavior.""",
        "algorithm": """This control the main detection algorithm, that relies on an optimal filter and a probability of false positive to detect sigle molecules.""",
        "superlocalization": """In a second step, one can 'superlocalize' the detected molecules by fitting the PSF on the detected positions.""",
}

class MoleculeLocalizationGui(GuiBase):
    _localization_logic = Connector(name="localization_logic", interface=MoleculeDetectionLogic)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw: MainWindow | None = None
    def on_activate(self):
        self._mw = MainWindow()
        self.connect_gui()
        self.show()
    def on_deactivate(self):
        if self._mw is not None:
            self._mw.close()
        self.disconnect_gui()
    def show(self):
        if self._mw is not None:
            self._mw.show()
            self._mw.activateWindow()
            self._mw.raise_()
    def connect_gui(self):
        if self._mw is None:
            return
        self._mw.actionSelect_tiff.triggered.connect(self.select_file)
        self._localization_logic.settings_updated.connect(self.update_settings)
    def disconnect_gui(self):
        if self._mw is None:
            return
        self._mw.actionSelect_tiff.triggered.disconnect(self.select_file)
        self._localization_logic.settings_updated.disconnect(self.update_settings)
    @Slot()
    def select_file(self):
        filename,_selected_filter = QFileDialog.getOpenFileName(
                parent=self._mw,
                caption="Open File",
                dir=os.path.expanduser("~"),
                selectedFilter = "TIFF files (*.tif *.tiff)"
                )
        if not os.path.isfile(filename):
            self.log.warning("Selected file does not appear to exist")
            return
        else:
            self._localization_logic.open_file(filename)
    @Slot(MoleculeDetectionSettings)
    def update_settings(self, settings:MoleculeDetectionSettings):
        if self._mw is None:
            return
        self.settings.fileLineEdit.setText(settings.input_file)
        self.settings.numberOfFramesSampledForGammaFittingSpinBox.setValue(settings.number_of_frames_sampled_for_gamma_fitting)
        self.settings.quantileGammaFitDoubleSpinBox.setValue(settings.quantile_gamma_fit)
        self.settings.quantileMaxEstimateDoubleSpinBox.setValue(settings.quantile_max_estimate)
        self.settings.thetaDoubleSpinBox.setValue(settings.theta)
        self.settings.assumedAmplitudeDoubleSpinBox.setValue(settings.assumed_amplitude)
        self.settings.digammaInterpolationMinimumDoubleSpinBox.setValue(settings.digamma_interpolation_mini)
        self.settings.digammaInterpolationMaximumDoubleSpinBox.setValue(settings.digamma_interpolation_maxi)
        self.settings.digammaInterpolationNumberSpinBox.setValue(settings.digamma_interpolation_num)
        self.settings.trigammaInterpolationMinimumDoubleSpinBox.setValue(settings.trigamma_interpolation_mini)
        self.settings.trigammaInterpolationMaximumDoubleSpinBox.setValue(settings.trigamma_interpolation_maxi)
        self.settings.trigammaInterpolationNumberSpinBox.setValue(settings.trigamma_interpolation_num)
        self.settings.numberOfFramesSampledForPSFEstimatonSpinBox.setValue(settings.number_of_frames_sampled_for_psf_estimation)
        self.settings.lorentzianSafetyMarginDoubleSpinBox
        self.settings.frequencyFWHMDoubleSpinBox
        self.settings.positionStdevDoubleSpinBox
        self.settings.kernelFrequencyLengthDoubleSpinBox
        self.settings.kernelSpatialLengthDoubleSpinBox
        self.settings.framesPerSliceSpinBox
        self.settings.numberOfWorkersSpinBox
        self.settings.probabilityOfFalsePositiveDoubleSpinBox
        self.settings.borderExcludedPixelsSpinBox
