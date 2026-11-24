import os
import sys

import numpy as np
import pyqtgraph
from lmfit import Parameter
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QMessageBox
from qudi.core.connector import Connector
from qudi.core.module import GuiBase
from qudi.core.statusvariable import StatusVar
from qudi.util.colordefs import QudiPalette
from qudi.util.fit_models.lorentzian import Lorentzian

from qudi.logic.molecule_detection_logic import (
    DatasetUpdate,
    DatasetUpdateKind,
    MoleculeDetectionLogic,
    MoleculeDetectionSettings,
    SingleMoleculeLocalizationDataset,
    UserActionableMoleculeDetectionSettings,
)
from qudi.util import uic


class MainWindow(QtWidgets.QMainWindow):
    sigDeleteMolecules = Signal(list)

    def __init__(self, *args, **kwargs):
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'main_window.ui')
        super().__init__(*args, **kwargs)
        uic.loadUi(ui_file, self)
        self.settings.helpNoiseStatisticsToolButton.setDefaultAction(self.actionHelp_Noise_Statistics)
        self.settings.helpPSFToolButton.setDefaultAction(self.actionHelp_PSF)
        self.settings.helpMultithreadingtoolButton.setDefaultAction(self.actionHelp_Multithreading)
        self.settings.helpAlgorithmToolButton.setDefaultAction(self.actionHelp_Algorithm)
        self.settings.helpSuperlocalizationToolButton.setDefaultAction(self.actionHelp_Superlocalization)
        self.settings.runNoiseStatisticsToolButton.setDefaultAction(self.actionRun_noise_statistics)
        self.settings.estimatePSFFromSampledFramesToolButton.setDefaultAction(self.actionEstimate_Point_Spread_Function)
        self.settings.runMoleculeDetectionAlgorithmToolButton.setDefaultAction(
            self.actionRun_molecule_detection_algorithm
        )
        self.settings.superlocalizeCurrentMoleculeToolButton.setDefaultAction(self.actionSuperlocalize_current_molecule)
        self.settings.superlocalizeAllMoleculesToolButton.setDefaultAction(self.actionSuperlocalize_all_molecules)
        self.settings.estimatePSFFromSampledFramesToolButton.setDefaultAction(self.actionEstimate_Point_Spread_Function)
        self.settings.browsePushButton.clicked.connect(self.actionSelect_tiff.trigger)
        self.parameters_estimation.set_units(x="counts", y="bins")
        self.psf_frequency.set_units(x="Hz", y="counts")
        psf_image_view = self.psf_image.getView()
        self.psf_target_item = pyqtgraph.TargetItem(label=True, movable=False, labelOpts={'color': QudiPalette.c3})
        psf_image_view.addItem(self.psf_target_item)
        self.progress_bar = QtWidgets.QProgressBar()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.results_scatter_item = pyqtgraph.ScatterPlotItem(pen=QudiPalette.c3)
        self.results_scatter_item.setSymbol('+')
        results_image_view = self.resultsImageView.getView()
        results_image_view.addItem(self.results_scatter_item)
        single_molecule_image_view = self.singleMoleculeImageView.getView()
        self.single_molecule_target_item = pyqtgraph.TargetItem(
            label=True, movable=False, labelOpts={'color': QudiPalette.c3}
        )
        single_molecule_image_view.addItem(self.single_molecule_target_item)
        self.singeMoleculeInteractiveCurvesWidget.set_units(x="Hz", y="counts")
        self.settings.probabilityOfFalsePositiveDoubleSpinBox.setMinimum(sys.float_info.epsilon)
        self.inhomogeneousInteractiveCurvesWidget.set_units(x="Hz", y="bins")
        self.inhomogeneousInteractiveCurvesWidget.set_labels(x="Peak Frequency")
        self.linewidthStatisticsInteractiveCurvesWidget.set_units(x="Hz", y="bins")
        self.linewidthStatisticsInteractiveCurvesWidget.set_labels(x="HWHM")
        self.inhomogeneous_broadening_vertical_line = pyqtgraph.InfiniteLine(pen=QudiPalette.green, label="selected molecule")
        self.inhomogeneousInteractiveCurvesWidget._plot_widget.addItem(self.inhomogeneous_broadening_vertical_line)
        self.linewidth_vertical_line = pyqtgraph.InfiniteLine(pen=QudiPalette.green, label="selected molecule")
        self.linewidthStatisticsInteractiveCurvesWidget._plot_widget.addItem(self.linewidth_vertical_line)

    @property
    def table_widgets(self):
        return [
                self.resultsTableWidget,
                self.inhomogeneousTableWidget,
                self.linewidthStatisticsTableWidget
                ]

    @property
    def currently_focused_table(self):
        for table in self.table_widgets:
            if table.hasFocus():
                return table
        return None

    def keyPressEvent(self, event):
        # https://doc.qt.io/qtforpython/PySide6/QtCore/Qt.html?highlight=key_escape#PySide6.QtCore.PySide6.QtCore.Qt.Key
        if event.key() == Qt.Key.Key_Delete:
            widget = self.currently_focused_table
            if widget is None:
                return
            to_be_deleted = [i.row() for i in widget.selectedIndexes()]
            self.sigDeleteMolecules.emit(to_be_deleted)


HELP_TEXTS = {
    "noise statistics": """The main algorithm assumes some properties on the noise statistics of the signal, in particular the theta parameter of the gamma distribution of noise (comes from the EMCC gain) and the amplitude of molecule signal. We estimate it by sampling some frames from the signal and fitting the distribution of pixel values.""",
    "psf": """To choose the correct dimension and properties of the filter, the algorithm needs to know the PSF. It can be extracted by fitting the data present in the signal.""",
    "multithreading": """The algorithm runs in parallel to avoid loading the whole video at once. These settings allow you to fine tune this behavior.""",
    "algorithm": """This control the main detection algorithm, that relies on an optimal filter and a probability of false positive to detect sigle molecules.""",
    "superlocalization": """In a second step, one can 'superlocalize' the detected molecules by fitting the PSF on the detected positions.""",
}


class MoleculeLocalizationGui(GuiBase):
    _localization_logic = Connector(name="localization_logic", interface=MoleculeDetectionLogic)
    _last_directory = StatusVar(default="~")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw: MainWindow | None = None

    def on_activate(self):
        self._mw = MainWindow()
        self.update_settings(self._localization_logic().settings)
        self._mw.metadata.setData(self._localization_logic().andor_metadata)
        # self._mw.flattened_image.setImage(self._localization_logic().flattened_image)
        # self._mw.movie.setImage(self._localization_logic().movie)
        self._mw.settings.readSamplePositionPushButton.setEnabled(self._localization_logic().read_position_available)
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
        self._mw.actionHelp_Noise_Statistics.triggered.connect(self.show_statistics_help_dialog)
        self._mw.actionHelp_PSF.triggered.connect(self.show_psf_help_dialog)
        self._mw.actionHelp_Multithreading.triggered.connect(self.show_multithreading_help_dialog)
        self._mw.actionHelp_Algorithm.triggered.connect(self.show_algorithm_help_dialog)
        self._mw.actionHelp_Superlocalization.triggered.connect(self.show_superlocalization_help_dialog)
        self._mw.actionRead_File.triggered.connect(self._localization_logic().reload_file)
        self._mw.actionRun_noise_statistics.triggered.connect(self.launch_parameters_estimation)
        self._mw.actionEstimate_Point_Spread_Function.triggered.connect(self.launch_psf_estimation)
        self._mw.actionRun_molecule_detection_algorithm.triggered.connect(self.launch_algorithm)
        self._mw.actionSuperlocalize_current_molecule.triggered.connect(self.on_superlocalize_current_molecule)
        self._mw.actionSuperlocalize_all_molecules.triggered.connect(self._localization_logic().superlocalize_all)
        self._mw.actionSave_parameter_estimation.triggered.connect(self.on_save_parameter_estimation)
        self._mw.actionSave_molecule_positions.triggered.connect(self.on_save_molecule_positions)
        self._mw.actionSave_superlocalization.triggered.connect(self.on_save_superlocalization)
        self._mw.tabWidget.currentChanged.connect(self.on_tab_change)
        self._mw.showAllMoleculesCheckBox.clicked.connect(self.on_plot_all_molecules_clicked)
        self._mw.resultsTableWidget.currentCellChanged.connect(self.on_table_result_cell_changed)
        self._mw.settings.readSamplePositionPushButton.clicked.connect(self._localization_logic().read_sample_position)
        self._mw.settings.updateCalibrationPushButton.clicked.connect(self.update_calibration)
        self._mw.sigDeleteMolecules.connect(self.on_delete_molecules_by_id)
        self._mw.inhomogeneousTableWidget.currentCellChanged.connect(self.on_table_inhomogeneous_broadening_cell_changed)
        self._mw.linewidthStatisticsTableWidget.currentCellChanged.connect(self.on_table_linewidth_statistics_cell_changed)
        self._localization_logic().metadata_updated.connect(self._mw.metadata.setData)
        self._localization_logic().flattened_image_updated.connect(self._mw.flattened_image.setImage)
        self._localization_logic().settings_updated.connect(self.update_settings)
        self._localization_logic().gamma_estimate_histogram_updated.connect(self.set_histogram)
        self._localization_logic().gamma_law_fit_updated.connect(self.set_gamma_fit)
        self._localization_logic().psf_updated.connect(self.on_new_psf_available)
        self._localization_logic().progress_update.connect(self._mw.progress_bar.setValue)
        self._localization_logic().results_updated.connect(self.on_results_updated)
        self._localization_logic().results_updated.connect(self.plot_molecules_positions)
        self._localization_logic().finished.connect(self.on_localization_finished)
        self._localization_logic().new_movie.connect(self._mw.movie.setImage)
        self._localization_logic().new_inhomogeneous_broadening_histogram.connect(self.on_new_inhomogeneous_broadening_histogram)
        self._localization_logic().new_linewidth_histogram.connect(self.on_new_linewidth_histogram)
        self._localization_logic().new_inhomogeneous_statistics.connect(self.on_new_inhomogeneous_broadening_statistics)
        self._localization_logic().new_linewidth_statistics.connect(self.on_new_linewidth_statistics)

    def disconnect_gui(self):
        if self._mw is None:
            return
        self._mw.actionSelect_tiff.triggered.disconnect(self.select_file)
        self._mw.actionHelp_Noise_Statistics.triggered.disconnect(self.show_statistics_help_dialog)
        self._mw.actionHelp_PSF.triggered.disconnect(self.show_psf_help_dialog)
        self._mw.actionHelp_Multithreading.triggered.disconnect(self.show_multithreading_help_dialog)
        self._mw.actionHelp_Algorithm.triggered.disconnect(self.show_algorithm_help_dialog)
        self._mw.actionHelp_Superlocalization.triggered.disconnect(self.show_superlocalization_help_dialog)
        self._mw.actionRead_File.triggered.disconnect(self._localization_logic().reload_file)
        self._mw.tabWidget.currentChanged.disconnect(self.on_tab_change)
        self._mw.actionRun_noise_statistics.triggered.disconnect(self.launch_parameters_estimation)
        self._mw.actionEstimate_Point_Spread_Function.triggered.disconnect(self.launch_psf_estimation)
        self._mw.actionRun_molecule_detection_algorithm.triggered.disconnect(self.launch_algorithm)
        self._mw.actionSuperlocalize_current_molecule.triggered.disconnect(self.on_superlocalize_current_molecule)
        self._mw.actionSuperlocalize_all_molecules.triggered.disconnect(self._localization_logic().superlocalize_all)
        self._mw.actionSave_parameter_estimation.triggered.disconnect(self.on_save_parameter_estimation)
        self._mw.actionSave_molecule_positions.triggered.disconnect(self.on_save_molecule_positions)
        self._mw.actionSave_superlocalization.triggered.disconnect(self.on_save_superlocalization)
        self._mw.showAllMoleculesCheckBox.clicked.disconnect(self.on_plot_all_molecules_clicked)
        self._mw.resultsTableWidget.currentCellChanged.disconnect(self.on_table_result_cell_changed)
        self._mw.settings.readSamplePositionPushButton.clicked.disconnect(
            self._localization_logic().read_sample_position
        )
        self._mw.settings.updateCalibrationPushButton.clicked.disconnect(self.update_calibration)
        self._mw.sigDeleteMolecules.disconnect(self.on_delete_molecules_by_id)
        self._mw.inhomogeneousTableWidget.currentCellChanged.connect(self.on_table_inhomogeneous_broadening_cell_changed)
        self._mw.linewidthStatisticsTableWidget.currentCellChanged.connect(self.on_table_linewidth_statistics_cell_changed)
        self._localization_logic().metadata_updated.disconnect(self._mw.metadata.setData)

        self._localization_logic().flattened_image_updated.disconnect(self._mw.flattened_image.setImage)
        self._localization_logic().settings_updated.disconnect(self.update_settings)
        self._localization_logic().gamma_estimate_histogram_updated.disconnect(self.set_histogram)
        self._localization_logic().gamma_law_fit_updated.disconnect(self.set_gamma_fit)
        self._localization_logic().psf_updated.disconnect(self.on_new_psf_available)
        self._localization_logic().progress_update.disconnect(self._mw.progress_bar.setValue)
        self._localization_logic().results_updated.disconnect(self.on_results_updated)
        self._localization_logic().results_updated.connect(self.plot_molecules_positions)
        self._localization_logic().finished.disconnect(self.on_localization_finished)
        self._localization_logic().new_movie.disconnect(self._mw.movie.setImage)
        self._localization_logic().new_inhomogeneous_broadening_histogram.disconnect(self.on_new_inhomogeneous_broadening_histogram)
        self._localization_logic().new_linewidth_histogram.disconnect(self.on_new_linewidth_histogram)
        self._localization_logic().new_inhomogeneous_statistics.disconnect(self.on_new_inhomogeneous_broadening_statistics)
        self._localization_logic().new_linewidth_statistics.disconnect(self.on_new_linewidth_statistics)

    @Slot()
    def select_file(self):
        filename, _selected_filter = QFileDialog.getOpenFileName(
            parent=self._mw,
            caption="Open File",
            dir=os.path.expanduser(self._last_directory),
            filter="TIFF files (*.tif *.tiff)",
        )
        if not os.path.isfile(filename):
            self.log.warning("Selected file does not appear to exist")
            return
        else:
            self._last_directory = os.path.dirname(filename)
            self._localization_logic().open_file(filename)

    @Slot(MoleculeDetectionSettings)
    def update_settings(self, settings: MoleculeDetectionSettings):
        if self._mw is None:
            return
        self._mw.settings.fileLineEdit.setText(settings.input_file)
        self._mw.settings.pixelSizeDoubleSpinBox.setValue(settings.pixel_physical_dimension)
        self._mw.settings.pixelOffsetXDoubleSpinBox.setValue(settings.pixels_offset[0])
        self._mw.settings.pixelOffsetYDoubleSpinBox.setValue(settings.pixels_offset[1])
        self._mw.settings.samplePositionXDoubleSpinBox.setValue(settings.sample_position[0])
        self._mw.settings.samplePositionYDoubleSpinBox.setValue(settings.sample_position[1])
        self._mw.settings.frequencyStepDoubleSpinBox.setValue(settings.frequency_step)
        self._mw.settings.frequencyOffsetDoubleSpinBox.setValue(settings.frequency_offset)
        self._mw.settings.numberOfFramesSampledForGammaFittingSpinBox.setValue(
            settings.number_of_frames_sampled_for_gamma_fitting
        )
        self._mw.settings.quantileGammaFitDoubleSpinBox.setValue(settings.quantile_gamma_fit)
        self._mw.settings.quantileMaxEstimateDoubleSpinBox.setValue(settings.quantile_max_estimate)
        self._mw.settings.thetaDoubleSpinBox.setValue(settings.theta)
        self._mw.settings.assumedAmplitudeDoubleSpinBox.setValue(settings.assumed_amplitude)
        self._mw.settings.numberOfFramesSampledForPSFEstimatonSpinBox.setValue(
            settings.number_of_frames_sampled_for_psf_estimation
        )
        self._mw.settings.lorentzianSafetyMarginDoubleSpinBox.setValue(settings.lorentzian_safety_margin)
        self._mw.settings.frequencyFWHMDoubleSpinBox.setValue(settings.frequency_fwhm)
        self._mw.settings.frequencyFWHMHzDoubleSpinBox.setValue(
            self._localization_logic().scale_frames_to_frequency(settings.frequency_fwhm)
        )
        self._mw.settings.positionStdevDoubleSpinBox.setValue(settings.position_stdev)
        self._mw.settings.positionStdevMDoubleSpinBox.setValue(
            float(self._localization_logic().scale_pixels_to_positions(settings.position_stdev))
        )
        self._mw.settings.kernelFrequencyLengthDoubleSpinBox.setValue(settings.kernel_frequency_length)
        self._mw.settings.kernelSpatialLengthDoubleSpinBox.setValue(settings.kernel_spatial_length)
        self._mw.settings.framesPerSliceSpinBox.setValue(settings.frame_per_slice)
        self._mw.settings.numberOfWorkersSpinBox.setValue(settings.number_of_workers)
        self._mw.settings.probabilityOfFalsePositiveDoubleSpinBox.setValue(settings.probability_false_positive)
        self._mw.settings.borderExcludedPixelsSpinBox.setValue(settings.border_excluded_pixels)

    @Slot()
    def save_settings(self):
        self._localization_logic().settings = UserActionableMoleculeDetectionSettings(
            number_of_frames_sampled_for_gamma_fitting=self._mw.settings.numberOfFramesSampledForGammaFittingSpinBox.value(),
            pixel_physical_dimension=self._mw.settings.pixelSizeDoubleSpinBox.value(),
            pixels_offset=(
                self._mw.settings.pixelOffsetXDoubleSpinBox.value(),
                self._mw.settings.pixelOffsetYDoubleSpinBox.value(),
            ),
            sample_position=(
                self._mw.settings.samplePositionXDoubleSpinBox.value(),
                self._mw.settings.samplePositionYDoubleSpinBox.value(),
            ),
            frequency_step=self._mw.settings.frequencyStepDoubleSpinBox.value(),
            frequency_offset=self._mw.settings.frequencyOffsetDoubleSpinBox.value(),
            quantile_gamma_fit=self._mw.settings.quantileGammaFitDoubleSpinBox.value(),
            quantile_max_estimate=self._mw.settings.quantileMaxEstimateDoubleSpinBox.value(),
            number_of_frames_sampled_for_psf_estimation=self._mw.settings.numberOfFramesSampledForPSFEstimatonSpinBox.value(),
            lorentzian_safety_margin=self._mw.settings.lorentzianSafetyMarginDoubleSpinBox.value(),
            frame_per_slice=self._mw.settings.framesPerSliceSpinBox.value(),
            number_of_workers=self._mw.settings.numberOfWorkersSpinBox.value(),
            probability_false_positive=self._mw.settings.probabilityOfFalsePositiveDoubleSpinBox.value(),
            border_excluded_pixels=self._mw.settings.borderExcludedPixelsSpinBox.value(),
        )

    def show_help_dialog(self, which: str):
        if self._mw is None:
            return
        msg = HELP_TEXTS[which.lower()]
        QMessageBox.information(self._mw, which.title(), msg)

    @Slot()
    def show_statistics_help_dialog(self):
        self.show_help_dialog("noise statistics")

    @Slot()
    def show_psf_help_dialog(self):
        self.show_help_dialog("psf")

    @Slot()
    def show_multithreading_help_dialog(self):
        self.show_help_dialog("multithreading")

    @Slot()
    def show_algorithm_help_dialog(self):
        self.show_help_dialog("algorithm")

    @Slot()
    def show_superlocalization_help_dialog(self):
        self.show_help_dialog("superlocalization")

    @Slot(np.ndarray, np.ndarray)
    def set_histogram(self, hist, edges):
        if "Histogram" not in self._mw.parameters_estimation.plot_names:
            self._mw.parameters_estimation.plot("Histogram", x=edges[:-1], y=hist, pen=QudiPalette.c1)
        else:
            self._mw.parameters_estimation.set_data("Histogram", x=edges[:-1], y=hist)
        self.switch_to_parameters_estimation()

    @Slot(np.ndarray, np.ndarray)
    def set_gamma_fit(self, x, y):
        if "Fit" not in self._mw.parameters_estimation.plot_names:
            self._mw.parameters_estimation.plot("Fit", x=x, y=y, pen=QudiPalette.c2)
        else:
            self._mw.parameters_estimation.set_data("Fit", x=x, y=y)
        self.switch_to_parameters_estimation()

    def switch_to_tab(self, widget):
        if self._mw is None:
            return
        index = self._mw.tabWidget.indexOf(widget)
        self._mw.tabWidget.setCurrentIndex(index)

    @Slot()
    def switch_to_movie(self):
        self.switch_to_tab(self._mw.movieTab)

    @Slot()
    def switch_to_metadata(self):
        self.switch_to_tab(self._mw.metadataTab)

    @Slot()
    def switch_to_flattened(self):
        self.switch_to_tab(self._mw.flattenedImageTab)

    @Slot()
    def switch_to_parameters_estimation(self):
        self.switch_to_tab(self._mw.parametersEstimationTab)

    @Slot()
    def switch_to_psf(self):
        self.switch_to_tab(self._mw.PSFTab)

    @Slot()
    def switch_to_results(self):
        self.switch_to_tab(self._mw.resultsTab)

    def show_toolbox(self, widget):
        if self._mw is None:
            return
        index = self._mw.settings.toolBox.indexOf(widget)
        self._mw.settings.toolBox.setCurrentIndex(index)

    @Slot()
    def show_parameters_estimation_settings(self):
        self.show_toolbox(self._mw.settings.parametersEstimationPage)

    @Slot()
    def show_polygamma_settings(self):
        self.show_toolbox(self._mw.settings.polygammaPage)

    @Slot()
    def show_psf_settings(self):
        self.show_toolbox(self._mw.settings.PSFPage)

    @Slot()
    def show_multithreading_settings(self):
        self.show_toolbox(self._mw.settings.multithreadingPage)

    @Slot()
    def show_algorithm_settings(self):
        self.show_toolbox(self._mw.settings.algorithmPage)

    @Slot()
    def show_superlocalization_settings(self):
        self.show_toolbox(self._mw.settings.superlocalizationPage)

    @Slot(int)
    def on_tab_change(self, _index):
        if self._mw is None:
            return
        widget = self._mw.tabWidget.currentWidget()
        if widget == self._mw.parametersEstimationTab:
            self.show_parameters_estimation_settings()
        elif widget == self._mw.PSFTab:
            self.show_psf_settings()
        elif widget == self._mw.resultsTab:
            self.show_algorithm_settings()

    @Slot()
    def launch_parameters_estimation(self):
        self._localization_logic().settings = UserActionableMoleculeDetectionSettings(
            number_of_frames_sampled_for_gamma_fitting=self._mw.settings.numberOfFramesSampledForGammaFittingSpinBox.value(),
            quantile_gamma_fit=self._mw.settings.quantileGammaFitDoubleSpinBox.value(),
            quantile_max_estimate=self._mw.settings.quantileMaxEstimateDoubleSpinBox.value(),
        )
        self._localization_logic().estimate_from_noise()
        self._mw.statusBar().showMessage("Estimating statistical parameters...")

    @Slot()
    def launch_psf_estimation(self):
        self._localization_logic().settings = UserActionableMoleculeDetectionSettings(
            pixel_physical_dimension=self._mw.settings.pixelSizeDoubleSpinBox.value(),
            pixels_offset=(
                self._mw.settings.pixelOffsetXDoubleSpinBox.value(),
                self._mw.settings.pixelOffsetYDoubleSpinBox.value(),
            ),
            sample_position=(
                self._mw.settings.samplePositionXDoubleSpinBox.value(),
                self._mw.settings.samplePositionYDoubleSpinBox.value(),
            ),
            frequency_step=self._mw.settings.frequencyStepDoubleSpinBox.value(),
            frequency_offset=self._mw.settings.frequencyOffsetDoubleSpinBox.value(),
            number_of_frames_sampled_for_psf_estimation=self._mw.settings.numberOfFramesSampledForPSFEstimatonSpinBox.value(),
            lorentzian_safety_margin=self._mw.settings.lorentzianSafetyMarginDoubleSpinBox.value(),
        )
        self._localization_logic().estimate_psf()
        self._mw.statusBar().showMessage("Estimating PSF...")

    @Slot()
    def launch_algorithm(self):
        self._localization_logic().settings = UserActionableMoleculeDetectionSettings(
            pixel_physical_dimension=self._mw.settings.pixelSizeDoubleSpinBox.value(),
            pixels_offset=(
                self._mw.settings.pixelOffsetXDoubleSpinBox.value(),
                self._mw.settings.pixelOffsetYDoubleSpinBox.value(),
            ),
            sample_position=(
                self._mw.settings.samplePositionXDoubleSpinBox.value(),
                self._mw.settings.samplePositionYDoubleSpinBox.value(),
            ),
            frequency_step=self._mw.settings.frequencyStepDoubleSpinBox.value(),
            frequency_offset=self._mw.settings.frequencyOffsetDoubleSpinBox.value(),
            lorentzian_safety_margin=self._mw.settings.lorentzianSafetyMarginDoubleSpinBox.value(),
            frame_per_slice=self._mw.settings.framesPerSliceSpinBox.value(),
            number_of_workers=self._mw.settings.numberOfWorkersSpinBox.value(),
            probability_false_positive=self._mw.settings.probabilityOfFalsePositiveDoubleSpinBox.value(),
            border_excluded_pixels=self._mw.settings.borderExcludedPixelsSpinBox.value(),
        )
        self._mw.progress_bar.reset()
        self._mw.statusBar().showMessage("Localizing molecules...")
        self._localization_logic().launch_algorithm()

    @Slot(SingleMoleculeLocalizationDataset)
    def on_new_psf_available(self, data: SingleMoleculeLocalizationDataset):
        if "Data" not in self._mw.psf_frequency.plot_names:
            self._mw.psf_frequency.plot(
                "Data", x=data.frequency_spectrum[0], y=data.frequency_spectrum[1], pen=QudiPalette.c1
            )
        else:
            self._mw.psf_frequency.set_data("Data", x=data.frequency_spectrum[0], y=data.frequency_spectrum[1])
        if data.fit_result_spectrally is not None:
            model = Lorentzian()
            param_dict = {k: Parameter(k, v) for (k, v) in data.fit_result_spectrally.items()}
            xs = np.linspace(start=data.frequency_spectrum[0].min(), stop=data.frequency_spectrum[0].max(), num=100)
            ys = model.eval(params=param_dict, x=xs)
            if "Fit" not in self._mw.psf_frequency.plot_names:
                self._mw.psf_frequency.plot("Fit", x=xs, y=ys, pen=QudiPalette.c2)
            else:
                self._mw.psf_frequency.set_data("Fit", x=xs, y=ys)
        extent_m = data.flattened_image_m[-1] - data.flattened_image_m[0]
        extent_n = data.flattened_image_n[-1] - data.flattened_image_n[0]
        self._mw.psf_image.setImage(
            data.flattened_image, pos=(data.flattened_image_m[0], data.flattened_image_n[0]), scale=(extent_m, extent_n)
        )
        if data.fit_result_spatially is not None:
            self._mw.psf_target_item.setPos(
                (data.fit_result_spatially['center_x'], data.fit_result_spatially['center_y'])
            )
        self.switch_to_psf()
        self._mw.statusBar().showMessage("PSF updated")

    @Slot(DatasetUpdate)
    def on_results_updated(self, update: DatasetUpdate):
        if update.kind == DatasetUpdateKind.AddedMultipleRows:
            data = update.dataset
            data_dict = {
                'Frequency (Hz)': data.coordinates_array[0, :],
                'x (m)': data.coordinates_array[1, :],
                'y (m)': data.coordinates_array[2, :],
            }
            if data.lorentzian_fit_array is not None:
                data_dict['HWHM (Hz)'] = data.lorentzian_fit_array['sigma']
                data_dict['amplitude (counts)'] = data.lorentzian_fit_array['amplitude']
            sent_data = [{k: data_dict[k][i] for k in data_dict} for i in range(data.coordinates_array.shape[1])]
            for table in self._mw.table_widgets:
                table.setData(sent_data)
            offset = np.asarray(data.algorithm_settings.pixels_offset) + np.asarray(
                data.algorithm_settings.sample_position
            )

            self._mw.resultsImageView.setImage(
                self._localization_logic().flattened_image,
                scale=(
                    data.algorithm_settings.pixel_physical_dimension,
                    data.algorithm_settings.pixel_physical_dimension,
                ),
                pos=offset,
            )
            self.switch_to_results()
        elif update.kind == DatasetUpdateKind.DeletedASingleRow:
            i = update.index
            for table in self._mw.table_widgets:
                current_row = table.currentRow()
                if current_row == i and current_row < table.rowCount() - 1:
                    table.setCurrentCell(current_row + 1, table.currentColumn())
                elif current_row == i and current_row >= table.rowCount() - 1:
                    table.setCurrentCell(current_row - 1, table.currentColumn())
                table.removeRow(i)
            self._mw.statusBar().showMessage(f"Deleted molecule {i}")
        elif update.kind == DatasetUpdateKind.DeletedManyRows:
            indices = update.index
            if indices is None:
                return
            if type(indices) is int:
                indices = [indices]
            for table in self._mw.table_widgets:
                old_state = table.currentCellChanged.blockSignals(True)
                indices = sorted(indices, reverse=True)
                for i in indices:  # ty: ignore[not-iterable]
                    table.setRangeSelected(
                        QtWidgets.QTableWidgetSelectionRange(i, 0, i, table.columnCount()), False
                    )
                    table.removeRow(i)
                table.currentCellChanged.blockSignals(old_state)
            self._mw.statusBar().showMessage(f"Deleted molecules {indices}")
        elif update.kind == DatasetUpdateKind.SuperlocalizedASingleRow:
            i = update.index
            row = list(update.dataset.coordinates_array[:, i])
            for table in self._mw.table_widgets:
                if table.columnCount() < 5:
                    table.setColumnCount(5)
                    table.setHorizontalHeaderLabels(
                        ['Frequency (Hz)', 'x (m)', 'y (m)', 'HWHM (Hz)', 'amplitude (counts)']
                    )
                if update.dataset.lorentzian_fit_array is not None:
                    row.append(update.dataset.lorentzian_fit_array['sigma'][i])
                    row.append(update.dataset.lorentzian_fit_array['amplitude'][i])
                table.setRow(i, row)
            self._mw.statusBar().showMessage(f"Superlocalized molecule {i}")

    @Slot()
    def on_plot_all_molecules_clicked(self):
        self.plot_molecules_positions()

    @Slot(DatasetUpdate)
    def plot_molecules_positions(self, update: DatasetUpdate | None = None):
        if update is not None and type(update) != bool:
            last_result = update.dataset
            append_data = update.kind is DatasetUpdateKind.AddedMultipleRows
        else:
            last_result = self._localization_logic().last_result
            append_data = False
        if last_result is None:
            self._mw.results_scatter_item.clear()
            return
        if self._mw.showAllMoleculesCheckBox.isChecked():
            dataset = last_result.coordinates_array[1:, :]
            if append_data:
                self._mw.results_scatter_item.addPoints(pos=dataset.transpose())
            else:
                self._mw.results_scatter_item.setData(pos=dataset.transpose())
        else:
            currently_selected = np.clip(
                self._mw.resultsTableWidget.currentRow(), a_min=-1, a_max=last_result.coordinates_array.shape[1] - 1
            )
            if currently_selected < 0:
                self._mw.results_scatter_item.clear()
            else:
                self._mw.results_scatter_item.setData(
                    x=[last_result.coordinates_array[1, currently_selected]],
                    y=[last_result.coordinates_array[2, currently_selected]],
                )

    @Slot(int, int, int, int)
    def on_table_result_cell_changed(self, x: int, _y=int | None, _prevx=int | None, _prevy=int | None):
        if x < 0:
            return
        if not self._mw.showAllMoleculesCheckBox.isChecked():
            self.plot_molecules_positions()
        try:
            last_result = self._localization_logic().last_result
            if last_result is None:
                return
            currently_selected = np.clip(x, a_min=0, a_max=last_result.coordinates_array.shape[1] - 1)
            data = self._localization_logic().get_data_by_molecule_id(currently_selected)
        except ValueError:
            self.log.warning("failed to display single molecule data.")
        else:
            self.display_single_molecule_data(data)

    @Slot(int, int, int, int)
    def on_table_inhomogeneous_broadening_cell_changed(self, x: int, _y=int | None, _prevx=int | None, _prevy=int | None):
        if x < 0:
            return
        last_result = self._localization_logic().last_result
        if last_result is None:
            return
        currently_selected = np.clip(x, a_min=0, a_max=last_result.coordinates_array.shape[1] - 1)
        frequency = last_result.coordinates_array[0, currently_selected]
        self._mw.inhomogeneous_broadening_vertical_line.setValue(frequency)

    @Slot(int, int, int, int)
    def on_table_linewidth_statistics_cell_changed(self, x: int, _y=int | None, _prevx=int | None, _prevy=int | None):
        if x < 0:
            return
        last_result = self._localization_logic().last_result
        if last_result is None:
            return
        currently_selected = np.clip(x, a_min=0, a_max=last_result.coordinates_array.shape[1] - 1)
        if last_result.lorentzian_fit_array is None:
            return
        linewidth = last_result.lorentzian_fit_array["sigma"]
        self._mw.linewidth_vertical_line.setValue(linewidth)

    @Slot()
    def on_localization_finished(self):
        self._mw.statusBar().showMessage("Localization finished!")

    @Slot(SingleMoleculeLocalizationDataset)
    def display_single_molecule_data(self, data: SingleMoleculeLocalizationDataset):
        if "Data" not in self._mw.singeMoleculeInteractiveCurvesWidget.plot_names:
            self._mw.singeMoleculeInteractiveCurvesWidget.plot(
                "Data", x=data.frequency_spectrum[0], y=data.frequency_spectrum[1], pen=QudiPalette.c1
            )
        else:
            self._mw.singeMoleculeInteractiveCurvesWidget.set_data(
                "Data", x=data.frequency_spectrum[0], y=data.frequency_spectrum[1]
            )
        if data.fit_result_spectrally is not None:
            model = Lorentzian()
            param_dict = {k: Parameter(k, v) for (k, v) in data.fit_result_spectrally.items()}
            xs = np.linspace(start=data.frequency_spectrum[0].min(), stop=data.frequency_spectrum[0].max(), num=100)
            ys = model.eval(params=param_dict, x=xs)
            index_of_frame = np.argmin(np.abs(data.frequency_spectrum[0] - data.fit_result_spectrally["center"]))
            if "Fit" not in self._mw.singeMoleculeInteractiveCurvesWidget.plot_names:
                self._mw.singeMoleculeInteractiveCurvesWidget.plot("Fit", x=xs, y=ys, pen=QudiPalette.c2)
            else:
                self._mw.singeMoleculeInteractiveCurvesWidget.set_data("Fit", x=xs, y=ys)
        else:
            if "Fit" in self._mw.singeMoleculeInteractiveCurvesWidget.plot_names:
                self._mw.singeMoleculeInteractiveCurvesWidget.set_data("Fit", x=[], y=[])
            index_of_frame = np.argmax(data.frequency_spectrum[1])
        settings = self._localization_logic().settings
        if data.movie is not None:
            self._mw.singleMoleculeImageView.setImage(
                data.movie,
                pos=(data.flattened_image_m[0], data.flattened_image_n[0]),
                xvals=data.frequency_spectrum[0],
                scale=(settings.pixel_physical_dimension, settings.pixel_physical_dimension),
            )
            n = self._mw.singleMoleculeImageView.nframes()
            self._mw.singleMoleculeImageView.jumpFrames(-n)
            self._mw.singleMoleculeImageView.jumpFrames(index_of_frame)
        else:
            self._mw.singleMoleculeImageView.setImage(
                data.flattened_image,
                pos=(data.flattened_image_m[0], data.flattened_image_n[0]),
                scale=(settings.pixel_physical_dimension, settings.pixel_physical_dimension),
            )
        if data.fit_result_spatially is not None:
            self._mw.single_molecule_target_item.setPos(
                (data.fit_result_spatially['center_x'], data.fit_result_spatially['center_y'])
            )

    @Slot()
    def update_calibration(self):
        if self._mw is None:
            return
        self._localization_logic().settings = UserActionableMoleculeDetectionSettings(
            pixel_physical_dimension=self._mw.settings.pixelSizeDoubleSpinBox.value(),
            pixels_offset=(
                self._mw.settings.pixelOffsetXDoubleSpinBox.value(),
                self._mw.settings.pixelOffsetYDoubleSpinBox.value(),
            ),
            sample_position=(
                self._mw.settings.samplePositionXDoubleSpinBox.value(),
                self._mw.settings.samplePositionYDoubleSpinBox.value(),
            ),
            frequency_step=self._mw.settings.frequencyStepDoubleSpinBox.value(),
            frequency_offset=self._mw.settings.frequencyOffsetDoubleSpinBox.value(),
        )

    @Slot()
    def on_superlocalize_current_molecule(self):
        table = self._mw.currently_focused_table
        if table is None:
            return
        id = table.currentRow()
        if id < 0:
            return
        superlocalized = self._localization_logic().superlocalize_by_id(id)
        if superlocalized is not None:
            self.display_single_molecule_data(superlocalized)

    @Slot(list)
    def on_delete_molecules_by_id(self, ids: list[int]):
        if len(ids) == 1:
            self._localization_logic().delete_molecule_by_id(ids[0])
        else:
            self._localization_logic().delete_molecules_by_id(ids)

    @Slot()
    def on_save_parameter_estimation(self):
        self._localization_logic().save_parameter_estimation()

    @Slot()
    def on_save_molecule_positions(self):
        self._localization_logic().save_result()

    @Slot()
    def on_save_superlocalization(self):
        self._localization_logic().save_superlocalization()

    @Slot(np.ndarray, np.ndarray)
    def on_new_inhomogeneous_broadening_histogram(self, hist:np.ndarray, edges: np.ndarray):
        if "Histogram" not in self._mw.inhomogeneousInteractiveCurvesWidget.plot_names:
            self._mw.inhomogeneousInteractiveCurvesWidget.plot("Histogram", x=edges[:-1], y=hist, pen=QudiPalette.c1)
        else:
            self._mw.inhomogeneousInteractiveCurvesWidget.set_data("Histogram", x=edges[:-1], y=hist)

    @Slot(np.ndarray, np.ndarray)
    def on_new_linewidth_histogram(self, hist:np.ndarray, edges: np.ndarray):
        if "Histogram" not in self._mw.linewidthStatisticsInteractiveCurvesWidget.plot_names:
            self._mw.linewidthStatisticsInteractiveCurvesWidget.plot("Histogram", x=edges[:-1], y=hist, pen=QudiPalette.c1)
        else:
            self._mw.linewidthStatisticsInteractiveCurvesWidget.set_data("Histogram", x=edges[:-1], y=hist)

    @Slot(float, float)
    def on_new_inhomogeneous_broadening_statistics(self, mean, std):
        self._mw.frequencyMeanDoubleSpinBox.setValue(mean)
        self._mw.frequencyStdDoubleSpinBox.setValue(std)

    @Slot(float, float)
    def on_new_linewidth_statistics(self, mean, std):
        self._mw.lineWidthMeanDoubleSpinBox.setValue(mean)
        self._mw.lineWidthStdDoubleSpinBox.setValue(std)

