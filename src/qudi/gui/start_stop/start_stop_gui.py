from typing import Iterable
from dataclasses import dataclass
import os
import numpy as np

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QMessageBox
from qudi.core.connector import Connector
from qudi.core.module import GuiBase
from qudi.core.statusvariable import StatusVar
from qudi.util.colordefs import QudiPalette
from qudi.util.colordefs import QudiPalettePale as palette

from qudi.util.paths import get_artwork_dir
from qudi.util.widgets.toggle_switch import ToggleSwitch
from qudi.util.widgets.scientific_spinbox import ScienDSpinBox

from qudi.util.widgets.fitting import FitConfigurationDialog 
from qudi.interface.start_stop_interface import (
    InputEdge,
    InputConfiguration,
    StartStopHardwareState,
    StartStopHistogram,
    StartStopHistogramConfiguration,
)
from qudi.logic.start_stop_logic import StartStopLogic
from qudi.util import uic

@dataclass
class InputSettingsWidgets:
    name: QtWidgets.QLineEdit
    level: ScienDSpinBox
    edge: QtWidgets.QComboBox
    delay: ScienDSpinBox

@dataclass
class HistogramSettingsWidgets:
    enabled: ToggleSwitch
    name: QtWidgets.QLineEdit
    bin_width: ScienDSpinBox
    length: ScienDSpinBox
    exposure_time: ScienDSpinBox

class CustomAxis(pg.AxisItem):
    """ This is a CustomAxis that extends the normal pyqtgraph to be able to nudge the axis labels.
    """

    @property
    def nudge(self):
        if not hasattr(self, "_nudge"):
            self._nudge = 5
        return self._nudge

    @nudge.setter
    def nudge(self, nudge):
        self._nudge = nudge
        s = self.size()
        # call resizeEvent indirectly
        self.resize(s + QtCore.QSizeF(1, 1))
        self.resize(s)

    def resizeEvent(self, ev=None):
        # Set the position of the label
        nudge = self.nudge
        br = self.label.boundingRect()
        p = QtCore.QPointF(0, 0)
        if self.orientation == "left":
            p.setY(int(self.size().height() / 2 + br.width() / 2))
            p.setX(-nudge)
        elif self.orientation == "right":
            p.setY(int(self.size().height() / 2 + br.width() / 2))
            p.setX(int(self.size().width() - br.height() + nudge))
        elif self.orientation == "top":
            p.setY(-nudge)
            p.setX(int(self.size().width() / 2.0 - br.width() / 2.0))
        elif self.orientation == "bottom":
            p.setX(int(self.size().width() / 2.0 - br.width() / 2.0))
            p.setY(int(self.size().height() - br.height() + nudge))
        self.label.setPos(p)
        self.picture = None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'start_stop_gui.ui')
        super().__init__(*args, **kwargs)
        uic.loadUi(ui_file, self)
        icon_path = os.path.join(get_artwork_dir(), 'icons')
        restore_icon = QtGui.QIcon(os.path.join(icon_path, 'view-refresh'))
        fit_settings_icon = QtGui.QIcon(os.path.join(icon_path, 'configure'))
        save_icon = QtGui.QIcon(os.path.join(icon_path, 'document-save'))
        stop_icon = QtGui.QIcon(os.path.join(icon_path, 'media-playback-stop.svgz'))
        self.toggle_run_icon = QtGui.QIcon()
        self.toggle_run_icon.addFile(os.path.join(icon_path, 'media-playback-start.svgz'), QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toggle_run_icon.addFile(os.path.join(icon_path, 'media-playback-pause.svgz'), QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
        configure_icon = QtGui.QIcon(os.path.join(icon_path, 'configure'))
        self.actionStopAcquisition.setIcon(stop_icon)
        self.actionSave.setIcon(save_icon)
        self.actionToggleAcquisition.setIcon(self.toggle_run_icon)
        self.actionConfigureHardware.setIcon(configure_icon)
        self.actionShowInputSettingsDock.setChecked(True)
        self.actionShowInputSettingsDock.setChecked(True)
        self.actionShowHistogramSettingsDock.setChecked(True)
        self.actionShowResultsDock.setChecked(True)
        self.actionShowHistogramSettingsDock.toggled.connect(self.histogramSettingsDockWidget.setVisible)
        self.actionShowInputSettingsDock.toggled.connect(self.inputSettingsDockWidget.setVisible)
        self.actionShowResultsDock.toggled.connect(self.resultsDockWidget.setVisible)
        self.plotWidget.setAxisItems(
            axisItems={'bottom': CustomAxis(orientation='bottom'),
                       'left'  : CustomAxis(orientation='left')}
                )
        self.plotWidget.getAxis('bottom').nudge = 0
        self.plotWidget.getAxis('left').nudge = 0
        self.plotWidget.showGrid(x=True, y=True, alpha=0.5)
        self.plotWidget.setLabel('left', 'Count')
        self.plotWidget.setMinimumHeight(300)
        self.plotWidget.setLabel('bottom', 'Time', units='s')

        self.fit_curve = self.plotWidget.plot()
        self.fit_curve.setPen(palette.c2, width=2)
        self.data_curves = {}
        self.fit_region = pg.LinearRegionItem(values=(0, 1),
                                              brush=pg.mkBrush(122, 122, 122, 30),
                                              hoverBrush=pg.mkBrush(196, 196, 196, 30))
        self.plotWidget.addItem(self.fit_region)

        self.target_point = pg.InfiniteLine(pos=0,
                                            angle=90,
                                            movable=True,
                                            pen=pg.mkPen(color='green', width=2))
        self.plotWidget.addItem(self.target_point)

        self.input_settings_widgets: dict[str, InputSettingsWidgets] = {}
        self.histogram_settings_widgets: dict[str, HistogramSettingsWidgets] = {}

    def add_curve(self, name):
        self.data_curves[name] = self.plotWidget.plot(symbol='o', symbolSize=5, name=name)
        self.data_curves[name].setPen(palette.c1, width=2)

    def clear_input_settings_widgets(self):
        input_layout = self.inputSettingsContents.layout()
        for widgets in self.input_settings_widgets.values():
            input_layout.removeWidget(widgets.name)
            input_layout.removeWidget(widgets.level)
            input_layout.removeWidget(widgets.edge)
            input_layout.removeWidget(widgets.delay)
        self.input_settings_widgets = {}

    def clear_histogram_settings_widgets(self):
        histogram_layout = self.histogramSettingsContents.layout()
        for widgets in self.histogram_settings_widgets.values():
            histogram_layout.removeWidget(widgets.enabled)
            histogram_layout.removeWidget(widgets.name)
            histogram_layout.removeWidget(widgets.bin_width)
            histogram_layout.removeWidget(widgets.length)
            histogram_layout.removeWidget(widgets.exposure_time)
            widgets.enabled.toggled.disconnect(widgets.enabled.setEnabled)
            widgets.enabled.toggled.disconnect(widgets.bin_width.setEnabled)
            widgets.enabled.toggled.disconnect(widgets.length.setEnabled)
            widgets.enabled.toggled.disconnect(widgets.exposure_time.setEnabled)
        self.histogram_settings_widgets = {}

    def add_input_settings_widget(self, name:str, config: InputConfiguration):
        if name not in self.input_settings_widgets:
            layout = self.inputSettingsContents.layout()
            name_widget = QtWidgets.QLineEdit()
            name_widget.setReadOnly(True)
            name_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
            level_widget = ScienDSpinBox()
            level_widget.setSuffix("V")
            level_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
            edge_widget = QtWidgets.QComboBox()
            edge_widget.addItems(["Rising", "Falling"])
            edge_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
            delay_widget = ScienDSpinBox()
            delay_widget.setSuffix("s")
            delay_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
            self.input_settings_widgets[name] = InputSettingsWidgets(
                name_widget, level_widget, edge_widget, delay_widget
            )
            row = layout.rowCount()
            layout.addWidget(name_widget, row, 0)
            layout.addWidget(level_widget, row, 1)
            layout.addWidget(edge_widget, row, 2)
            layout.addWidget(delay_widget, row, 3)
        self.update_input_settings_widget(name, config)

    def update_input_settings_widget(self, name:str, config: InputConfiguration):
        self.input_settings_widgets[name].name.setText(name)
        self.input_settings_widgets[name].level.setValue(config.level)
        self.input_settings_widgets[name].delay.setValue(config.delay)
        if config.edge == InputEdge.RISING:
            index = 0
        else:
            index = 1
        self.input_settings_widgets[name].edge.setCurrentIndex(index)

    def add_histogram_settings_widget(self, name:str, config: StartStopHistogramConfiguration):
        if name not in self.histogram_settings_widgets:
            layout = self.histogramSettingsContents.layout()
            enabled_widget = ToggleSwitch()
            enabled_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
            name_widget = QtWidgets.QLineEdit()
            name_widget.setReadOnly(True)
            name_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
            bin_width_widget = ScienDSpinBox()
            bin_width_widget.setSuffix("s")
            bin_width_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
            length_widget = ScienDSpinBox()
            length_widget.setSuffix("s")
            length_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
            exposure_time_widget = ScienDSpinBox()
            exposure_time_widget.setSuffix("s")
            exposure_time_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
            self.histogram_settings_widgets[name] = HistogramSettingsWidgets(
                enabled_widget, name_widget, bin_width_widget, length_widget, exposure_time_widget
            )
            row = layout.rowCount()
            layout.addWidget(enabled_widget, row, 0)
            layout.addWidget(name_widget, row, 1)
            layout.addWidget(bin_width_widget, row, 2)
            layout.addWidget(length_widget, row, 3)
            layout.addWidget(exposure_time_widget, row, 4)
            enabled_widget.toggled.connect(name_widget.setEnabled)
            enabled_widget.toggled.connect(bin_width_widget.setEnabled)
            enabled_widget.toggled.connect(length_widget.setEnabled)
            enabled_widget.toggled.connect(exposure_time_widget.setEnabled)
        self.update_histogram_settings_widget(name, config)

    def update_histogram_settings_widget(self, name:str, config: StartStopHistogramConfiguration):
        self.histogram_settings_widgets[name].enabled.setChecked(config.enabled)
        self.histogram_settings_widgets[name].name.setText(name)
        self.histogram_settings_widgets[name].bin_width.setValue(config.bin_width_ns*1e-9)
        self.histogram_settings_widgets[name].length.setValue(config.histogram_length_ns*1e-9)
        self.histogram_settings_widgets[name].exposure_time.setValue(config.exposure_time_s)



class StartStopGui(GuiBase):
    start_stop_logic = Connector(interface=StartStopLogic)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw: MainWindow = MainWindow()
    def on_activate(self):
        self.show()
        self.show_status(self.start_stop_logic().current_status)
        self.build_input_settings(self.start_stop_logic().input_configurations)
        self.build_histogram_settings(self.start_stop_logic().configuration)
        self.fit_settings_dialog = FitConfigurationDialog(
                parent=self._mw,
                fit_config_model=self.start_stop_logic().fit_config_model
                )
        self._mw.fitWidget.link_fit_container(self.start_stop_logic().fit_container)
        self.__signals_map = (
                (self.start_stop_logic().sigNewData, self.on_new_data),
                (self.start_stop_logic().sigNewStatus, self.show_status),
                (self.start_stop_logic().sigAcquisitionRunning, self.on_acquisition_started),
                (self.start_stop_logic().sigAcquisitionPaused, self.on_acquisition_paused),
                (self.start_stop_logic().sigAcquisitionStopped, self.on_acquisition_stopped),
                (self.start_stop_logic().sigNewInputsConfiguration, self.on_new_inputs_configuration),
                (self.start_stop_logic().sigNewHistogramsConfiguration, self.on_new_histograms_configuration),
                (self._mw.actionToggleAcquisition.toggled, self.on_toggle_acquisition),
                (self._mw.actionStopAcquisition.triggered, self.on_stop_acquisition),
                (self._mw.actionConfigureHardware.triggered, self.on_configure_hardware),
                (self._mw.actionShowFitSettings.triggered, self.fit_settings_dialog.show),
                (self._mw.fitWidget.sigDoFit, self.start_stop_logic().do_fit),
                (self._mw.fitRegionHistogramComboBox.currentTextChanged, self.start_stop_logic().set_fit_histogram),
                (self.start_stop_logic().sigFitUpdated, self.on_update_fit),
                (self.start_stop_logic().sigFitRegionUpdated, self.on_update_fit_region_from_logic),
                (self._mw.fitRegionFromDoubleSpinBox.editingFinished, self.on_fit_region_numerical_update),
                (self._mw.fit_region.sigRegionChangeFinished, self.on_fit_region_update_from_graphics),
                (self._mw.fitRegionToDoubleSpinBox.editingFinished, self.on_fit_region_numerical_update),
                (self._mw.fitRegionHistogramComboBox.currentIndexChanged, self.on_fit_histogram_changed),
                (self._mw.showFitRegionCheckBox.toggled, self.on_show_fit_region_toggled),
                (self._mw.fitVisibleCheckBox.toggled, self.on_show_fit_toggled),
                (self._mw.deleteCurrentFitPushButton.clicked, self.on_delete_current_fit),
                )
        self._mw.actionStopAcquisition.setEnabled(False)
        self.connect_signals()

    def on_deactivate(self):
        self.disconnect_signals()
        self._mw.close()

    def show(self):
        self._mw.show()
        self._mw.activateWindow()
        self._mw.raise_()

    def connect_signals(self):
        for sender,receiver in self.__signals_map:
            sender.connect(receiver)

    def disconnect_signals(self):
        for sender,receiver in self.__signals_map:
            sender.disconnect(receiver)

    def show_status(self, status:StartStopHardwareState):
        msg = ""
        if status == StartStopHardwareState.UNCONFIGURED:
            msg = "unconfigured"
        elif status == StartStopHardwareState.ERROR:
            msg = "error"
        elif status == StartStopHardwareState.IDLE:
            msg = "idle"
        elif status == StartStopHardwareState.PAUSED:
            msg = "paused"
        elif status == StartStopHardwareState.RUNNING:
            msg = "running"
        self._mw.statusBar().showMessage(f"Hardware status: {msg}")

    def build_input_settings(self, settings:dict[str, InputConfiguration]):
        self._mw.clear_input_settings_widgets
        self.on_new_inputs_configuration(settings)

    def build_histogram_settings(self, settings:dict[str, StartStopHistogramConfiguration]):
        self._mw.clear_histogram_settings_widgets()
        self.on_new_histograms_configuration(settings)

    def on_new_data(self, data: dict[str, StartStopHistogram]):
        for name, histogram in data.items():
            if name not in self._mw.data_curves:
                self._mw.add_curve(name)
            combo_index = self._mw.fitRegionHistogramComboBox.findText(name)
            if combo_index < 0:
                self._mw.fitRegionHistogramComboBox.addItem(name)
            times = np.linspace(start=0, stop=histogram.bin_width_ns*histogram.histogram.shape[0]*1e-9, num=histogram.histogram.shape[0])
            times = times - times[histogram.zero]
            self._mw.data_curves[name].setData(x=times, y=histogram.histogram)
        to_be_removed = []
        for name in self._mw.data_curves:
            if name not in data:
                to_be_removed.append(name)
        for name in to_be_removed:
            self._mw.plotWidget.removeItem(self._mw.data_curves[name])
            self._mw.data_curves.pop(name)
            combo_index = self._mw.fitRegionHistogramComboBox.findText(name)
            self._mw.fitRegionHistogramComboBox.removeItem(combo_index)

    def on_acquisition_started(self):
        self._mw.actionToggleAcquisition.blockSignals(True)
        self._mw.actionToggleAcquisition.setChecked(True)
        self._mw.actionToggleAcquisition.blockSignals(False)
        self._mw.actionStopAcquisition.setEnabled(True)

    def on_acquisition_paused(self):
        self._mw.actionToggleAcquisition.blockSignals(True)
        self._mw.actionToggleAcquisition.setChecked(False)
        self._mw.actionToggleAcquisition.blockSignals(False)

    def on_acquisition_stopped(self):
        self._mw.actionToggleAcquisition.blockSignals(True)
        self._mw.actionToggleAcquisition.setChecked(False)
        self._mw.actionToggleAcquisition.blockSignals(False)
        self._mw.actionStopAcquisition.setEnabled(False)

    def on_configure_hardware(self):
        input_configurations: dict[str, InputConfiguration] = {}
        for name,widgets in self._mw.input_settings_widgets.items():
            input_configurations[name] = InputConfiguration(
                level=widgets.level.value(),
                edge=InputEdge.RISING if widgets.edge.currentText() == "Rising" else InputEdge.FALLING,
                delay=widgets.delay.value()
            )
        self.start_stop_logic().configure_inputs(input_configurations)
        histogram_configurations : dict[str, StartStopHistogramConfiguration] = {}
        for name,widgets in self._mw.histogram_settings_widgets.items():
            histogram_configurations[name] = StartStopHistogramConfiguration(
                    enabled = widgets.enabled.isChecked(),
                    bin_width_ns= widgets.bin_width.value()*1e9,
                    histogram_length_ns=widgets.length.value()*1e9,
                    exposure_time_s=widgets.exposure_time.value(),
                    )
        self.start_stop_logic().configuration = histogram_configurations

    def on_toggle_acquisition(self, running:bool):
        if not running and self.start_stop_logic().current_status is not StartStopHardwareState.PAUSED:
            self.start_stop_logic().pause()
        elif running and self.start_stop_logic().current_status is not StartStopHardwareState.RUNNING:
            self.start_stop_logic().acquire()

    def on_stop_acquisition(self):
        self.start_stop_logic().stop()

    def on_new_inputs_configuration(self, settings:dict[str, InputConfiguration]):
        for name, config in settings.items():
            self._mw.add_input_settings_widget(name, config)
    def on_new_histograms_configuration(self, settings:dict[str, StartStopHistogramConfiguration]):
        for name, config in settings.items():
            self._mw.add_histogram_settings_widget(name, config)

    def on_update_fit(self, fit_method: str, fit_results):
        if fit_method != 'No Fit' and fit_results is not None:
            # redraw the fit curve in the GUI plot.
            self._mw.fit_curve.setData(x=fit_results.high_res_best_fit[0],
                                       y=fit_results.high_res_best_fit[1])
        else:
            self._mw.fit_curve.setData(x=[], y=[])

    def on_update_fit_region_from_logic(self, fit_region:tuple[float,float]):
        self._mw.fitRegionFromDoubleSpinBox.blockSignals(True)
        self._mw.fit_region.blockSignals(True)
        self._mw.fitRegionToDoubleSpinBox.blockSignals(True)
        self._mw.fitRegionFromDoubleSpinBox.setValue(fit_region[0])
        self._mw.fit_region.setRegion(fit_region)
        self._mw.fitRegionToDoubleSpinBox.setValue(fit_region[1])
        self._mw.fitRegionFromDoubleSpinBox.blockSignals(False)
        self._mw.fit_region.blockSignals(False)
        self._mw.fitRegionToDoubleSpinBox.blockSignals(False)

    def on_fit_region_numerical_update(self):
        self.start_stop_logic().fit_region = (self._mw.fitRegionFromDoubleSpinBox.value(), self._mw.fitRegionToDoubleSpinBox.value())

    def on_fit_histogram_changed(self, i:int):
        pass

    def on_show_fit_region_toggled(self, visible:bool):
        if visible:
            self._mw.plotWidget.addItem(self._mw.fit_region)
        else:
            self._mw.plotWidget.removeItem(self._mw.fit_region)

    def on_show_fit_toggled(self, visible:bool):
        if visible:
            self._mw.plotWidget.addItem(self._mw.fit_curve)
        else:
            self._mw.plotWidget.removeItem(self._mw.fit_curve)

    def on_delete_current_fit(self):
        self._mw.fit_curve.setData(x=[], y=[])

    def on_fit_region_update_from_graphics(self):
        self.start_stop_logic().fit_region = self._mw.fit_region.getRegion()
