from PySide2 import QtCore
from PySide2 import QtWidgets
import pyqtgraph as pg
from qudi.util.colordefs import QudiPalettePale as palette

class ModeScanWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)
        self.mode_hops = []
        self.plot_widget = pg.PlotWidget(
            axisItems={'bottom': pg.AxisItem(orientation='bottom'),
                       'left'  : pg.AxisItem(orientation='left')}
        )
        self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
        # Create an empty plot curve to be filled later, set its pen
        self.data_curve = self.plot_widget.plot()
        self.data_curve.setPen(palette.c1, width=2)

        self.plot_widget.setLabel('left', 'Photodiode', units='V')
        self.plot_widget.setLabel('bottom', 'Piezo', units='arb. u.')
        self.plot_widget.setMinimumHeight(300)
        main_layout.addWidget(self.plot_widget)

    def update_data(self, data):
        self.data_curve.setData(x=data[0,:], y=data[1,:])

    def update_mode_hops(self, mode_hops):
        for l in self.mode_hops:
            self.plot_widget.removeItem(l)
        self.mode_hops = []
        for h in mode_hops:
            l = pg.InfiniteLine(pos=h,
                                angle=90,
                                movable=False,
                                pen=pg.mkPen(color='green', width=2))
            self.plot_widget.addItem(l)
            self.mode_hops.append(l)
