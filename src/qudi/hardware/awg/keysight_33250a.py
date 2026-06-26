try:
    import pyvisa as visa
except ImportError:
    import visa


from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.interface.process_control_interface import ProcessSetpointInterface, ProcessControlConstraints

class Keysight33250A(ProcessSetpointInterface):
    _visa_address = ConfigOption(name='visa_address',
                                 default='ASLR6::INSTR',
                                 missing='warn')
    _baud_rate = ConfigOption(name='baud_rate',
                                 default=57600
                             )
    _data_bits = ConfigOption(name='data_bits',
                                 default=8
                             )
    _awg_timeout = ConfigOption(name='awg_timeout', default=20, missing='warn')
    _rmparam = ConfigOption(name='rm_param', default="@py")
    _last_value = StatusVar(name="last_value", default=0.0)
    _active = StatusVar(name="active", default=True)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._BRAND = ''
        self._MODEL = ''
        self._SERIALNUMBER = ''
        self._FIRMWARE_VERSION = ''
        self._debug_check_all_commands = False       # # For development purpose, might slow down
        self.awg = None
    def on_activate(self):
        self._rm = visa.ResourceManager(self._rmparam)
        available = self._rm.list_resources()
        self.log.debug(f"Available resources: {available}")
        try:
            self.awg = self._rm.open_resource(self._visa_address, baud_rate=self._baud_rate, data_bits=self._data_bits)
            # set timeout by default to 30 sec
            self.awg.timeout = self._awg_timeout * 1000
        except:
            self.awg = None
            self.log.error('VISA address "{0}" not found by the pyVISA resource manager.\nCheck '
                           'the connection by using for example "Keysight Connection Expert".'
                           ''.format(self._visa_address))
            return
        if self.awg is not None:
            mess = self.query('*IDN?').split(',')
            self._BRAND = mess[0]
            self._MODEL = mess[1]
            self._SERIALNUMBER = mess[2]
            self._FIRMWARE_VERSION = mess[3]

            self.log.info('Load the device model "{0}" from "{1}" with '
                          'serial number "{2}" and firmware version "{3}" '
                          'successfully.'.format(self._MODEL, self._BRAND,
                                                 self._SERIALNUMBER,
                                                 self._FIRMWARE_VERSION))
            self.write("DISP:TEXT 'Hello, Qudi'")
            self.set_setpoint("output", self._last_value)
            self.set_activity_state("output", self._active)

        
    def on_deactivate(self):
        try:
            self.awg.close()
        except:
            self.log.warning('Closing AWG connection using pyvisa failed.')
        self.log.info('Closed connection to AWG')
    def set_setpoint(self, channel, value):
        self.write(f"APPL:DC DEF, DEF, {value}")
        self._last_value = value
    def get_setpoint(self, channel):
        val = float(self.query("VOLT:OFFS?"))
        self._last_value = val
        return val
    @property
    def constraints(self):
        mini = float(self.query("VOLT:OFFS? MIN"))
        maxi = float(self.query("VOLT:OFFS? MAX"))
        constraints = ProcessControlConstraints(
            setpoint_channels=("output",),
            units={"output":"V"},
            limits={"output":(mini, maxi)},
            dtypes={"output":float}
        )
        return constraints
    def set_activity_state(self, channel, active):
        if active:
            self.write("OUTP ON")
        else:
            self.write("OUTP OFF")
        self._active = active

    def get_activity_state(self, channel):
        res = self.query("OUTP?") == 1
        self._active = res
        return res

    def check_dev_error(self):
        has_error_occured = False
        for i in range(20):  # error buffer of device is 30
            raw_str = self.query(':SYST:ERR?', force_no_check=True)
            is_error = not ('0,' == raw_str[0:2])
            if is_error:
                self.log.warn("AWG issued error: {}".format(raw_str))
                has_error_occured = True
            else:
                break

        return has_error_occured

    def query(self, question, force_no_check=False):
        """ Asks the device a 'question' and receive and return an answer from it.

        @param string question: string containing the command

        @return string: the answer of the device to the 'question' in a string
        """
        ret = self.awg.query(question).strip().strip('"')
        if self._debug_check_all_commands and not force_no_check:
            if 0 != self.check_dev_error():
                self.log.warn("Check failed after query: {}".format(question))

        return ret

    def write(self, command):
        """ Sends a command string to the device.

            @param string command: string containing the command

            @return int: error code (0:OK, -1:error)
        """
        bytes_written = self.awg.write(command)

        if self._debug_check_all_commands:
            if 0 != self.check_dev_error():
                self.log.warn("Check failed after command: {}".format(command))

        return 0
