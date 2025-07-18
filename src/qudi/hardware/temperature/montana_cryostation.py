import socket

from qudi.core.module import Base
from qudi.core.configoption import ConfigOption
from qudi.interface.process_control_interface import ProcessControlInterface, ProcessControlConstraints

class Cryostation(ProcessControlInterface):
    """
    This is a connector to Montana's legacy Cryostation software
    """
    _ip_address = ConfigOption('ip_address', default='localhost')
    _ip_port = ConfigOption('port', default=7773)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cryostation_socket = None
    
    def on_activate(self):
      """ Initialisation performed during activation of the module.
      """
      self.cryostation_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.cryostation_socket.connect((self._ip_address, self._ip_port))
      
    def on_deactivate(self):
        if self.cryostation_socket:
            self.cryostation_socket.close()
      
    def send_cmd(self, cmd):
        l = len(cmd)
        self.cryostation_socket.send(("%.2d%s" % (l, cmd)).encode('ascii'))

    def send_cmd_with_float(self, cmd, arg):
        self.send_cmd(cmd + " " + ("%.6e" % arg))

    def recv_response(self):
        response = self.cryostation_socket.recv(1024)
        l = int(response[0:2])
        return response[2:2+l]

    def query(self, cmd):
        self.send_cmd(cmd)
        return self.recv_response()

    def query_with_float(self, cmd, arg):
        self.send_cmd_with_float(cmd, arg)
        return self.recv_response()
      
    # ProcessControlInterfaceBase
    @property
    def constraints(self):
        return ProcessControlConstraints(
            setpoint_channels = ["user stage temperature", "platform temperature"],
            process_channels = ["chamber pressure", "platform heater power", 
            "platform temperature", "platform stability", "stage 1 heater power", "stage 2 heater power",
            "sample temperature", "sample stability", "user temperature", "user stability"
            ],
            units = {"platform temperature":"K", "platform stability":"K"},
            limits = {"platform temperature":(2.0,300.0)},
            dtypes = {
                "chamber pressure": float,
                "user stage temperature": float,
                "platform temperature": float,
                "platform heater power": float,
                "platform stability" : float,   
                "stage 1 heater power": float, 
                "stage 2 heater power": float,
                "sample temperature": float, 
                "sample stability": float, 
                "user temperature": float, 
                "user stability": float                
            },
        )
    def set_activity_state(self, channel, active):
        pass
    def get_activity_state(self, channel):
        return True
        
    # ProcessSetpointInterface
    def set_setpoint(self, channel, value):
        if channel == "user stage temperature":
            self.query_with_float("SHTSP", value)
        elif channel == "platform temperature":
            self.query_with_float("STSP", value)
        else:
            pass
    def get_setpoint(self, channel):
        if channel == "user stage temperature":
            return float(self.query("GHTSP"))
        elif channel == "platform temperature":
            return float(self.query("GTSP"))
        else:
            pass
            
    # ProcessValueInterface
    def get_process_value(self, channel):
        if channel == "chamber pressure":
            return float(self.query("GCP"))
        elif channel == "platform heater power":
            return float(self.query("GPHP"))
        elif channel == "platform temperature":
            return float(self.query("GPT"))
        elif channel == "platform stability":
            return float(self.query("GPS"))
        elif channel == "platform stability":
            return float(self.query("GPS"))
        elif channel == "stage 1 heater power":
            return float(self.query("GS1T"))
        elif channel == "stage 2 heater power":
            return float(self.query("GS2T"))
        elif channel == "sample temperature":
            return float(self.query("GST"))
        elif channel == "sample stability":
            return float(self.query("GSS"))
        elif channel == "user temperature":
            return float(self.query("GUT"))
        elif channel == "user stability":
            return float(self.query("GUS"))
            
            