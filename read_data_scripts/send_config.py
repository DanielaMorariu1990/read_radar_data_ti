import serial
import csv
# Import the actual TI Toolbox functions
from parseFrame import *
from gui_parser import *

# --- CONFIG ---
DATA_PORT = "COM5" 
BAUD_RATE = 921600
LOG_FILENAME = "radar_official_output.csv"
fname = "../AOP_6m_staticRetention.cfg"

ti_parser=UARTParser(type='DoubleCOMPort')
ti_parser.cliCom=serial.Serial("COM4", 115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)


with open(fname, "r") as cfg_file:
    cfg = cfg_file.readlines()
              
ti_parser.sendCfg(cfg)

