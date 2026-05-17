# /// script
# dependencies = [
#   "pyserial",
#   "pandas",
#   "numpy",
#   "json-fix",
# ]
# ///

import serial
import os
import csv
import sys

# 1. Dynamically find the absolute path of the 'common' directory
current_dir = os.path.dirname(os.path.abspath(__file__)) # inside read_data
parent_dir = os.path.dirname(current_dir)                # inside read_radar_data_ti
common_dir_path = os.path.join(parent_dir, 'common')     # path to common/

# 2. Add 'common' to the Python search path
if common_dir_path not in sys.path:
    sys.path.append(common_dir_path)
# Import the actual TI Toolbox functions
from parseFrame import *
from gui_parser import *



# --- CONFIG ---
DATA_PORT = "COM5" 
BAUD_RATE = 921600
LOG_FILENAME = "radar_official_output.csv"
fname = "AOP_6m_staticRetention.cfg"

full_cfg_path = os.path.join(current_dir, fname)

ti_parser=UARTParser(type='DoubleCOMPort')
ti_parser.cliCom=serial.Serial("COM4", 115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)


with open(full_cfg_path, "r") as cfg_file:
    cfg = cfg_file.readlines()
              
ti_parser.sendCfg(cfg)

