import serial
import csv
# Import the actual TI Toolbox functions
from parseFrame import *
from gui_parser import *
import pickle
import datetime
from write_to_db import save_raw_ti_data
from sqlalchemy import create_engine, exc
import psycopg2

# --- CONFIG ---
DATA_PORT = "COM5" 
BAUD_RATE = 921600
#LOG_FILENAME = "radar_official_output.csv"
#fname = f"../bin_data/{datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.pkl"

engine = create_engine()


ti_parser=UARTParser(type='DoubleCOMPort')
ti_parser.dataCom=serial.Serial(DATA_PORT, 921600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)

while True:
    output_dict=ti_parser.readAndParseUartDoubleCOMPort()
    print(output_dict)
    with engine.connect() as conn: 
        raw_conn=conn.connection
        save_raw_ti_data(raw_conn, output_dict)
   


#https://www.ti.com/tool/IWRL6432AOPEVM