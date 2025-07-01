""" Jul 01 2025 """

"""
Type of setup:

Light ---> Device Under Test ---> Detector 1
       |
       |
       v
       Detector 2
"""

from bellMotors import APTMotor
from bellMotors import RotationController
import pyvisa
import pyvisa.errors
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.optimize import curve_fit
import sys
from numba import jit, prange
from multiprocessing import Pool, TimeoutError
import usb
from concurrent.futures import ThreadPoolExecutor
import csv
import glob
import shutil

# Connect to powermeter 1 and 2

# Start collecting data for a given time

# Save data
# Raw power from det 1 and det 2
# calculate normalized power

# calculates wanted parameter, such as extinction ratio or transmitted power

# Plot power and normalized power

# prints results

def setup_powermeter(addr,countNum):
    try:
        # Open the resource
        pmeter = rm.open_resource(addr, open_timeout=1000)
        
        # Set the communication timeout, taking integration time into account
        pmeter.timeout = 5000 + 0.5*countNum
        
        # Query and print the instrument's identification
        print(pmeter.query("*IDN?"))
        
        # Configure the instrument settings
        pmeter.write("SENS:CURR:RANGE:AUTO ON")  # Set the range to auto
        pmeter.write("SENS:CORR:WAV 1550")  # Set the wavelength to 1550 nm
        pmeter.write("SENS:POW:UNIT W")     # Set the power unit to watts
        pmeter.write("sense:average:count "+str(int(countNum)))      # Set the averaging to 1000
        
        # Return the instrument object
        print("Powermeter has been succesfully set up.")
        return pmeter

    except pyvisa.errors.VisaIOError as e:
        print(f"Error setting up instrument: {str(e)}")
        return e
    
    except usb.core.USBTimeoutError as e:
        print(f'usb error: {str(e)}')
        return e
    except:
        print('unknown error on this pmeter:')
        print(pmAddr)
        return -1

def connectAndReadPower(addr,countNum):
    times = [0,0,0]
    times[0] = time.time()
    success = False
    try:
        pm = setup_powermeter(addr,countNum)
        if pm == -1:
            success = False
            return -1
        else:
            success = True 
    except pyvisa.errors.VisaIOError as e:
        print(f'error connecting to PM: {e}')
        success = False
        return e
    if success is True:
        times[1] = time.time()
        pow = float(pm.query("meas:pow?"))
        times[2] = time.time()
        pm.close()
        return pow, times

def setCountTime(pms, countNum):
    for pm in pms:
        pm.write("sense:average:count "+str(int(countNum)))
        pm.timeout = 5000 + 0.5*countNum

def getTwoPowers(pms):
    ps = np.zeros((2,3))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = executor.map(getPower, pms)
        row = 0
        for future in futures:
            ps[row] = future
            row += 1
    return ps    

def TakeData(path,pm, pmNorm, countNum, nLoops):

    fname = path + f'test'.txt'

    setCountTime([pm,pmNorm], countNum)
    
    for n in np.arange(nLoops):

        print('Reading power...')

        powers = getTwoPowers([pm,pmNorm])
        time.sleep(0.5)

        np.savetxt(fname,powers,delimiter=',')

pmAddr = 'USB0::4883::32885::P5002859::0::INSTR'
pmNormAddr = 'USB0::4883::32886::M01112547::0::INSTR'

countNum = 1

connectAndReadPower(pmAddr, countNum)
connectAndReadPower(pmNormAddr, countNum)

path = '/Users/inb4/Photon'

TakeData(path,pm, pmNorm, countNum, 200)




