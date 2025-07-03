""" Jul 01 2025 """

"""s

Light ---> Device Under Test ---> Detector 1
       |
       |
       v
       Detector 2
"""

#from bellMotors import APTMotor
#from bellMotors import RotationController
import pyvisa
import pyvisa.errors
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.optimize import curve_fit
import sys
from multiprocessing import Pool, TimeoutError
import usb
from concurrent.futures import ThreadPoolExecutor
import csv
import glob
import shutil
import usb
import os
# Connect to powermeter 1 and 2

# Start collecting data for a given time

# Save data
# Raw power from det 1 and det 2
# calculate normalized power

# calculates wanted parameter, such as extinction ratio or transmitted power

# Plot power and normalized power

# prints resultss

def force_detach(addr):
    # Extract the product ID from the VISA address (3rd field)
    try:
        pid = int(addr.split("::")[2], 16)
    except Exception:
        pid = 0x8078 if '8078' in addr else 0x8076
    vid = 0x1313  # Thorlabs vendor ID
    dev = usb.core.find(idVendor=vid, idProduct=pid)
    if dev and dev.is_kernel_driver_active(0):
        try:
            dev.detach_kernel_driver(0)
            print(f"Detached kernel driver from {hex(pid)}.")
        except usb.core.USBError as e:
            print("Detach failed:", e)

def setup_powermeter(addr, count_num):
    force_detach(addr)
    pm = rm.open_resource(addr, open_timeout=2000)
    pm.timeout = 5000 + 500 * count_num
    pm.write("*CLS")
    pm.write("SENS:CORR:WAV 1550")
    pm.write("SENS:POW:UNIT W")
    pm.write(f"SENS:AVER:COUN {int(count_num)}")
    print(pm.query("*IDN?").strip())
    return pm

def set_count_time(pms, count_num):
    for pm in pms:
        pm.write(f"SENS:AVER:COUN {int(count_num)}")
        pm.timeout = 5000 + 500 * count_num

def get_power(pm):
    t0 = time.time()
    p = float(pm.query("MEAS:POW?"))
    t1 = time.time()
    return p, t0, t1

def get_two_powers(pms):
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(get_power, pms))

def take_data(filepath, pm1, pm2, count_num=1, n_loops=200, pause=0.5):
    set_count_time([pm1, pm2], count_num)
    header = ['P1 [W]', 't0_1', 't1_1', 'P2 [W]', 't0_2', 't1_2']

    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)

        for i in range(n_loops):
            print(f'[{i+1:04}/{n_loops}] reading power …')
            (p1, t0_1, t1_1), (p2, t0_2, t1_2) = get_two_powers([pm1, pm2])
            w.writerow([p1, t0_1, t1_1, p2, t0_2, t1_2])
            time.sleep(pause)

def plot_data(fn):
    """
    Plot power‑vs‑time traces stored by take_data().
    
    Parameters
    ----------
    fn : str
        Path to the CSV file written by take_data().
    """

    cols = ['P1', 't0_1', 't1_1', 'P2', 't0_2', 't1_2']
    data = {c: [] for c in cols}

    with open(fn, newline='') as f:
        rdr = csv.reader(f)
        next(rdr)                    # skip header
        for row in rdr:
            for k, v in zip(cols, row):
                data[k].append(float(v))

    P1 = 1e6 * np.asarray(data['P1'])        # W -> µW
    P2 = 1e6 * np.asarray(data['P2'])        # W -> µW
    t  = np.asarray(data['t0_1'])            # acquisition start times
    Pnorm = P1 / P2                          # unitless

    # Re‑zero time for prettier x‑axis
    t -= t[0]

    fig, ax1 = plt.subplots()

    ax1.plot(t, P1, label='PM (µW)', linewidth=1.2)
    ax1.plot(t, P2, label='PM Norm (µW)', linewidth=1.2, linestyle='--')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Power [µW]')
    ax1.grid(True, alpha=0.3)

    # Twin axis for the normalized power
    ax2 = ax1.twinx()
    ax2.plot(t, Pnorm, label='P_norm', color='tab:red', linewidth=1)
    ax2.set_ylabel('Normalized Power (P₁/P₂) [unitless]', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Collect legends from both y‑axes
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title('Power traces and normalized power vs. time')
    plt.tight_layout()

    fig.savefig(fn.replace('.csv', '.png'), dpi=300, bbox_inches='tight')

    plt.show()
    
rm = pyvisa.ResourceManager('@py')  # use pyvisa-py backend

pm_addr = 'USB0::4883::32886::M01112547::0::INSTR'       # PM100D
pm_norm_addr = 'USB0::4883::32888::P0040935::0::INSTR'   # PM101U
save_path = '/home/qlab/OpticsCal'
os.makedirs(save_path, exist_ok=True)

pm1 = setup_powermeter(pm_addr, count_num=1)
pm2 = setup_powermeter(pm_norm_addr, count_num=1)

fn = 'pow_log_00.csv'

try:
    take_data(os.path.join(save_path, fn),
              pm1, pm2, count_num=1, n_loops=10)
    plot_data(f'/home/qlab/OpticsCal/{fn}')

finally:
    pm1.close()
    pm2.close()






