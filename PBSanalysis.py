"""
Handle power measurements from 6 setups:
H -> PBS -> Det @ T
H -> PBS -> Det @ R
V -> PBS -> Det @ T
V -> PBS -> Det @ R
H -> Det 
V -> Det

All power values and standard deviations are in microwatts (uW)
"""

import numpy as np
import matplotlib.pyplot as plt

# Background values in uW
background_R = 0.004371 #uW
background_T = 0.01906 #uW

def calc_er(power, power_leak, background):
    """
    Calculate extinction ratio
    power: transmitted/reflected power in uW
    power_leak: leaked power in uW
    background: background power in uW
    Returns: dimensionless ratio
    """
    return power / (power_leak - background)

def calc_tp(power, power_no_PBS):
    """
    Calculate power transmission
    power: power after PBS in uW
    power_no_PBS: power without PBS in uW
    Returns: dimensionless ratio
    """
    return power / power_no_PBS

def calc_er_std(power, power_leak, background, power_std, power_leak_std):
    """
    Calculate uncertainty in extinction ratio
    All input values in uW
    Returns: uncertainty in the ratio
    """
    A = power
    B = power_leak - background
    dA = power_std
    dB = power_leak_std  # background uncertainty is negligible
    return np.sqrt((dA/A)**2 + (dB/B)**2) * (A/B)

def calc_tp_std(power, power_no_PBS, power_std, power_no_PBS_std):
    """
    Calculate uncertainty in power transmission
    All input values in uW
    Returns: uncertainty in the ratio
    """
    A = power
    B = power_no_PBS
    dA = power_std
    dB = power_no_PBS_std
    return np.sqrt((dA/A)**2 + (dB/B)**2) * (A/B)

def is_numeric_line(line):
    # Check if line contains only numbers and commas
    line = line.strip()
    if not line:  # Skip empty lines
        return False
    # Split by comma and check if each part is a number
    parts = line.split(',')
    try:
        [float(x.strip()) for x in parts]
        return True
    except ValueError:
        return False

# Initialize lists to store power values and standard deviations
power_sets = []
std_sets = []

# Read all lines from the file
with open('/Users/irene/Documents/Photon/PBS_char_jun25/PBS_data_test_3.txt', 'r') as f:
    lines = f.readlines()
    
    # Process pairs of lines (power values and std deviations)
    i = 0
    while i < len(lines):
        # Skip non-numeric lines
        while i < len(lines) and not is_numeric_line(lines[i]):
            i += 1
        
        if i + 1 < len(lines):
            # Process power values
            power_str = lines[i].strip()
            power_values = [float(x.strip()) for x in power_str.split(',')]
            power_sets.append(np.array(power_values))
            
            # Process standard deviations
            std_str = lines[i + 1].strip()
            std_values = [float(x.strip()) for x in std_str.split(',')]
            std_sets.append(np.array(std_values))
            
            i += 2  # Move to next pair
        else:
            break

meas_type = ['V -> PBS -> Det @ R', 
             'V -> PBS -> Det @ T', 
             'V -> Det',
             'H -> PBS -> Det @ T', 
             'H -> PBS -> Det @ R', 
             'H -> Det']

# Calculate averages and combined standard deviations
print("\nMeasurement Results:")
avg_powers = []
avg_stds = []

for i, (power, std) in enumerate(zip(power_sets, std_sets)):
    # Calculate average power
    avg_power = np.mean(power)
    
    # Calculate combined standard deviation
    # For the mean, we divide by sqrt(n) and use quadrature sum
    combined_std = np.sqrt(np.sum(std**2)) / np.sqrt(len(std))
    
    avg_powers.append(avg_power)
    avg_stds.append(combined_std)
    
    print(f"\nMeasurement: {meas_type[i]}")
    print(f"Individual power values (uW): {power}")
    print(f"Individual standard deviations (uW): {std}")
    print(f"Average power (uW): {avg_power:.1f} ± {avg_stds[i]:.3f}")

# Calculate extinction ratio and transmitted power
print("\nResults:")

# For H polarization
H_T = avg_powers[3]  # H -> PBS -> Det @ T
H_R = avg_powers[4]  # H -> PBS -> Det @ R
H_no_PBS = avg_powers[5]  # H -> Det
H_T_std = avg_stds[3]
H_R_std = avg_stds[4]
H_no_PBS_std = avg_stds[5]

# For V polarization
V_T = avg_powers[1]  # V -> PBS -> Det @ T
V_R = avg_powers[0]  # V -> PBS -> Det @ R
V_no_PBS = avg_powers[2]  # V -> Det
V_T_std = avg_stds[1]
V_R_std = avg_stds[0]
V_no_PBS_std = avg_stds[2]

# Calculate extinction ratio for H polarization
er_H = calc_er(H_T, V_T, background_T)
er_H_std = calc_er_std(H_T, V_T, background_T, H_T_std, V_T_std)
print(f"\nExtinction Ratio (H, Transmitted): {er_H:.2f} ± {er_H_std:.3f}")

# Calculate extinction ratio for V polarization
er_V = calc_er(V_R, H_R, background_R)
er_V_std = calc_er_std(V_R, H_R, background_R, V_R_std, H_R_std)
print(f"Extinction Ratio (V, Reflected): {er_V:.2f} ± {er_V_std:.3f}")

# Calculate transmitted power for H polarization
tp_H = calc_tp(H_T, H_no_PBS)
tp_H_std = calc_tp_std(H_T, H_no_PBS, H_T_std, H_no_PBS_std)
print(f"Transmitted Power (H, Transmitted): {100*tp_H:.2f} ± {100*tp_H_std:.3f}")

# Calculate transmitted power for V polarization
tp_V = calc_tp(V_R, V_no_PBS)
tp_V_std = calc_tp_std(V_R, V_no_PBS, V_R_std, V_no_PBS_std)
print(f"Transmitted Power (V, Reflected): {100*tp_V:.2f} ± {100*tp_V_std:.3f}")