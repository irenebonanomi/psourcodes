""" Jul 03 2025 """

import pyvisa, textwrap
rm = pyvisa.ResourceManager('@py')
for addr in rm.list_resources():
    try:
        idn = rm.open_resource(addr, read_termination='\n').query('*IDN?')
        print(f"{addr:40} → {idn.strip()}")
    except pyvisa.VisaIOError:
        pass

