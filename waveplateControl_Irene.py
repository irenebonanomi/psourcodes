"""

This code controls 5 rotating stages and a powermeter to measure the error in the retardance from a half wave plate.

!!! BEFORE RUNNING THE CODE !!!
(1) Turn on all instruments (rotating stages, powermeter, laser)
(2) Change file names / directories if needed
(3) Change addresses for the rotating stages (use lsusb and search for Future Technology Devices International)

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

'''************************************************************************'''
''' hard coded parameters specific to this experiment '''
'''************************************************************************'''

today = 'xxxxxxxx'

# Initialize the resource manager
rm = pyvisa.ResourceManager()

# Adjust these addresses to match your hardware setup
rot_stage_addresses = ['ASRL/dev/ttyUSB0::INSTR',
                       'ASRL/dev/ttyUSB1::INSTR',
                       'ASRL/dev/ttyUSB2::INSTR',
                       'ASRL/dev/ttyUSB3::INSTR',
                       'ASRL/dev/ttyUSB4::INSTR']
# pmAddr = 'USB0::4883::32888::P0046227::0::INSTR'
# pmNormAddr = 'USB0::4883::32888::P0040935::0::INSTR'

pmAddr = 'USB0::4883::32885::P5002859::0::INSTR'
pmNormAddr = 'USB0::4883::32886::M01112547::0::INSTR'


"""
Below are the angles as indicated on the rotating wheel. 
H = parallel to the table (transmission plane)
V = perpendicular to the table
"""
QWP_1_deg = 38.8 # H
pol_1_deg = 25.3 # H
HWP_deg = 152.9 # H
QWP_2_deg = 104.9 # H
pol_2_deg_H = 22.9 # H
pol_2_deg_V = 112.9 # V

""" zero angles (Sep 29 2024) """
QWP_1_deg = 112.04
pol_1_deg = 25.99 # H
HWP_deg =  31.21
QWP_2_deg = 105.77 
pol_2_deg_H = 23.45 
pol_2_deg_V = 113.45

"""
Changing the names of the stages to match the optics on the table.

Old name: QWP_2_deg (qwp_2_stage)
Current name: HWP_1_deg (hwp_1_stage)

Old name: HWP_deg (hwp_under_test_stage)
Current name: sample_deg (sample_stage)

"""

""" zero angles (Oct 9 2024) """
""" updated names January 27 2025"""
HWP_1_deg = 48.75098 # holding state-preparation HWP 
pol_1_deg = 33.0808 # H
sample_deg =  121.0361
QWP_1_deg = 152.5743
pol_2_deg_H =  23.2634
pol_2_deg_V = 113.2634

# List of stage information
stages_info = [
    {'name': 'hwp_1_stage', 'serial': 27268569, 'initial_angle': HWP_1_deg, 'minVel': 1, 'acc': 5, 'maxVel': 25, 'zero': 0},
    {'name': 'pol_1_stage', 'serial': 27267846, 'initial_angle': pol_1_deg, 'minVel': 1, 'acc': 5, 'maxVel': 25, 'zero': 0},
    {'name': 'sample_stage', 'serial': 27005686, 'initial_angle': sample_deg, 'minVel': 5, 'acc': 1, 'maxVel': 25, 'zero': 0},
    {'name': 'qwp_1_stage', 'serial': 27268643, 'initial_angle': QWP_1_deg, 'minVel': 1, 'acc': 5, 'maxVel': 25, 'zero': 0},
    {'name': 'pol_2_stage', 'serial': 27268267, 'initial_angle': pol_2_deg_H, 'minVel': 1, 'acc': 5, 'maxVel': 25, 'zero': 0}
]

def initRotationStages(rm, stInfo, stAddresses):
    stages = {}
    rot_stages = []
    for addr in stAddresses:
        with rm.open_resource(addr) as stage_resource:
            rot_stages.append(stage_resource)

    # Loop through the list to create stage instances
    for stage_info in stages_info:
        stage = RotationController(stage_info)
        stages[stage_info['name']] = stage
    return stages

# Function to setup the power meter instrument with error handling
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

def getPower(pm):
    start = time.time()
    try:
        pow = float(pm.query("meas:pow?"))
        end = time.time()
        return pow, start, end

    except pyvisa.errors.VisaIOError as e:
        raise pyvisa.errors.VisaIOError
        
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

def simulPowerRead(pmAddr1,pmAddr2,countNum):
    # simultaneously read power from two power meters using Pool function from multiprocessing
    p = Pool(2)
    try:
        p1,p2=p.starmap(connectAndReadPower, [(pmAddr1,countNum), (pmAddr2,countNum)])
        return p1, p2
    except pyvisa.errors.VisaIOError as e:
        print(f'error in simulPowerRead: {e}')
        return e
    except:
        print(f'unknown error')

def simulPowerRead(pmAddr1,pmAddr2,countNum):
    # simultaneously read power from two power meters using Pool function from multiprocessing
    with Pool(5) as p:
        
        results = p.starmap_async(connectAndReadPower, [(pmAddr1,countNum), (pmAddr2,countNum)])
        try:
            res = results.get(timeout=2)
            print(res)
            return res
        except pyvisa.errors.VisaIOError as e:
            print(f'error in simulPowerRead: {e}')
            p.terminate()
            p.close()
            return e
        except TimeoutError as e:
            print(f'timeout error: {e}')
            p.terminate()
            p.close()
            return e
        except:
            print("uknown error in simulPowerRead, terminating")
            p.terminate()
            p.close()
            return -1

def moveBL(stage,angle):
    curPos = stage.getPos()
    angle = angle%360
    if angle-curPos > 2:
        print(f'moving clockwise from {curPos} to {angle}')
        stage.move(angle-curPos)
    else:
        print(f'moving counterclockwise from {curPos} to {angle-5}')
        stage.move(angle-curPos-5)
        print(f'now moving clockwise to {angle}')
        stage.move(angle-stage.getPos())

"""
Jan 27 2025 
Updating the names 'qwp2Zero' and 'hwpZero' to match the optics in the setup.

Old name: 'qwp2Zero'
Current name: 'hwp1Zero'

Old name: 'hwpZero'
Current name: 'sampleZero'

"""

def fourPowerBirefringenceMeas(stages, pm, pmNorm, countNum, qwp1Zero=152.5743,pol1Zero=33.0808,sampleZero=121.0361,hwp1Zero=48.75098,pol2Zero=23.2634):

    setCountTime([pm,pmNorm], countNum)

    powers = np.zeros((4,6))
    
    print('first Boulder meas')
    moveBL(stages['qwp_1_stage'],qwp1Zero+45)
    moveBL(stages['pol_2_stage'],pol2Zero)
    time.sleep(0.5)
    ps = getTwoPowers([pm,pmNorm])
    powers[0,:] = np.hstack((ps[0,:],ps[1,:]))

    print('second Boulder meas')
    moveBL(stages['qwp_1_stage'],qwp1Zero+45)
    moveBL(stages['pol_2_stage'],pol2Zero+90)
    time.sleep(0.5)
    ps = getTwoPowers([pm,pmNorm])
    powers[1,:] = np.hstack((ps[0,:],ps[1,:]))

    print('third Boulder meas')
    moveBL(stages['qwp_1_stage'],qwp1Zero-45)
    moveBL(stages['pol_2_stage'],pol2Zero+90)
    time.sleep(0.5)
    ps = getTwoPowers([pm,pmNorm])
    powers[3,:] = np.hstack((ps[0,:],ps[1,:]))

    print('fourth Boulder meas')
    moveBL(stages['qwp_1_stage'],qwp1Zero-45)
    moveBL(stages['pol_2_stage'],pol2Zero)
    time.sleep(0.5)
    ps = getTwoPowers([pm,pmNorm])
    powers[2,:] = np.hstack((ps[0,:],ps[1,:]))

    return powers

def twoDeSenarmontFriedelMeas(stages, pm, pmNorm, countNum, qwp1Zero=152.5743,pol1Zero=33.0808,sampleZero=121.0361,hwp1Zero=48.75098,pol2Zero=23.2634):

    setCountTime([pm,pmNorm], countNum)

    range = 7
    stepSize = 0.1

    ''' move QWP 1 into position for -45 degree sweep ''' 
    moveBL(stages['qwp_1_stage'],qwp1Zero-45)
    data2 = rotateAndCountNorm(stages['pol_2_stage'],pol2Zero+45-range,pol2Zero+45+range,stepSize,pm,pmNorm,countNum)
    #data2 = rotateAndCountNorm(stages['pol_2_stage'],pol2Zero+135-range,pol2Zero+135+range,stepSize,pm,pmNorm,countNum)

    ''' move QWP 1 into position for +45 degree sweep '''  
    moveBL(stages['qwp_1_stage'],qwp1Zero+45)
    data1 = rotateAndCountNorm(stages['pol_2_stage'],pol2Zero+45-range,pol2Zero+45+range,stepSize,pm,pmNorm,countNum)
    #data1 = rotateAndCountNorm(stages['pol_2_stage'],pol2Zero+135-range,pol2Zero+135+45+range,stepSize,pm,pmNorm,countNum)

    data  =  np.vstack((data1,data2))

    return data

def alternateBoulderAndDSFMethods(path, nameAppend, stages, pm, pmNorm, countNumBoulder, countNumDSF):
    """ zero angles (Oct 9 2024) """
    QWP_1_deg = 152.5743
    pol_1_deg = 33.0808 # H
    sample_deg =  121.0361
    HWP_1_deg = 48.75098 # holding state-preparation HWP 
    pol_2_deg_H =  23.2634
    pol_2_deg_V = 113.2634

    ''' initialize fixed stage positions '''
    moveBL(stages['hwp_1_stage'],HWP_1_deg+22.5)
    moveBL(stages['pol_1_stage'],pol_1_deg+45)
    moveBL(stages['sample_stage'],sample_deg)
    
    nLoops = 1000

    fnameBoulder = path + f'/boulder_countNum_{countNumBoulder}_'+nameAppend+'.csv'
    fnameDSF = path + f'/DSF_countNum_{countNumDSF}_'+nameAppend+'.csv'

    for n in np.arange(nLoops):

        powers = fourPowerBirefringenceMeas(stages, pm, pmNorm, countNumBoulder)
        if n == 0:
            np.savetxt(fnameBoulder,powers,delimiter=',')
            powersTotal = powers
        else:
            powersSaved = np.genfromtxt(fnameBoulder,delimiter=',')
            powersTotal = np.hstack((powersSaved,powers))
            np.savetxt(fnameBoulder,powersTotal,delimiter=',')

        data = twoDeSenarmontFriedelMeas(stages, pm, pmNorm, countNumDSF)
        if n == 0:
            np.savetxt(fnameDSF,data,delimiter=',')
            dataTotal = data
        else:
            dataSaved = np.genfromtxt(fnameDSF,delimiter=',')
            dataTotal = np.hstack((dataSaved,data))
            np.savetxt(fnameDSF,dataTotal,delimiter=',')
    
    return powersTotal,dataTotal

def initializeInputState(stages,pol1Offset,qwp1Zero=152.5743,pol1Zero=33.0808,sampleZero=121.0361,hwp1Zero=48.75098,pol2Zero=23.2634):
    # initialize angles for the input state (qwp2 (really a halfwaveplate, now hpw1) and pol1)
    moveBL(stages['hwp_1_stage'],hwp1Zero+22.5+90*pol1Offset/2)
    moveBL(stages['pol_1_stage'],pol1Zero+45+90*pol1Offset)

def alternateBoulderAndDSFMethodsAllCombos(path, nameAppend, stages, pm, pmNorm, countNumBoulder, countNumDSF, sampleIn = 1, nLoops=1000, qwp1Zero=152.5743,pol1Zero=33.0808,sampleZero=121.0361,hwp1Zero=48.75098,pol2Zero=23.2634):
    # for this measurement I want to try performing measurements at all combinations of flipping:
    #   QWP1 by 180 degrees ---> 2 dSF meas, 2 Boulder meas
    #   pol1 by 90 degrees ---> 2 dSF meas, 2 Boulder meas (and also rotate HWP in front of it, qwp2 (now hwp1), appropriately)
    #   pol2 by 180 degrees ---> 2 dSF meas, 2 Boulder meas
    # this gives 8 total combinations. Each of the above will be saved in a separate file to analyse each independently
    # I will keep track of/cycle through these 8 combinations using "Offsets" parameters
    # for now, I won't bother with randomizing the order

    # initialize HWP Under Test to zero degrees
    moveBL(stages['sample_stage'],sampleZero)

    for n in np.arange(nLoops):
        
        # cycle through angle-flipping combinations
        for pol1Offset in np.arange(2):
            ''' initialize fixed stage positions '''
            moveBL(stages['hwp_1_stage'],hwp1Zero+22.5+90*pol1Offset/2) 
            moveBL(stages['pol_1_stage'],pol1Zero+45+90*pol1Offset)
            
            for qwp1Offset in np.arange(2):
                for pol2Offset in np.arange(2):
                    # find filenames for this specific angle-flipping combination                    
                    angleFlip = '_pol1_'+str(pol1Offset)+'_qwp1_'+str(qwp1Offset)+'_pol2_'+str(pol2Offset)
                    fnameBoulder = path + f'/boulder_countNum_{countNumBoulder}_'+nameAppend+angleFlip+'.csv'
                    fnameDSF = path + f'/DSF_countNum_{countNumDSF}_'+nameAppend+angleFlip+'.csv'
                    
                    # if qwp1 or pol2 are flipped 180 degrees, add 180 to their zero angles
                    powers = fourPowerBirefringenceMeas(stages, pm, pmNorm, countNumBoulder, qwp1Zero=qwp1Zero+180*qwp1Offset, pol2Zero=pol2Zero+180*pol2Offset)
                    if n == 0:
                        np.savetxt(fnameBoulder,powers,delimiter=',')
                        powersTotal = powers
                    else:
                        powersSaved = np.genfromtxt(fnameBoulder,delimiter=',')
                        powersTotal = np.hstack((powersSaved,powers))
                        np.savetxt(fnameBoulder,powersTotal,delimiter=',')

                    # if qwp1 or pol2 are flipped 180 degrees, add 180 to their zero angles
                    # if sample is not in, set pol2 to measure near input state 
                    # if sample is in, set pol2 to measure 90 away from input state
                    data = twoDeSenarmontFriedelMeas(stages, pm, pmNorm, countNumDSF, qwp1Zero=qwp1Zero+180*qwp1Offset, pol2Zero=(pol2Zero+180*pol2Offset+90*pol1Offset+90*(1-sampleIn))%360)
                    if n == 0:
                        np.savetxt(fnameDSF,data,delimiter=',')
                        dataTotal = data
                    else:
                        dataSaved = np.genfromtxt(fnameDSF,delimiter=',')
                        dataTotal = np.hstack((dataSaved,data))
                        np.savetxt(fnameDSF,dataTotal,delimiter=',')


# spin a stage and measure (normalized) power
def rotateAndCountNorm(stage,start,end,stepSize,pm,pmNorm,countNum):
    angles = np.arange(start,end,stepSize)
    nAngles = np.size(angles)
    powers = np.zeros([nAngles,8])

    setCountTime([pm,pmNorm], countNum)

    for n in np.arange(nAngles):

        curPos = stage.getPos()
        if angles[n]-curPos > 0:
            print(f'moving clockwise from {curPos} to {angles[n]}')
            stage.move(angles[n]-curPos)
        else:
            print(f'moving counterclockwise from {curPos} to {angles[n]-5}')
            stage.move(angles[n]-curPos-5)
            print(f'now moving clockwise to {angles[n]}')
            stage.move(angles[n]-stage.getPos())

        time.sleep(0.1)

        stgAngle = stage.getPos()

        twoPowers = getTwoPowers([pm,pmNorm])

        powers[n,0] = angles[n]
        powers[n,1] = stgAngle
        powers[n,2:5] = twoPowers[0,:]
        powers[n,5:8] = twoPowers[1,:]
        print('angle', angles[n])
        print('norm Power', powers[n,2]/powers[n,5])

    return powers

def powersLoop(fname,pm,pmNorm,countNum,nLoops,sleepTime):
    setCountTime([pm,pmNorm],countNum)

    for n in np.arange(nLoops):
        powers = getTwoPowers([pm,pmNorm])
    
        with open(fname, 'ba') as f:
            np.savetxt(f,np.array([np.ravel(powers)]))

        time.sleep(sleepTime)

def calcAllanDev(x):
    nPoints = np.size(x)

    allanDevs = np.zeros((int(nPoints/3),2))

    for intNum in np.arange(1,int(nPoints/3)+1):
        xTemp = np.mean(np.reshape(x[:int(nPoints/intNum)*intNum],[int(nPoints/intNum),intNum]),axis=1)
        allanDevs[intNum-1,1] = np.sqrt(np.mean(np.diff(xTemp)**2)/(2))
        allanDevs[intNum-1,0] = intNum

    return allanDevs

def calcAllanDevNorm(x,y):
    nPoints = np.size(x)

    allanDevs = np.zeros((int(nPoints/3),2))

    for intNum in np.arange(1,int(nPoints/3)+1):
        xTemp = np.mean(np.reshape(x[:int(nPoints/intNum)*intNum],[int(nPoints/intNum),intNum]),axis=1)
        yTemp = np.mean(np.reshape(y[:int(nPoints/intNum)*intNum],[int(nPoints/intNum),intNum]),axis=1)

        allanDevs[intNum-1,1] = np.sqrt(np.mean(np.diff(xTemp/yTemp)**2)/2)
        allanDevs[intNum-1,0] = intNum
    return allanDevs

def analysePowersLoop(fname):
    data = np.genfromtxt(fname,delimiter=' ')

    allanDevRaw = calcAllanDev(data[:,0])
    allanDevNorm = calcAllanDev(data[:,0]/data[:,3]*np.mean(data[:,3]))
    allanDevNorm2 = calcAllanDevNorm(data[:,0],data[:,3])

    f,ax = plt.subplots(3,2,figsize =(12,10))
    ax[0,0].plot(data[:,0],'.',label='raw data')
    ax[0,0].plot(data[:,0]/data[:,3]*np.mean(data[:,3]),'.',label='normalized data')
    ax[0,0].plot(data[:,3]/np.mean(data[:,3])*np.mean(data[:,0]),label='normalization data, rescaled')
    ax[0,0].legend()
    ax[0,0].set_ylabel('power (W)')
    ax[0,0].set_xlabel('measurement number')
    ax[0,0].set_title('data',fontweight='bold')
  
    ax[1,0].plot(allanDevRaw[:,0],allanDevRaw[:,1],label='Allan deviation, raw')
    ax[1,0].plot(allanDevNorm[:,0],allanDevNorm[:,1],label='Allan deviation, normalized')
    ax[1,0].plot(allanDevNorm2[:,0],allanDevNorm2[:,1]*np.mean(data[:,3]),label='Allan deviation, normalized 2')
    ax[1,0].set_yscale('log')
    ax[1,0].set_xscale('log')
    ax[1,0].legend()
    ax[1,0].set_ylabel('uncertainty (W)')
    ax[1,0].set_xlabel('number of averaged measurements')
    ax[1,0].set_title('Allan deviation of power measurements',fontweight='bold')
    ax[1,0].grid(which='major',linewidth='0.5',color='gray')
    ax[1,0].grid(which='minor',linewidth='0.5',color='lightgray')

    ax[2,0].plot(np.mean(np.diff(data[:,1]))*allanDevRaw[:,0],1/(4*np.sqrt(2)*np.pi)*allanDevRaw[:,1]/np.mean(data[:,0]),label='Allan deviation, raw')
    ax[2,0].plot(np.mean(np.diff(data[:,1]))*allanDevNorm[:,0],1/(4*np.sqrt(2)*np.pi)*allanDevNorm[:,1]/np.mean(data[:,0]),label='Allan deviation, normalized')
    ax[2,0].plot(np.mean(np.diff(data[:,1]))*allanDevNorm2[:,0],1/(4*np.sqrt(2)*np.pi)*allanDevNorm2[:,1]/(np.mean(data[:,0])/(np.mean(data[:,3]))),label='Allan deviation, normalized 2')
    ax[2,0].set_yscale('log')
    ax[2,0].set_xscale('log')
    ax[2,0].legend()
    ax[2,0].set_xlabel('total acquisition time (s)')
    ax[2,0].set_ylabel('uncertainty (waves)')
    ax[2,0].set_title('estimated phase measurement uncertainty',fontweight='bold')
    ax[2,0].grid(which='major',linewidth='0.5',color='gray')
    ax[2,0].grid(which='minor',linewidth='0.5',color='lightgray')

    ax[0,1].plot(data[:,2]-data[:,1],label='pmeter')
    ax[0,1].plot(data[:,5]-data[:,4],label='pmeternorm')
    ax[0,1].set_ylabel('integration time (s)')
    ax[0,1].set_xlabel('measurement number')
    ax[0,1].legend()
    ax[0,1].set_title('power meter integration times',fontweight='bold')

    ax[1,1].plot(data[:,2]-data[:,5],label='end time mismatch')
    ax[1,1].plot(data[:,1]-data[:,4],label='start time mismatch')
    ax[1,1].set_ylabel('seconds')
    ax[1,1].set_xlabel('measurement number')
    ax[1,1].set_title('power measurement timing mismatches',fontweight='bold')
    ax[1,1].legend()

    ax[2,1].plot((np.abs(data[:,1]-data[:,4])+np.abs(data[:,2]-data[:,5]))/(data[:,2]-data[:,1]))
    ax[2,1].set_ylabel('fractional timing mismatch')
    ax[2,1].set_xlabel('measurement number')
    ax[2,1].set_title('relative timing mismatches',fontweight='bold')

    f.suptitle(fname,fontweight='bold',fontsize=14)
    plt.tight_layout()

    plt.show()

    return data

# spin a stage and measure power from a powermeter
def rotateAndCount(stage,start,end,stepSize,pm,countNum):
    angles = np.arange(start,end,stepSize)
    nAngles = np.size(angles)
    powers = np.zeros((nAngles,3))

    setCountTime([pm], countNum)

    for n in np.arange(nAngles):

        curPos = stage.getPos()
        if angles[n]-curPos > 0:
            print(f'moving clockwise from {curPos} to {angles[n]}')
            stage.move(angles[n]-curPos)
        else:
            print(f'moving counterclockwise from {curPos} to {angles[n]-5}')
            stage.move(angles[n]-curPos-5)
            print(f'now moving clockwise to {angles[n]}')
            stage.move(angles[n]-stage.getPos())


        time.sleep(0.100)

        power = getPower(pm)
        stgAngle = stage.getPos()

        powers[n,0] = angles[n]
        powers[n,1] = stgAngle
        powers[n,2] = power[0]
        print('power', power[0])

    return powers

def measZerosLoop(fname,stage,zero1,zero2,range,stepSize,pm,countNum,nLoops):

    for n in np.arange(nLoops):
        stage.mHome()


        data1 = rotateAndCount(stage,zero1-range,zero1+range,stepSize,pm,countNum)
        data2 = rotateAndCount(stage,zero2-range,zero2+range,stepSize,pm,countNum)
    
        data  =  np.vstack((data1,data2))

        if n == 0:
            np.savetxt(fname,data,delimiter=',')
            dataNew = data
        else:
            dataSaved = np.genfromtxt(fname,delimiter=',')
            dataNew = np.hstack((dataSaved,data[:,1:]))
            np.savetxt(fname,dataNew,delimiter=',')
         
    return dataNew

def measZerosLoopNorm(fname,stage,zero1,zero2,range,stepSize,pm,pmNorm,countNum,nLoops):

    for n in np.arange(nLoops):
        stage.mHome()


        data1 = rotateAndCountNorm(stage,zero1-range,zero1+range,stepSize,pm,pmNorm,countNum)
        data2 = rotateAndCountNorm(stage,zero2-range,zero2+range,stepSize,pm,pmNorm,countNum)
    
        data  =  np.vstack((data1,data2))

        if n == 0:
            np.savetxt(fname,data,delimiter=',')
        else:
            dataSaved = np.genfromtxt(fname,delimiter=',')
            dataNew = np.hstack((dataSaved,data[:,1:]))
            np.savetxt(fname,dataNew,delimiter=',')
         
    return dataNew

def combineMultipleSingleZerosScans(path,nameRoot):
    allFiles = glob.glob(path+nameRoot+'*.csv')
    allFiles.sort()

    for i, fname in enumerate(allFiles):
        if i==0:
            dataTotal = np.genfromtxt(fname,delimiter=',')
        else:
            data = np.genfromtxt(fname,delimiter=',')
            dataTotal = np.hstack((dataTotal,data[:,1:]))
        
    np.savetxt(path+'combined_'+nameRoot+'.csv', dataTotal,delimiter=',')

def analyseDSFLoop(fname, norm=False, analyseNorm=False, plots=True):

    allData = np.genfromtxt(fname,delimiter=',')

    if norm == True:
        dataShape = np.shape(allData)
        nMeas = int((dataShape[1])/8)

        reshapedData = np.zeros((dataShape[0],int(1+2*nMeas)))
        reshapedData[:,0] = allData[:,0]

        anglesColumnIdx = np.arange(1,int(1+8*nMeas),8)
        reshapedAnglesColumnIdx = np.arange(1,int(1+2*nMeas),2)

        reshapedData[:,reshapedAnglesColumnIdx] = allData[:,anglesColumnIdx]

        powersColumnIdx = np.arange(2,int(1+8*nMeas),8)
        powersNormColumnIdx = np.arange(5,int(1+8*nMeas),8)
        reshapedPowersColumnIdx = np.arange(2,int(1+2*nMeas),2)

        if analyseNorm == True:
            reshapedData[:,reshapedPowersColumnIdx] = allData[:,powersColumnIdx]/allData[:,powersNormColumnIdx]*np.mean(allData[:,powersNormColumnIdx])

        elif analyseNorm == False:
            reshapedData[:,reshapedPowersColumnIdx] = allData[:,powersColumnIdx]
        
        allData = reshapedData

    dataShape = np.shape(allData)
    lDataset = dataShape[0]
    print(lDataset)
    nDatasets = (dataShape[1]-1)/2
    print(dataShape)

    print(f'{nDatasets} total datasets')

    z1Guess = allData[int(lDataset/4),0]
    z2Guess = allData[int(3*lDataset/4),0]

    Cguess = np.min(allData[:,2])
    Aguess = (allData[0,2]-allData[int(lDataset/4),2])/((allData[0,0]-allData[int(lDataset/4),0])**2)

    print(z1Guess,z2Guess,Cguess,Aguess)

    results = np.zeros((int(nDatasets),6))

    for n in np.arange(nDatasets):
        data1 = allData[0:int(lDataset/2),(2*int(n)+1):(2*int(n)+3)]
        data2 = allData[int(lDataset/2):,(2*int(n)+1):(2*int(n)+3)]

        params1, cov1 = curve_fit(parabola,data1[:,0],data1[:,1],p0=[Aguess,z1Guess,Cguess],sigma=(0.05*data1[:,1]))

        ''' save fit params to act as initial guess for next fit '''
        Aguess = params1[0]
        z1Guess = params1[1]
        Cguess = params1[2]
 
        params2, cov2 = curve_fit(parabola,data2[:,0],data2[:,1],p0=[Aguess,z2Guess,Cguess],sigma=(0.05*data2[:,1]))

        z2Guess = params2[1]


        pol2Zero = 23.2634

        phase1 = ((-90-2*(params1[1]-pol2Zero))+180)%360 - 180
        phase2 = ((90+2*(params2[1]-pol2Zero))+180)%360 -180

        results[int(n),:] = np.array([phase1,np.sqrt(cov1[1,1]),phase2,np.sqrt(cov2[1,1]),(phase1+phase2)/2,np.sqrt(cov1[1,1]+cov2[1,1])])

        if n==0:
            data1First = data1
            data2First = data2
            params1First = params1
            cov1First = cov1
            params2First = params2
            cov2First = cov2

            print(f'first fit: z1: {params1[1]} +/- {np.sqrt(cov1[1,1])}')
        
        if n==nDatasets-1:
            data1Last = data1
            data2Last = data2
            params1Last = params1
            cov1Last = cov1
            params2Last = params2
            cov2Last = cov2
            print(f'last fit: z1: {params1[1]} +/- {np.sqrt(cov1[1,1])}')


    print(f'zero1: {np.mean(results[:,0]):.7} +/- {np.std(results[:,0]):.5}')
    print(f'zero2: {np.mean(results[:,2]):.7} +/- {np.std(results[:,2]):.5}')
    print(f'zero2-zero1: {np.mean(results[:,4]):.7} +/- {np.std(results[:,4]):.5}')

    print(f'zero2-zero1 co-std dev: {np.sqrt(np.abs(np.cov(np.transpose(results[:,[0,2]]))))}')
    print(f'zero2-zero1 correlation coeff: {np.corrcoef(np.transpose(results[:,[0,2]]))[0,1]:.5}')

    if nDatasets > 1:
        # fit zero positions to linear line
        linParams1, cov1 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,0])
        linParams2, cov2 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,2])
        linParams3, cov3 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,4])

        print(f'zero1 slope: {linParams1[0]:.5}')
        print(f'zero2 slope: {linParams2[0]:.5}')
        print(f'mean slope: {(linParams1[0]+linParams2[0])/2:.5}')
        print(f'correction factor based on slope: {360/(360+(linParams1[0]+linParams2[0])/2)}')
        print(f'correction factor based on slope: {(360+(linParams1[0]+linParams2[0])/2)/360}')

        print(f'correction factor based on period: {180/(np.mean(results[:,4]))}')
        print(f'correction factor based on period: {(np.mean(results[:,4]))/180}')

    ''' plot results '''
    if plots:
        nbins=np.max([10,int(np.sqrt(nDatasets))])
        f,ax = plt.subplots(3,3,figsize =(16,10))
        ax[0,0].hist(results[:,0],bins=nbins,label=f'+45 : {np.mean(results[:,0]):.7} +/- {np.std(results[:,0]):.3}')
        ax[0,1].hist(results[:,2],bins=nbins,label=f'zero2: {np.mean(results[:,2]):.7} +/- {np.std(results[:,2]):.3}')
        ax[0,2].hist(results[:,4],bins=nbins,label=f'hwp phase meas: {np.mean(results[:,4]):.7} +/- {np.std(results[:,4]):.3}')
        ax[1,0].plot(results[:,0],results[:,2],'.',label=f'correlation coeff: {np.corrcoef(np.transpose(results[:,[0,2]]))[0,1]:.5}')
        ax[1,1].plot(results[:,0]-np.mean(results[:,0]),'.',label=f'+45, mean fit error:{np.mean(results[:,1]):.3}')
        ax[1,1].plot(results[:,2]-np.mean(results[:,2]),'.',label=f'-45, mean fit error:{np.mean(results[:,3]):.3}')
        if nDatasets > 1:
            ax[1,1].plot(linParams1[0]*(np.arange(nDatasets)-(nDatasets-1)/2),label=f'fit 1: slope = {linParams1[0]:.3} +/- {np.sqrt(cov1[0,0]):.3}')
            ax[1,1].plot(linParams2[0]*(np.arange(nDatasets)-(nDatasets-1)/2),label=f'fit 2: slope = {linParams2[0]:.3} +/- {np.sqrt(cov2[0,0]):.3}')
        ax[1,2].plot(results[:,4],'.',label=f'data, mean fit error:{np.mean(results[:,5]):.3}')
        if nDatasets > 1:
            ax[1,2].plot(linParams3[0]*(np.arange(nDatasets))+linParams3[1],label=f'fit: slope = {linParams3[0]:.3} +/- {np.sqrt(cov2[0,0]):.3}')

        ax[0,0].legend()
        ax[0,1].legend()
        ax[0,2].legend()
        ax[1,0].legend()
        ax[1,1].legend()
        ax[1,2].legend()

        ax[0,0].set_title('+45 zero position distribution', fontweight='bold')
        ax[0,1].set_title('-45 zero position distribution', fontweight='bold')
        ax[0,2].set_title('HWP phase measurement', fontweight='bold')

        ax[1,0].set_title('+45 <--> -45 zeros correlation', fontweight='bold')
        ax[1,1].set_title('zero positions \'time\' dependence', fontweight='bold')
        ax[1,2].set_title('HWP phase measurement \'time\' dependence', fontweight='bold')

        ax[2,0].set_title('+45 fit performance', fontweight='bold')
        ax[2,1].set_title('-45 fit performance', fontweight='bold')

        ax[0,0].set_xlabel('+45 birefringence (degrees)')
        ax[0,1].set_xlabel('-45 birefringence (degrees)')
        ax[0,2].set_xlabel('measured HWP phase (degrees)')
        ax[1,0].set_xlabel('+45 zero position (degrees)')
        ax[1,0].set_ylabel('-45 zero position (degrees)')
        ax[1,1].set_xlabel('dataset number')
        ax[1,1].set_ylabel('offset zero position (degrees)')
        ax[1,2].set_xlabel('dataset number')
        ax[1,2].set_ylabel('zero 2 - zero 1 position (degrees)')

        data1Dense = np.arange(data1First[0,0],data1First[-1,0],0.001)
        data2Dense = np.arange(data2First[0,0],data2First[-1,0],0.001)

        ax[2,0].plot(data1First[:,0],data1First[:,1],'.',label='first data set')
        ax[2,0].plot(data1Last[:,0],data1Last[:,1],'.',label='last data set')
        ax[2,0].plot(data1Dense,parabola(data1Dense,*params1First),label='first fit')
        ax[2,0].plot(data1Dense,parabola(data1Dense,*params1Last),label='last fit')
        ax[2,0].grid(which='major',linewidth='0.5',color='gray')

        ax[2,1].plot(data2First[:,0],data2First[:,1],'.',label='first data set')
        ax[2,1].plot(data2Last[:,0],data2Last[:,1],'.',label='last data set')
        ax[2,1].plot(data2Dense,parabola(data2Dense,*params2First),label='first fit')
        ax[2,1].plot(data2Dense,parabola(data2Dense,*params2Last),label='last fit')

        ax[2,0].legend()
        ax[2,1].legend()
        ax[2,0].set_xlabel('bellMotors\' stage position (degrees)')
        ax[2,1].set_xlabel('bellMotors\' stage position (degrees)')
        ax[2,0].set_ylabel('power (W)')
        ax[2,1].set_ylabel('power (W)')

        print(nDatasets)

        if nDatasets >= 6:
            # calculate and plot Allan deviations of zero positions and zero-position separation
            allanDevZ1 = calcAllanDev(results[:,0])
            allanDevZ2 = calcAllanDev(results[:,2])
            allanDevZsep = calcAllanDev(results[:,4])

            ax[2,2].plot(allanDevZ1[:,0],allanDevZ1[:,1],label='+45 zero')
            ax[2,2].plot(allanDevZ2[:,0],allanDevZ2[:,1],label='-45 zero')
            ax[2,2].plot(allanDevZsep[:,0],allanDevZsep[:,1],label='HWP biref')
            ax[2,2].set_xscale('log')
            ax[2,2].set_yscale('log')
            ax[2,2].grid(which='major',linewidth='0.5',color='gray')
            ax[2,2].grid(which='minor',linewidth='0.5',color='lightgray')
            ax[2,2].set_xlabel('number of measurements')
            ax[2,2].set_ylabel('position uncertainty (degrees)')
            ax[2,2].legend()
            ax[2,2].set_title('Allan deviations', fontweight='bold')
        
        f.suptitle(fname+'\n',fontweight = 'bold',fontsize=14)

        if norm == True:
            if analyseNorm == False:
                f.suptitle(fname+'    raw data\n',fontweight = 'bold',fontsize=14)
            elif analyseNorm == True:
                f.suptitle(fname+'    normalized data\n',fontweight = 'bold',fontsize=14)


        plt.tight_layout()

        plt.show()

    return allData, results

def analyseFourPowersLoop(fname, zeroBiref=False, plots=True):

    allData = np.genfromtxt(fname,delimiter=',')

    # background-subtract normalization measurements
    allData[:,3::6] = allData[:,3::6] - 0.000000080

    dataShape = np.shape(allData)
    nMeas = int((dataShape[1])/6)

    nDatasets = (dataShape[1])/6
    print(f'{nDatasets} total datasets')

    results = np.zeros((int(nDatasets),6))

    for n in np.arange(int(nDatasets)):
        pRaw = allData[:,6*n]
        pNorm = allData[:,6*n]/allData[:,6*n+3]*np.mean(allData[:,6*n+3])

        print(pRaw)
        print(pNorm)

        # QWP at +45 degrees
        phiRaw = 180 + 180/np.pi*(pRaw[1]-pRaw[0])/(pRaw[0]+pRaw[1]) 
        phiNorm = 180 + 180/np.pi*(pNorm[1]-pNorm[0])/(pNorm[0]+pNorm[1])
        phiRaw = 180 + 180/np.pi*np.arcsin((pRaw[1]-pRaw[0])/(pRaw[0]+pRaw[1]))
        phiNorm = 180 + 180/np.pi*np.arcsin((pNorm[1]-pNorm[0])/(pNorm[0]+pNorm[1]))
        results[n,0] = phiRaw
        results[n,1] = phiNorm

        # QWP at -45 degrees
        phiRaw = 180 + 180/np.pi*(pRaw[2]-pRaw[3])/(pRaw[2]+pRaw[3])
        phiNorm = 180 + 180/np.pi*(pNorm[2]-pNorm[3])/(pNorm[2]+pNorm[3])
        phiRaw = 180 + 180/np.pi*np.arcsin((pRaw[2]-pRaw[3])/(pRaw[2]+pRaw[3]))
        phiNorm = 180 + 180/np.pi*np.arcsin((pNorm[2]-pNorm[3])/(pNorm[2]+pNorm[3]))
        results[n,2] = phiRaw
        results[n,3] = phiNorm

        results[n,4] = 180 + 90*(np.arcsin((pRaw[1]-pRaw[0])/(pRaw[0]+pRaw[1]))/np.pi + np.arcsin((pRaw[2]-pRaw[3])/(pRaw[2]+pRaw[3]))/np.pi)
        results[n,5] = 180 + 90*(np.arcsin((pNorm[1]-pNorm[0])/(pNorm[0]+pNorm[1]))/np.pi + np.arcsin((pNorm[2]-pNorm[3])/(pNorm[2]+pNorm[3]))/np.pi)


    if zeroBiref == True:
        results = results - 180

    print(f'phi1Raw: {np.mean(results[:,0]):.7} +/- {np.std(results[:,0]):.5}')
    print(f'phi1Norm: {np.mean(results[:,1]):.7} +/- {np.std(results[:,1]):.5}')

    print(f'phi2Raw: {np.mean(results[:,2]):.7} +/- {np.std(results[:,2]):.5}')
    print(f'phi2Norm: {np.mean(results[:,3]):.7} +/- {np.std(results[:,3]):.5}')
    
    print(f'phiAvgRaw: {np.mean(results[:,4]):.7} +/- {np.std(results[:,4]):.5}')
    print(f'phiAvgNorm: {np.mean(results[:,5]):.7} +/- {np.std(results[:,5]):.5}')
    
    if nDatasets > 1:
        # fit zero positions to linear line
        print(np.shape(results))
        print(np.shape(np.arange(nDatasets)))
        linParams1, cov1 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,0])
        linParams2, cov2 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,1])

        print(f'phiRaw slope: {linParams1[0]:.5}')
        print(f'phiNorm slope: {linParams2[0]:.5}')
        print(f'mean slope: {(linParams1[0]+linParams2[0])/2:.5}')


    if plots:
        nbins=np.max([10,int(np.sqrt(nDatasets))])
        f,ax = plt.subplots(3,3,figsize =(16,10))
        ax[0,0].hist(results[:,0],bins=nbins,label=f'raw data: {np.mean(results[:,0]):.7} +/- {np.std(results[:,0]):.3}')
        ax[0,0].hist(results[:,1],bins=nbins,label=f'norm data: {np.mean(results[:,1]):.7} +/- {np.std(results[:,1]):.3}')
        ax[0,1].hist(results[:,2],bins=nbins,label=f'raw data: {np.mean(results[:,2]):.7} +/- {np.std(results[:,2]):.3}')
        ax[0,1].hist(results[:,3],bins=nbins,label=f'norm data: {np.mean(results[:,3]):.7} +/- {np.std(results[:,3]):.3}')
        ax[0,2].hist(results[:,4],bins=nbins,label=f'raw data: {np.mean(results[:,4]):.7} +/- {np.std(results[:,4]):.3}')
        ax[0,2].hist(results[:,5],bins=nbins,label=f'norm data: {np.mean(results[:,5]):.7} +/- {np.std(results[:,5]):.3}')
        ax[1,0].plot(results[:,0],results[:,2],'.',label=f'raw: phi +45 <--> -45 corr. coeff: {np.corrcoef(np.transpose(results[:,[0,2]]))[0,1]:.5}')
        ax[1,0].plot(results[:,1],results[:,3],'.',label=f'norm: phi +45 <--> -45 corr. coeff: {np.corrcoef(np.transpose(results[:,[1,3]]))[0,1]:.5}')
        ax[1,1].plot(results[:,1]-np.mean(results[:,1]),'.',label=f'phi+45')
        ax[1,1].plot(results[:,3]-np.mean(results[:,3]),'.',label=f'phi-45')
        ax[1,2].plot(results[:,4],'.',label='raw data')
        ax[1,2].plot(results[:,5],'.',label='norm data')
        # if nDatasets > 1:
        #     ax[1,1].plot(linParams1[0]*(np.arange(nDatasets)-(nDatasets-1)/2),label=f'fit 1: slope = {linParams1[0]:.3} +/- {np.sqrt(cov1[0,0]):.3}')
        #     ax[1,1].plot(linParams2[0]*(np.arange(nDatasets)-(nDatasets-1)/2),label=f'fit 2: slope = {linParams2[0]:.3} +/- {np.sqrt(cov2[0,0]):.3}')
        # ax[1,2].plot(results[:,4],'.',label=f'data, mean fit error:{np.mean(results[:,5]):.3}')
        # if nDatasets > 1:
        #     ax[1,2].plot(linParams3[0]*(np.arange(nDatasets))+linParams3[1],label=f'fit: slope = {linParams3[0]:.3} +/- {np.sqrt(cov2[0,0]):.3}')

        ax[0,0].legend()
        ax[0,1].legend()
        ax[0,2].legend()
        ax[1,0].legend()
        ax[1,1].legend()
        ax[1,2].legend()

        ax[0,0].set_title('phi+45 distribution', fontweight='bold')
        ax[0,1].set_title('phi-45 distribution', fontweight='bold')
        ax[0,2].set_title('HWP phase measurement distribution', fontweight='bold')

        ax[1,0].set_title('phi+45 <--> phi-45 correlation', fontweight='bold')
        ax[1,1].set_title('phi meas \'time\' dependence', fontweight='bold')
        ax[1,2].set_title('HWP phase meas \'time\' dependence', fontweight='bold')

        ax[2,0].set_title('+45 data performance', fontweight='bold')
        ax[2,1].set_title('-45 data performance', fontweight='bold')

        ax[0,0].set_xlabel('birefringence (degrees)')
        ax[0,1].set_xlabel('birefringence (degrees)')
        ax[0,2].set_xlabel('measured HWP phase (degrees)')
        ax[1,0].set_xlabel('+45 birefringence (degrees)')
        ax[1,0].set_ylabel('-45 birefringence (degrees)')
        ax[1,1].set_xlabel('dataset number')
        ax[1,1].set_ylabel('offset birefringence (degrees)')
        ax[1,2].set_xlabel('dataset number')
        ax[1,2].set_ylabel('HWP phase (degrees)')

        ax[2,0].plot(allData[0,::6]/(allData[0,::6]+allData[1,::6]),'.',label='raw H')
        ax[2,0].plot(allData[1,::6]/(allData[0,::6]+allData[1,::6]),'.',label='raw V')
        ax[2,1].plot(allData[2,::6]/(allData[2,::6]+allData[3,::6]),'.',label='raw H')
        ax[2,1].plot(allData[3,::6]/(allData[2,::6]+allData[3,::6]),'.',label='raw V')
        
        ax[2,0].plot(allData[0,::6]/allData[0,3::6]/((allData[0,::6]/allData[0,3::6]+allData[1,::6]/allData[1,3::6])),'.',label='norm H')
        ax[2,0].plot(allData[1,::6]/allData[1,3::6]/((allData[0,::6]/allData[0,3::6]+allData[1,::6]/allData[1,3::6])),'.',label='norm V')
        ax[2,1].plot(allData[2,::6]/allData[2,3::6]/((allData[2,::6]/allData[2,3::6]+allData[3,::6]/allData[3,3::6])),'.',label='norm H')
        ax[2,1].plot(allData[3,::6]/allData[3,3::6]/((allData[2,::6]/allData[2,3::6]+allData[3,::6]/allData[3,3::6])),'.',label='norm V')

        ax[2,0].legend()
        ax[2,1].legend()
        ax[2,0].set_xlabel('dataset number')
        ax[2,1].set_xlabel('dataset number')
        ax[2,0].set_ylabel('transmission probability')
        ax[2,1].set_ylabel('transmission probability')

        print(nDatasets)

        if nDatasets >= 6:
            # calculate and plot Allan deviations of zero positions and zero-position separation
            allanDevZ1 = calcAllanDev(results[:,1])
            allanDevZ2 = calcAllanDev(results[:,3])
            allanDevZsep = calcAllanDev(results[:,5])

            ax[2,2].plot(allanDevZ1[:,0],allanDevZ1[:,1],label='+45')
            ax[2,2].plot(allanDevZ2[:,0],allanDevZ2[:,1],label='-45')
            ax[2,2].plot(allanDevZsep[:,0],allanDevZsep[:,1],label='mean')
            ax[2,2].set_xscale('log')
            ax[2,2].set_yscale('log')
            ax[2,2].grid(which='major',linewidth='0.5',color='gray')
            ax[2,2].grid(which='minor',linewidth='0.5',color='lightgray')
            ax[2,2].set_xlabel('number of measurements')
            ax[2,2].set_ylabel('birefringence uncertainty (degrees)')
            ax[2,2].legend()
            ax[2,2].set_title('Allan deviations', fontweight='bold')
        
        f.suptitle(fname+'\n',fontweight = 'bold',fontsize=14)

        plt.tight_layout()

        plt.show()

    return allData, results



    allData = np.genfromtxt(fname,delimiter=',')

    if norm == True:
        dataShape = np.shape(allData)
        nMeas = int((dataShape[1]-1)/8)

        reshapedData = np.zeros((dataShape[0],int(1+2*nMeas)))
        reshapedData[:,0] = allData[:,0]

        anglesColumnIdx = np.arange(1,int(1+8*nMeas),8)
        reshapedAnglesColumnIdx = np.arange(1,int(1+2*nMeas),2)

        reshapedData[:,reshapedAnglesColumnIdx] = allData[:,anglesColumnIdx]

        powersColumnIdx = np.arange(2,int(1+8*nMeas),8)
        powersNormColumnIdx = np.arange(5,int(1+8*nMeas),8)
        reshapedPowersColumnIdx = np.arange(2,int(1+2*nMeas),2)

        if analyseNorm == True:
            reshapedData[:,reshapedPowersColumnIdx] = allData[:,powersColumnIdx]/allData[:,powersNormColumnIdx]*np.mean(allData[:,powersNormColumnIdx])

        elif analyseNorm == False:
            reshapedData[:,reshapedPowersColumnIdx] = allData[:,powersColumnIdx]
        
        allData = reshapedData

    dataShape = np.shape(allData)
    lDataset = dataShape[0]
    print(lDataset)
    nDatasets = (dataShape[1]-1)/2
    print(dataShape)

    print(f'{nDatasets} total datasets')

    z1Guess = allData[int(lDataset/4),0]
    z2Guess = allData[int(3*lDataset/4),0]

    Cguess = np.min(allData[:,2])
    Aguess = (allData[0,2]-allData[int(lDataset/4),2])/((allData[0,0]-allData[int(lDataset/4),0])**2)

    print(z1Guess,z2Guess,Cguess,Aguess)

    results = np.zeros((int(nDatasets),6))

    for n in np.arange(nDatasets):
        print(f'n = {n}')
        data1 = allData[0:int(lDataset/2),(2*int(n)+1):(2*int(n)+3)]
        data2 = allData[int(lDataset/2):,(2*int(n)+1):(2*int(n)+3)]

        params1, cov1 = curve_fit(parabola,data1[:,0],data1[:,1],p0=[Aguess,z1Guess,Cguess],sigma=(0.05*data1[:,1]))
        print('p1 fit')

        ''' save fit params to act as initial guess for next fit '''
        Aguess = params1[0]
        z1Guess = params1[1]
        Cguess = params1[2]
 
        params2, cov2 = curve_fit(parabola,data2[:,0],data2[:,1],p0=[Aguess,z2Guess,Cguess],sigma=(0.05*data2[:,1]))
        print('p2 fit')

        z2Guess = params2[1]

        results[int(n),:] = np.array([params1[1],np.sqrt(cov1[1,1]),params2[1],np.sqrt(cov2[1,1]),(params2[1]-params1[1])/2+180,np.sqrt(cov1[1,1]+cov2[1,1])])

        if n==0:
            data1First = data1
            data2First = data2
            params1First = params1
            cov1First = cov1
            params2First = params2
            cov2First = cov2

            print(f'first fit: z1: {params1[1]} +/- {np.sqrt(cov1[1,1])}')
        
        if n==nDatasets-1:
            data1Last = data1
            data2Last = data2
            params1Last = params1
            cov1Last = cov1
            params2Last = params2
            cov2Last = cov2
            print(f'last fit: z1: {params1[1]} +/- {np.sqrt(cov1[1,1])}')


    print(f'zero1: {np.mean(results[:,0]):.7} +/- {np.std(results[:,0]):.5}')
    print(f'zero2: {np.mean(results[:,2]):.7} +/- {np.std(results[:,2]):.5}')
    print(f'zero2-zero1: {np.mean(results[:,4]):.7} +/- {np.std(results[:,4]):.5}')

    print(f'zero2-zero1 co-std dev: {np.sqrt(np.abs(np.cov(np.transpose(results[:,[0,2]]))))}')
    print(f'zero2-zero1 correlation coeff: {np.corrcoef(np.transpose(results[:,[0,2]]))[0,1]:.5}')

    if nDatasets > 1:
        # fit zero positions to linear line
        linParams1, cov1 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,0])
        linParams2, cov2 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,2])
        linParams3, cov3 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,4])

        print(f'zero1 slope: {linParams1[0]:.5}')
        print(f'zero2 slope: {linParams2[0]:.5}')
        print(f'mean slope: {(linParams1[0]+linParams2[0])/2:.5}')
        print(f'correction factor based on slope: {360/(360+(linParams1[0]+linParams2[0])/2)}')
        print(f'correction factor based on slope: {(360+(linParams1[0]+linParams2[0])/2)/360}')

        print(f'correction factor based on period: {180/(np.mean(results[:,4]))}')
        print(f'correction factor based on period: {(np.mean(results[:,4]))/180}')

    nbins=np.max([10,int(np.sqrt(nDatasets))])
    f,ax = plt.subplots(3,3,figsize =(16,10))
    ax[0,0].hist(results[:,0],bins=nbins,label=f'zero1: {np.mean(results[:,0]):.7} +/- {np.std(results[:,0]):.3}')
    ax[0,1].hist(results[:,2],bins=nbins,label=f'zero2: {np.mean(results[:,2]):.7} +/- {np.std(results[:,2]):.3}')
    ax[0,2].hist(results[:,4],bins=nbins,label=f'zero2-zero1: {np.mean(results[:,4]):.7} +/- {np.std(results[:,4]):.3}')
    ax[1,0].plot(results[:,0],results[:,2],'.',label=f'zero2-zero1 correlation coeff: {np.corrcoef(np.transpose(results[:,[0,2]]))[0,1]:.5}')
    ax[1,1].plot(results[:,0]-np.mean(results[:,0]),'.',label=f'zero1, mean fit error:{np.mean(results[:,1]):.3}')
    ax[1,1].plot(results[:,2]-np.mean(results[:,2]),'.',label=f'zero2, mean fit error:{np.mean(results[:,3]):.3}')
    if nDatasets > 1:
        ax[1,1].plot(linParams1[0]*(np.arange(nDatasets)-(nDatasets-1)/2),label=f'fit 1: slope = {linParams1[0]:.3} +/- {np.sqrt(cov1[0,0]):.3}')
        ax[1,1].plot(linParams2[0]*(np.arange(nDatasets)-(nDatasets-1)/2),label=f'fit 2: slope = {linParams2[0]:.3} +/- {np.sqrt(cov2[0,0]):.3}')
    ax[1,2].plot(results[:,4],'.',label=f'data, mean fit error:{np.mean(results[:,5]):.3}')
    if nDatasets > 1:
        ax[1,2].plot(linParams3[0]*(np.arange(nDatasets))+linParams3[1],label=f'fit: slope = {linParams3[0]:.3} +/- {np.sqrt(cov2[0,0]):.3}')

    ax[0,0].legend()
    ax[0,1].legend()
    ax[0,2].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    ax[1,2].legend()

    ax[0,0].set_title('zero 1 position distribution', fontweight='bold')
    ax[0,1].set_title('zero 2 position distribution', fontweight='bold')
    ax[0,2].set_title('HWP phase measurement', fontweight='bold')

    ax[1,0].set_title('zero 1 <--> zero 2 correlation', fontweight='bold')
    ax[1,1].set_title('zero positions \'time\' dependence', fontweight='bold')
    ax[1,2].set_title('zero1 - zero2 distance \'time\' dependence', fontweight='bold')

    ax[2,0].set_title('zero 1 fit performance', fontweight='bold')
    ax[2,1].set_title('zero 2 fit performance', fontweight='bold')

    ax[0,0].set_xlabel('zero 1 position (degrees)')
    ax[0,1].set_xlabel('zero 2 position (degrees)')
    ax[0,2].set_xlabel('measured HWP phase (degrees)')
    ax[1,0].set_xlabel('zero 1 position (degrees)')
    ax[1,0].set_ylabel('zero 2 position (degrees)')
    ax[1,1].set_xlabel('dataset number')
    ax[1,1].set_ylabel('offset zero position (degrees)')
    ax[1,2].set_xlabel('dataset number')
    ax[1,2].set_ylabel('zero 2 - zero 1 position (degrees)')

    data1Dense = np.arange(data1First[0,0],data1First[-1,0],0.001)
    data2Dense = np.arange(data2First[0,0],data2First[-1,0],0.001)

    ax[2,0].plot(data1First[:,0],data1First[:,1],'.',label='first data set')
    ax[2,0].plot(data1Last[:,0],data1Last[:,1],'.',label='last data set')
    ax[2,0].plot(data1Dense,parabola(data1Dense,*params1First),label='first fit')
    ax[2,0].plot(data1Dense,parabola(data1Dense,*params1Last),label='last fit')
    ax[2,0].plot(data1First[:-1,0]+np.diff(data1First[:,0]),np.diff(data1First[:,1]),'.')
    ax[2,0].grid(which='major',linewidth='0.5',color='gray')

    ax[2,1].plot(data2First[:,0],data2First[:,1],'.',label='first data set')
    ax[2,1].plot(data2Last[:,0],data2Last[:,1],'.',label='last data set')
    ax[2,1].plot(data2Dense,parabola(data2Dense,*params2First),label='first fit')
    ax[2,1].plot(data2Dense,parabola(data2Dense,*params2Last),label='last fit')

    ax[2,0].legend()
    ax[2,1].legend()
    ax[2,0].set_xlabel('bellMotors\' stage position (degrees)')
    ax[2,1].set_xlabel('bellMotors\' stage position (degrees)')
    ax[2,0].set_ylabel('power (W)')
    ax[2,1].set_ylabel('power (W))')

    print(nDatasets)

    if nDatasets >= 6:
        # calculate and plot Allan deviations of zero positions and zero-position separation
        allanDevZ1 = calcAllanDev(results[:,0])
        allanDevZ2 = calcAllanDev(results[:,2])
        allanDevZsep = calcAllanDev(results[:,4])

        ax[2,2].plot(allanDevZ1[:,0],allanDevZ1[:,1],label='zero 1')
        ax[2,2].plot(allanDevZ2[:,0],allanDevZ2[:,1],label='zero 2')
        ax[2,2].plot(allanDevZsep[:,0],allanDevZsep[:,1],label='zeros\' separation')
        ax[2,2].set_xscale('log')
        ax[2,2].set_yscale('log')
        ax[2,2].grid(which='major',linewidth='0.5',color='gray')
        ax[2,2].grid(which='minor',linewidth='0.5',color='lightgray')
        ax[2,2].set_xlabel('number of measurements')
        ax[2,2].set_ylabel('position uncertainty (degrees)')
        ax[2,2].legend()
        ax[2,2].set_title('zero position Allan deviations', fontweight='bold')
    
    f.suptitle(fname+'\n',fontweight = 'bold',fontsize=14)

    if norm == True:
        if analyseNorm == False:
            f.suptitle(fname+'    raw data\n',fontweight = 'bold',fontsize=14)
        elif analyseNorm == True:
            f.suptitle(fname+'    normalized data\n',fontweight = 'bold',fontsize=14)


    plt.tight_layout()

    plt.show()

    return allData, results

def compareFourPowersAndDSFMethods(fnameDSF,fnameFourPower,zeroDSF,zeroFourPower,zeroBiref=False):
    dDSF,resDSF = analyseDSFLoop(fnameDSF, norm=True, analyseNorm=True)
    dFP,resFP = analyseFourPowersLoop(fnameFourPower,zeroBiref)

    if np.size(resFP) > np.size(resDSF):
        resFP = resFP[:-1,:]
    
    birefDSF = resDSF[:,4]
    birefFP = resFP[:,5]
    resultsBoth = np.vstack((birefDSF,birefFP))
    resultsBothCorr = np.vstack((birefDSF-zeroDSF,birefFP-zeroFourPower))

    minMeas = np.min([np.min(resultsBoth),np.min(resultsBothCorr)])
    maxMeas = np.max([np.max(resultsBoth),np.max(resultsBothCorr)])

    f,ax = plt.subplots(2,2,figsize =(9,10))

    ax[0,0].plot(resultsBoth[0,:],resultsBoth[1,:],'.',label='individual measurement pairs')
    ax[0,0].plot(resultsBoth[0,:]-zeroDSF,resultsBoth[1,:]-zeroFourPower,'.',label='zero-corrected pairs')
    ax[0,0].plot([minMeas-0.1,maxMeas+0.1],[minMeas-0.1,maxMeas+0.1],label='y=x')
    ax[0,0].set_xlim([minMeas-0.1,maxMeas+0.1])
    ax[0,0].set_ylim([minMeas-0.1,maxMeas+0.1])
    ax[0,0].set_aspect('equal')
    ax[0,0].legend()
    ax[0,0].set_title(f'correlation = {np.corrcoef(resultsBoth)[0,1]:.3}',fontweight='bold')
    ax[0,0].set_xlabel('dSF result (degrees)')
    ax[0,0].set_ylabel('Boulder result (degrees)')

    ax[0,1].plot(resultsBoth[0,:]-resultsBoth[1,:],'.',label='measurement difference')
    ax[0,1].plot(resultsBothCorr[0,:]-resultsBothCorr[1,:],'.',label='zero-corrected difference')
    ax[0,1].plot([0,np.size(birefFP)],np.mean(resultsBoth[0,:]-resultsBoth[1,:])*np.ones(2),label='mean measurement diff')
    ax[0,1].plot([0,np.size(birefFP)],np.mean(resultsBothCorr[0,:]-resultsBothCorr[1,:])*np.ones(2),label='mean zero-corrected diff')
    ax[0,1].set_xlabel('measurement number')
    ax[0,1].set_ylabel('difference (degrees)')
    ax[0,1].set_title('measurement differences',fontweight='bold')
    ax[0,1].legend()

    ax[1,0].hist(resultsBoth[0,:]-resultsBoth[1,:],bins=np.max([10,int(np.sqrt(np.size(birefFP)))]),alpha=0.5,label=f'base: {np.mean(resultsBoth[0,:]-resultsBoth[1,:]):.3} +/- {np.std(resultsBoth[0,:]-resultsBoth[1,:]):.3}')
    ax[1,0].hist(resultsBothCorr[0,:]-resultsBothCorr[1,:],bins=np.max([10,int(np.sqrt(np.size(birefFP)))]),alpha=0.5,label=f'zero-corrected: {np.mean(resultsBothCorr[0,:]-resultsBothCorr[1,:]):.3} +/- {np.std(resultsBothCorr[0,:]-resultsBothCorr[1,:]):.3}')
    ax[1,0].set_title('difference distribution',fontweight='bold')
    ax[1,0].set_xlabel('measurement difference (degrees)')
    ax[1,0].legend()

    if np.size(birefFP) >= 6:
        # calculate and plot Allan deviations difference between two techniques
        allanDevDiff = calcAllanDev(birefDSF-birefFP)

        ax[1,1].plot(allanDevDiff[:,0],allanDevDiff[:,1],label='measurement difference')
        ax[1,1].set_xscale('log')
        ax[1,1].set_yscale('log')
        ax[1,1].grid(which='major',linewidth='0.5',color='gray')
        ax[1,1].grid(which='minor',linewidth='0.5',color='lightgray')
        ax[1,1].set_xlabel('number of measurements')
        ax[1,1].set_ylabel('difference uncertainty (degrees)')
        ax[1,1].legend()
        ax[1,1].set_title('Allan deviation', fontweight='bold')
    
    f.suptitle(fnameDSF+'\n'+fnameFourPower+'\n',fontweight = 'bold',fontsize=14)

    plt.tight_layout()
    plt.show()

def analyseBoulderDSFMethodsAllCombos(path,nameBoulderRoot,nameDSFRoot,norm=False,analyseNorm=False,zeroBiref=False):

    resBoulderSummary = np.zeros((2,2,2,2,3))
    resDSFSummary = np.zeros((2,2,2,2,3))

    # cycle through angle-flipping combinations
    for pol1Offset in np.arange(2):
        for qwp1Offset in np.arange(2):
            for pol2Offset in np.arange(2):
                # find filenames for this specific angle-flipping combination                    
                angleFlip = '_pol1_'+str(pol1Offset)+'_qwp1_'+str(qwp1Offset)+'_pol2_'+str(pol2Offset)

                fnameBoulder = path + nameBoulderRoot + angleFlip + '.csv'
                fnameDSF = path + nameDSFRoot + angleFlip + '.csv'

                print(fnameBoulder)
                print(fnameDSF)

                dDSF,resDSF = analyseDSFLoop(fnameDSF, norm, analyseNorm, plots=False)
                dFP,resFP = analyseFourPowersLoop(fnameBoulder,zeroBiref, plots=False)

                if (pol1Offset==0) & (qwp1Offset==0) & (pol2Offset==0):
                    allResultsFP = np.zeros((8,np.shape(resFP)[0],np.shape(resFP)[1]))
                    allResultsDSF = np.zeros((8,np.shape(resFP)[0],np.shape(resFP)[1]))

                # with no sample, sending in |A> polarization measures the negative of the birefringence
                if zeroBiref:
                    if pol1Offset == 0:
                        resFP = -resFP

                    if pol1Offset == 1:
                        resFP = resFP
                        resDSF = (resDSF-90)%180-90 # this shifts resDSF to be a number close to 0, instead of close to -180 or +180

                # with a sample, sending in |A> polarization measures 180-error, instead of 180 + error
                # with no sample, sending in |A> polarization measures the negative of the birefringence
                else:
                    if pol1Offset == 1:
                        resFP = 360 -resFP
                        resDSF[:,[0,2,4]] = 180+resDSF[:,[0,2,4]]

                # save birefringence measured from each individual dataset
                allResultsFP[int(str(pol1Offset)+str(qwp1Offset)+str(pol2Offset),base=2),0:(np.shape(resFP)[0]),0:(np.shape(resFP)[1])] = resFP
                allResultsDSF[int(str(pol1Offset)+str(qwp1Offset)+str(pol2Offset),base=2),0:np.shape(resDSF)[0],0:np.shape(resDSF)[1]] = resDSF

                print(np.shape(resFP[:,[1,3,5]]))
                print(np.shape(np.mean(resFP[:,[1,3,5]],axis=0)))

                print(np.shape(resDSF[:,[0,2,4]]))
                print(np.shape(np.mean(resDSF[:,[0,2,4]],axis=0)))

                resBoulderSummary[pol1Offset,qwp1Offset,pol2Offset,0,:] = np.mean(resFP[:,[1,3,5]],axis=0)
                resDSFSummary[pol1Offset,qwp1Offset,pol2Offset,0,:] = np.mean(resDSF[:,[0,2,4]],axis=0)

                resBoulderSummary[pol1Offset,qwp1Offset,pol2Offset,1,:] = np.std(resFP[:,[1,3,5]],axis=0,ddof=1)
                resDSFSummary[pol1Offset,qwp1Offset,pol2Offset,1,:] = np.std(resDSF[:,[0,2,4]],axis=0,ddof=1)

    f,ax = plt.subplots(3,3,figsize =(16,11))

    ''' average +45/-45 input state measurements with each other '''
    allResFPAvg = 1/2*(np.mean(allResultsFP[0:4,:],axis=0) +  np.mean(allResultsFP[4:8,:],axis=0))
    allResDSFAvg = 1/2*(np.mean(allResultsDSF[0:4],axis=0) + np.mean(allResultsDSF[4:8],axis=0))

    allFP00Avg = 1/2*(allResultsFP[0,0:np.shape(allResultsFP)[1]-1,5]+allResultsFP[4,0:np.shape(allResultsFP)[1]-1,5])
    allFP01Avg = 1/2*(allResultsFP[1,0:np.shape(allResultsFP)[1]-1,5]+allResultsFP[5,0:np.shape(allResultsFP)[1]-1,5])
    allFP10Avg = 1/2*(allResultsFP[2,0:np.shape(allResultsFP)[1]-1,5]+allResultsFP[6,0:np.shape(allResultsFP)[1]-1,5])
    allFP11Avg = 1/2*(allResultsFP[3,0:np.shape(allResultsFP)[1]-1,5]+allResultsFP[7,0:np.shape(allResultsFP)[1]-1,5])

    ax[0,0].hist(allFP00Avg,alpha=0.5,label=f'Boulder: {np.mean(allFP00Avg):.6} +/- {np.std(allFP00Avg,ddof=1):.3}')
    ax[0,1].hist(allFP01Avg,alpha=0.5,label=f'Boulder: {np.mean(allFP01Avg):.6} +/- {np.std(allFP01Avg,ddof=1):.3}')
    ax[1,0].hist(allFP10Avg,alpha=0.5,label=f'Boulder: {np.mean(allFP10Avg):.6} +/- {np.std(allFP10Avg,ddof=1):.3}')
    ax[1,1].hist(allFP11Avg,alpha=0.5,label=f'Boulder: {np.mean(allFP11Avg):.6} +/- {np.std(allFP11Avg,ddof=1):.3}')

    allDSF00Avg = 1/2*(allResultsDSF[0,0:np.shape(allResultsDSF)[1]-1,4]+allResultsDSF[4,0:np.shape(allResultsDSF)[1]-1,4])
    allDSF01Avg = 1/2*(allResultsDSF[1,0:np.shape(allResultsDSF)[1]-1,4]+allResultsDSF[5,0:np.shape(allResultsDSF)[1]-1,4])
    allDSF10Avg = 1/2*(allResultsDSF[2,0:np.shape(allResultsDSF)[1]-1,4]+allResultsDSF[6,0:np.shape(allResultsDSF)[1]-1,4])
    allDSF11Avg = 1/2*(allResultsDSF[3,0:np.shape(allResultsDSF)[1]-1,4]+allResultsDSF[7,0:np.shape(allResultsDSF)[1]-1,4])
             
    ax[0,0].hist(allDSF00Avg,alpha=0.5,label=f'DSF: {np.mean(allDSF00Avg):.6} +/- {np.std(allDSF00Avg,ddof=1):.3}')
    ax[0,1].hist(allDSF01Avg,alpha=0.5,label=f'DSF: {np.mean(allDSF01Avg):.6} +/- {np.std(allDSF01Avg,ddof=1):.3}')
    ax[1,0].hist(allDSF10Avg,alpha=0.5,label=f'DSF: {np.mean(allDSF10Avg):.6} +/- {np.std(allDSF10Avg,ddof=1):.3}')
    ax[1,1].hist(allDSF11Avg,alpha=0.5,label=f'DSF: {np.mean(allDSF11Avg):.6} +/- {np.std(allDSF11Avg,ddof=1):.3}')

    allFPAvg = 1/2*(allResultsFP[0:4,0:np.shape(allResultsFP)[1]-1,5]+allResultsFP[4:8,0:np.shape(allResultsFP)[1]-1,5]).flatten()
    allDSFAvg = 1/2*(allResultsDSF[0:4,0:np.shape(allResultsDSF)[1]-1,4]+allResultsDSF[4:8,0:np.shape(allResultsDSF)[1]-1,4]).flatten()
    ax[0,2].hist(allFPAvg,alpha=0.5,label=f'Boulder: {np.mean(allFPAvg):.6} +/- {np.std(allFPAvg,ddof=1):.3}')
    ax[0,2].hist(allDSFAvg,alpha=0.5,label=f'DSF: {np.mean(allDSFAvg):.6} +/- {np.std(allDSFAvg,ddof=1):.3}')

    ax[2,2].plot(allFPAvg,label='Boulder')
    ax[2,2].plot(allDSFAvg,label='dSF')

    # ax[1,2].plot(allDSFAvg,allFPAvg,'o',label='data')
    # dMin = np.min([np.min(allDSFAvg),np.min(allFPAvg)])
    # dMax = np.max([np.max(allDSFAvg),np.max(allFPAvg)])
    # bMin = dMin - 0.1*(dMax-dMin)
    # bMax = dMax + 0.1*(dMax-dMin)
    # ax[1,2].plot(np.array([bMin,bMax]),np.array([bMin,bMax]),label='y=x')
    # ax[1,2].set_xlim([bMin,bMax])
    # ax[1,2].set_ylim([bMin,bMax])
    # ax[1,2].set_aspect('equal')
    # ax[1,2].set_title('all qwp/pol2 combinations',fontweight = 'bold')
    # ax[1,2].set_xlabel('dSF biref (degrees)')
    # ax[1,2].set_ylabel('Boulder biref (degrees)')

    # calculate and plot Allan deviations difference between two techniques
    allanDevFP = calcAllanDev(allFPAvg)
    allanDevDSF = calcAllanDev(allDSFAvg)
    allanDevDiff = calcAllanDev(allFPAvg-allDSFAvg)

    ax[1,2].plot(allanDevFP[:,0],allanDevFP[:,1],label='boulder')
    ax[1,2].plot(allanDevDSF[:,0],allanDevDSF[:,1],label='dSF')
    ax[1,2].plot(allanDevDiff[:,0],allanDevDiff[:,1],label='diff')
    ax[1,2].set_title('Allan deviation, all qwp/pol2 combinations',fontweight = 'bold')
    ax[1,2].set_xlabel('averaging number')
    ax[1,2].set_ylabel('Allan deviation (degrees)')
    ax[1,2].set_xscale('log')
    ax[1,2].set_yscale('log')
    ax[1,2].grid(which='major',linewidth='0.5',color='gray')
    ax[1,2].grid(which='minor',linewidth='0.5',color='lightgray')
    ax[1,2].legend()

    ''' calculate average std deviation of each group of four 00,01,10,11 combinations '''
    allDSFAvgSum = np.vstack((allDSF00Avg,allDSF01Avg,allDSF10Avg,allDSF11Avg))
    print(np.shape(allDSFAvgSum))
    print('dsf std')
    print(np.mean(np.std(allDSFAvgSum,axis=0,ddof=1)))

    allFPAvgSum = np.vstack((allFP00Avg,allFP01Avg,allFP10Avg,allFP11Avg))
    print(np.shape(allFPAvgSum))
    print('boulder std')
    print(np.mean(np.std(allFPAvgSum,axis=0,ddof=1)))

    ax[2,0].plot(allDSF00Avg,label='0,0')
    ax[2,0].plot(allDSF01Avg,label='0,180')
    ax[2,0].plot(allDSF10Avg,label='180,0')
    ax[2,0].plot(allDSF11Avg,label='180,180')

    ax[2,1].plot(allFP00Avg,label='0,0')
    ax[2,1].plot(allFP01Avg,label='0,180')
    ax[2,1].plot(allFP10Avg,label='180,0')
    ax[2,1].plot(allFP11Avg,label='180,180')

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    ax[0,2].legend()
    ax[2,0].legend()
    ax[2,1].legend()
    ax[2,2].legend()

    ax[0,0].set_title('qwp: 0, pol2: 0',fontweight = 'bold')
    ax[0,1].set_title('qwp: 0, pol2: +180',fontweight = 'bold')
    ax[1,0].set_title('qwp: +180, pol2: 0',fontweight = 'bold')
    ax[1,1].set_title('qwp: +180, pol2: +180',fontweight = 'bold')
    ax[0,2].set_title('all qwp/pol2 combinations',fontweight = 'bold')
    ax[2,0].set_title('dSF time dependence',fontweight = 'bold')
    ax[2,1].set_title('Boulder time dependence',fontweight = 'bold')
    ax[2,2].set_title('all qwp/pol2 combinations time dependence',fontweight = 'bold')

    ax[0,0].set_xlabel('birefringence (degrees)')
    ax[0,1].set_xlabel('birefringence (degrees)')
    ax[1,0].set_xlabel('birefringence (degrees)')
    ax[1,1].set_xlabel('birefringence (degrees)')
    ax[0,2].set_xlabel('birefringence (degrees)')
    ax[2,0].set_xlabel('measurement number')
    ax[2,1].set_xlabel('measurement number')
    ax[2,2].set_xlabel('measurement number')

    ax[2,0].set_ylabel('birefringence (degrees)')
    ax[2,1].set_ylabel('birefringence (degrees)')
    ax[2,2].set_ylabel('birefringence (degrees)')

    f.suptitle(path+nameDSFRoot+'\n'+path+nameBoulderRoot+'\n',fontweight = 'bold',fontsize=14)

    plt.tight_layout()
    plt.show()

    return resBoulderSummary, resDSFSummary, allResultsFP, allResultsDSF

def analyseZerosLoop(fname, norm=False, analyseNorm=False):

    allData = np.genfromtxt(fname,delimiter=',')

    if norm == True:
        dataShape = np.shape(allData)
        nMeas = int((dataShape[1]-1)/7)

        reshapedData = np.zeros((dataShape[0],int(1+2*nMeas)))
        reshapedData[:,0] = allData[:,0]

        anglesColumnIdx = np.arange(1,int(1+7*nMeas),7)
        reshapedAnglesColumnIdx = np.arange(1,int(1+2*nMeas),2)

        reshapedData[:,reshapedAnglesColumnIdx] = allData[:,anglesColumnIdx]

        powersColumnIdx = np.arange(2,int(1+7*nMeas),7)
        powersNormColumnIdx = np.arange(5,int(1+7*nMeas),7)
        reshapedPowersColumnIdx = np.arange(2,int(1+2*nMeas),2)

        if analyseNorm == True:
            reshapedData[:,reshapedPowersColumnIdx] = allData[:,powersColumnIdx]/allData[:,powersNormColumnIdx]*np.mean(allData[:,powersNormColumnIdx])

        elif analyseNorm == False:
            reshapedData[:,reshapedPowersColumnIdx] = allData[:,powersColumnIdx]
        
        allData = reshapedData


    dataShape = np.shape(allData)
    lDataset = dataShape[0]
    print(lDataset)
    nDatasets = (dataShape[1]-1)/2
    print(dataShape)


    print(f'{nDatasets} total datasets')

    z1Guess = allData[int(lDataset/4),0]
    z2Guess = allData[int(3*lDataset/4),0]

    Cguess = np.min(allData[:,2])
    Aguess = (allData[0,2]-allData[int(lDataset/4),2])/((allData[0,0]-allData[int(lDataset/4),0])**2)

    print(z1Guess,z2Guess,Cguess,Aguess)

    results = np.zeros((int(nDatasets),6))

    for n in np.arange(nDatasets):
        data1 = allData[0:int(lDataset/2),(2*int(n)+1):(2*int(n)+3)]
        data2 = allData[int(lDataset/2):,(2*int(n)+1):(2*int(n)+3)]

        params1, cov1 = curve_fit(parabola,data1[:,0],data1[:,1],p0=[Aguess,z1Guess,Cguess],sigma=(0.05*data1[:,1]))
        params2, cov2 = curve_fit(parabola,data2[:,0],data2[:,1],p0=[Aguess,z2Guess,Cguess],sigma=(0.05*data2[:,1]))

        results[int(n),:] = np.array([params1[1],np.sqrt(cov1[1,1]),params2[1],np.sqrt(cov2[1,1]),np.abs(params2[1]-params1[1]),np.sqrt(cov1[1,1]+cov2[1,1])])

        if n==0:
            data1First = data1
            data2First = data2
            params1First = params1
            cov1First = cov1
            params2First = params2
            cov2First = cov2

            print(f'first fit: z1: {params1[1]} +/- {np.sqrt(cov1[1,1])}')
        
        if n==nDatasets-1:
            data1Last = data1
            data2Last = data2
            params1Last = params1
            cov1Last = cov1
            params2Last = params2
            cov2Last = cov2
            print(f'last fit: z1: {params1[1]} +/- {np.sqrt(cov1[1,1])}')


    print(f'zero1: {np.mean(results[:,0]):.7} +/- {np.std(results[:,0]):.5}')
    print(f'zero2: {np.mean(results[:,2]):.7} +/- {np.std(results[:,2]):.5}')
    print(f'zero2-zero1: {np.mean(results[:,4]):.7} +/- {np.std(results[:,4]):.5}')

    print(f'zero2-zero1 co-std dev: {np.sqrt(np.abs(np.cov(np.transpose(results[:,[0,2]]))))}')
    print(f'zero2-zero1 correlation coeff: {np.corrcoef(np.transpose(results[:,[0,2]]))[0,1]:.5}')

    if nDatasets > 1:
        # fit zero positions to linear line
        linParams1, cov1 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,0])
        linParams2, cov2 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,2])
        linParams3, cov3 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,4])

        print(f'zero1 slope: {linParams1[0]:.5}')
        print(f'zero2 slope: {linParams2[0]:.5}')
        print(f'mean slope: {(linParams1[0]+linParams2[0])/2:.5}')
        print(f'correction factor based on slope: {360/(360+(linParams1[0]+linParams2[0])/2)}')
        print(f'correction factor based on slope: {(360+(linParams1[0]+linParams2[0])/2)/360}')

        print(f'correction factor based on period: {180/(np.mean(results[:,4]))}')
        print(f'correction factor based on period: {(np.mean(results[:,4]))/180}')




    nbins=np.max([10,int(np.sqrt(nDatasets))])
    f,ax = plt.subplots(3,3,figsize =(16,10))
    ax[0,0].hist(results[:,0],bins=nbins,label=f'zero1: {np.mean(results[:,0]):.7} +/- {np.std(results[:,0]):.3}')
    ax[0,1].hist(results[:,2],bins=nbins,label=f'zero2: {np.mean(results[:,2]):.7} +/- {np.std(results[:,2]):.3}')
    ax[0,2].hist(results[:,4],bins=nbins,label=f'zero2-zero1: {np.mean(results[:,4]):.7} +/- {np.std(results[:,4]):.3}')
    ax[1,0].plot(results[:,0],results[:,2],'.',label=f'zero2-zero1 correlation coeff: {np.corrcoef(np.transpose(results[:,[0,2]]))[0,1]:.5}')
    ax[1,1].plot(results[:,0]-np.mean(results[:,0]),'.',label=f'zero1, mean fit error:{np.mean(results[:,1]):.3}')
    ax[1,1].plot(results[:,2]-np.mean(results[:,2]),'.',label=f'zero2, mean fit error:{np.mean(results[:,3]):.3}')
    if nDatasets > 1:
        ax[1,1].plot(linParams1[0]*(np.arange(nDatasets)-(nDatasets-1)/2),label=f'fit 1: slope = {linParams1[0]:.3} +/- {np.sqrt(cov1[0,0]):.3}')
        ax[1,1].plot(linParams2[0]*(np.arange(nDatasets)-(nDatasets-1)/2),label=f'fit 2: slope = {linParams2[0]:.3} +/- {np.sqrt(cov2[0,0]):.3}')
    ax[1,2].plot(results[:,4],'.',label=f'data, mean fit error:{np.mean(results[:,5]):.3}')
    if nDatasets > 1:
        ax[1,2].plot(linParams3[0]*(np.arange(nDatasets))+linParams3[1],label=f'fit: slope = {linParams3[0]:.3} +/- {np.sqrt(cov2[0,0]):.3}')

    ax[0,0].legend()
    ax[0,1].legend()
    ax[0,2].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    ax[1,2].legend()

    ax[0,0].set_title('zero 1 position distribution', fontweight='bold')
    ax[0,1].set_title('zero 2 position distribution', fontweight='bold')
    ax[0,2].set_title('zero1 - zero2 distance distribution', fontweight='bold')

    ax[1,0].set_title('zero 1 <--> zero 2 correlation', fontweight='bold')
    ax[1,1].set_title('zero positions \'time\' dependence', fontweight='bold')
    ax[1,2].set_title('zero1 - zero2 distance \'time\' dependence', fontweight='bold')

    ax[2,0].set_title('zero 1 fit performance', fontweight='bold')
    ax[2,1].set_title('zero 2 fit performance', fontweight='bold')

    ax[0,0].set_xlabel('zero 1 position (degrees)')
    ax[0,1].set_xlabel('zero 2 position (degrees)')
    ax[0,2].set_xlabel('zero 2 - zero 1 position (degrees)')
    ax[1,0].set_xlabel('zero 1 position (degrees)')
    ax[1,0].set_ylabel('zero 2 position (degrees)')
    ax[1,1].set_xlabel('dataset number')
    ax[1,1].set_ylabel('offset zero position (degrees)')
    ax[1,2].set_xlabel('dataset number')
    ax[1,2].set_ylabel('zero 2 - zero 1 position (degrees)')

    data1Dense = np.arange(data1First[0,0],data1First[-1,0],0.001)
    data2Dense = np.arange(data2First[0,0],data2First[-1,0],0.001)

    ax[2,0].plot(data1First[:,0],data1First[:,1],'.',label='first data set')
    ax[2,0].plot(data1Last[:,0],data1Last[:,1],'.',label='last data set')
    ax[2,0].plot(data1Dense,parabola(data1Dense,*params1First),label='first fit')
    ax[2,0].plot(data1Dense,parabola(data1Dense,*params1Last),label='last fit')
    ax[2,0].plot(data1First[:-1,0]+np.diff(data1First[:,0]),np.diff(data1First[:,1]),'.')
    ax[2,0].grid(which='major',linewidth='0.5',color='gray')

    ax[2,1].plot(data2First[:,0],data2First[:,1],'.',label='first data set')
    ax[2,1].plot(data2Last[:,0],data2Last[:,1],'.',label='last data set')
    ax[2,1].plot(data2Dense,parabola(data2Dense,*params2First),label='first fit')
    ax[2,1].plot(data2Dense,parabola(data2Dense,*params2Last),label='last fit')

    ax[2,0].legend()
    ax[2,1].legend()
    ax[2,0].set_xlabel('bellMotors\' stage position (degrees)')
    ax[2,1].set_xlabel('bellMotors\' stage position (degrees)')
    ax[2,0].set_ylabel('power (W)')
    ax[2,1].set_ylabel('power (W))')

    print(nDatasets)

    if nDatasets >= 6:
        # calculate and plot Allan deviations of zero positions and zero-position separation
        allanDevZ1 = calcAllanDev(results[:,0])
        allanDevZ2 = calcAllanDev(results[:,2])
        allanDevZsep = calcAllanDev(results[:,4])

        ax[2,2].plot(allanDevZ1[:,0],allanDevZ1[:,1],label='zero 1')
        ax[2,2].plot(allanDevZ2[:,0],allanDevZ2[:,1],label='zero 2')
        ax[2,2].plot(allanDevZsep[:,0],allanDevZsep[:,1],label='zeros\' separation')
        ax[2,2].set_xscale('log')
        ax[2,2].set_yscale('log')
        ax[2,2].grid(which='major',linewidth='0.5',color='gray')
        ax[2,2].grid(which='minor',linewidth='0.5',color='lightgray')
        ax[2,2].set_xlabel('number of measurements')
        ax[2,2].set_ylabel('position uncertainty (degrees)')
        ax[2,2].legend()
        ax[2,2].set_title('zero position Allan deviations', fontweight='bold')
    
    f.suptitle(fname+'\n',fontweight = 'bold',fontsize=14)

    if norm == True:
        if analyseNorm == False:
            f.suptitle(fname+'    raw data\n',fontweight = 'bold',fontsize=14)
        elif analyseNorm == True:
            f.suptitle(fname+'    normalized data\n',fontweight = 'bold',fontsize=14)


    plt.tight_layout()

    plt.show()

    return allData, results


def stages_prel_steps():
    # Home all stages
    for stage in stages.values():
        stage.mHome()

    print("Stages homed succesfully.")

    # Stage the rotators to desired initial angles
    for info in stages_info:
        stages[info['name']].mAbs(info['initial_angle'])

    print("Stages set to initial angles.")

def measure_round(degrees, powers_H, powers_V, ax):

    stages['qwp_1_stage'].mAbs(QWP_1_deg + 45)

    for degree in degrees:
        
        print(f"Measuring degree {degree}; last polarizer set at {pol_2_deg_H + degree}.")

        stages['pol_1_stage'].mAbs(pol_1_deg - 45 + degree)
        stages['sample_stage'].mAbs(sample_deg + degree)
        stages['hwp_1_stage'].mAbs(HWP_1_deg + 45 + degree)
        stages['pol_2_stage'].mAbs(pol_2_deg_H + degree)
    
        time.sleep(0.5)

        print(f"Measuring degree {degree}; last polarizer set at {pol_2_deg_V + degree}.")

        power = capture_power(pmeter)
        powers_H.append(power)

        stages['pol_2_stage'].mAbs(pol_2_deg_V + degree)

        time.sleep(0.5)

        power = capture_power(pmeter)
        powers_V.append(power)

        update_plot(degrees[:len(powers_H)], powers_H, degrees[:len(powers_V)], powers_V, ax)

    save_to_file(data_H, powers_H)
    save_to_file(data_V, powers_V)
    print(f"Files saved to {data_H} and {data_V}.")

def update_plot(x1, y1, x2, y2, ax):
    ax.cla()  # Clear previous plot
    ax.plot(x1, y1, label='Measured Power H')  # Plot the first set of data
    ax.plot(x2, y2, label='Measured Power V')  # Plot the second set of data
    ax.set_xlabel('Degree')
    ax.set_ylabel('Power (W)')
    ax.legend()
    plt.pause(0.1)  # Pause to show the update

def capture_power(pmeter):
    power = pmeter.query("MEAS:POW?")
    return float(power)

def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def fit_and_plot(degrees, powers_H, powers_V, ax):
    # Fit a sine function to the data
    if len(powers_H) > 3:
        try:
            params_H, _ = curve_fit(sine_function, degrees, powers_H, p0=[1, 2 * np.pi / 360, 0, np.mean(powers_H)])
            fitted_powers_H = sine_function(degrees, *params_H)
        except RuntimeError as e:
            print(f"Error fitting V data: {e}")
            fitted_powers_H = np.zeros_like(powers_H)
    else:
        fitted_powers_H = np.zeros_like(powers_H)
    
    if len(powers_V) > 3:
        try:
            params_V, _ = curve_fit(sine_function, degrees, powers_V, p0=[1, 2 * np.pi / 360, 0, np.mean(powers_V)])
            fitted_powers_V = sine_function(degrees, *params_V)
        except RuntimeError as e:
            print(f"Error fitting H data: {e}")
            fitted_powers_V = np.zeros_like(powers_V)
    else:
        fitted_powers_V = np.zeros_like(powers_V)

    # Plot the fitted function
    ax.cla()
    ax.plot(degrees, powers_H, label='Measured Power H')
    ax.plot(degrees, fitted_powers_H, 'b--', label='Fitted Power H')
    ax.plot(degrees, powers_V, label='Measured Power V')
    ax.plot(degrees, fitted_powers_V, 'r--', label='Fitted Power V')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Power (W)')
    ax.legend()
    plt.pause(0.1)

    save_to_file(data_H_fit, fitted_powers_H)
    save_to_file(data_V_fit, fitted_powers_V)

    # Save final plot and data
    plt.savefig(graph_path)
    print(f"Plot saved to {graph_path}")

def save_to_file(file_name, data_list):
    with open(file_name, 'w') as f:
        for item in data_list:
            f.write(f"{item},")
    print(f"Data saved to {file_name}")

def calc_error_est(powers_H, powers_V):
    errors = []
    for pV_datapoint, pH_datapoint in zip(powers_H, powers_V):
        tot_power = pV_datapoint + pH_datapoint
        error_est = (-1 / np.pi) * ((pH_datapoint / tot_power) - 0.5)
        errors.append(error_est)
    return errors

def calc_error_ideal(powers_H):
    errors_ideal = []
    for pH_datapoint in powers_H:
        error_ideal = (-1 / np.pi) * (pH_datapoint - 0.5)
        errors_ideal.append(error_ideal)
    return errors_ideal

def plot_error(degrees, err):
    fig, ax = plt.subplots()
    ax.plot(degrees, err, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Error Estimation')
    ax.set_title('Error Estimation vs. Degree')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save final plot and data
    plt.savefig(error_graph_path)
    print(f"Plot saved to {error_graph_path}")

# Function to read QWP2_CalChart.txt and return degrees and percentage differences
def read_calibration_chart(filename):
    degrees = []
    percentage_differences = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace
            if line and not line.startswith('Degree'):  # Skip empty lines and header
                parts = line.split()
                if len(parts) >= 2:  # Ensure there are at least two parts
                    try:
                        degree = float(parts[0])
                        percentage_diff = float(parts[1])
                        degrees.append(degree)
                        percentage_differences.append(percentage_diff)
                    except ValueError:
                        print(f"Warning: Could not convert line to float: {line}")

    return np.array(degrees), np.array(percentage_differences)
import numpy as np

def correct_QWP(powers_H, powers_V, degrees, cal_degrees, percentage_differences):
    corrected_powers_H = []
    corrected_powers_V = []
    
    for i, degree in enumerate(degrees):
        # Find the percentage difference for the current degree
        percentage_diff_H = percentage_differences[np.where(cal_degrees == degree)[0][0]]
        # Calculate corrected power for H
        corrected_power_H = powers_H[i] + (powers_H[i] * percentage_diff_H / 100)
        corrected_powers_H.append(corrected_power_H)
        
        # Adjust degree for V
        degree_V = degree + 90
        if degree_V >= 360:
            degree_V -= 360
        # Find the percentage difference for the adjusted degree
        percentage_diff_V = percentage_differences[np.where(cal_degrees == degree_V)[0][0]]
        # Calculate corrected power for V
        corrected_power_V = powers_V[i] + (powers_V[i] * percentage_diff_V / 100)
        corrected_powers_V.append(corrected_power_V)
    
    # Return corrected powers for both H and V
    return np.array(corrected_powers_H), np.array(corrected_powers_V)

def plot_corrected_power(corrected_powers_H, corrected_powers_V):
    # Plot the corrected power values
    fig, ax = plt.subplots()
    ax.plot(degrees, corrected_powers_H, label='Corrected Power H')
    ax.plot(degrees, corrected_powers_V, label='Corrected Power V')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Corrected Power (W)')
    ax.legend()
    plt.show()

    # Save final plot and data
    plt.savefig(corrected_graph_path)
    print(f"Plot saved to {corrected_graph_path}")

def parabola(x, a, x0, c):
    return a*(x-x0)**2 + c

def negCosine(x, a, x0, c, w):
    #return -a*np.cos(w*(x-x0)) + c
    return -a*np.cos(np.pi/90*(x-x0)) + c

def fitParabola(angles,powers,p0guess):
    params, covariance = curve_fit(parabola, angles, powers, p0=p0guess)
    return params, covariance

def fitRotationData(fname):
    # fits specific type of dataset stored in csv file
    # with first column representing angle
    # second column representing power
    # fifth column another power measurement for normalization
    data = np.genfromtxt(fname,delimiter=',')
    
    angles = data[:,0]
    powRaw = data[:,1]
    powNorm = data[:,1]/data[:,4]*np.mean(data[:,4])

    #angles = np.arange(360)
    #powRaw = data[:,0]
    #powNorm = data[:,0]/data[:,4]*np.mean(data[:,4])

    denseAngles = np.arange(0,360,0.01)
    
    ''' estimate zero positions '''
    # fit raw data
    #paramsRaw, rawCov = curve_fit(sine_function,angles,powRaw,p0=[np.mean(powRaw)/2,2*np.pi/180,0,np.mean(powRaw)],sigma=np.sqrt(np.min(powRaw))*np.sqrt(powRaw/np.min(powRaw)))
    paramsRaw, rawCov = curve_fit(sine_function,angles,powRaw,p0=[np.mean(powRaw)/2,2*np.pi/180,0,np.mean(powRaw)],sigma=powNorm)
    fitted_powRaw = sine_function(denseAngles,*paramsRaw)
    rawZero1 = (-(paramsRaw[2]/paramsRaw[1]) - 2*np.pi/paramsRaw[1]/4)%360
    rawZero2 = (rawZero1 + 2*np.pi/paramsRaw[1])%360

    print(rawZero1)
    print(rawZero2)

    ''' estimate zero positions '''    
    # fit normalized data
    #paramsNorm, normCov = curve_fit(sine_function,angles,powNorm,p0=[np.mean(powNorm)/2,2*np.pi/180,0,np.mean(powNorm)])
    paramsNorm, normCov = curve_fit(sine_function,angles,powNorm,p0=[np.mean(powNorm)/2,2*np.pi/180,0,np.mean(powNorm)],sigma=powNorm)
    fitted_powNorm = sine_function(denseAngles,*paramsNorm)
    
    normZero1 = (-(paramsNorm[2]/paramsNorm[1]) - 2*np.pi/paramsNorm[1]/4)%360
    normZero2 = (normZero1 + 2*np.pi/paramsNorm[1])%360

    print(normZero1)
    print(normZero2)

    rawPeriod = 2*np.pi/paramsRaw[1]
    rawPeriodError = 2*np.pi*np.sqrt(np.diag(rawCov))[1]/(paramsRaw[1])
    print(f'1st raw period: {rawPeriod:.7} +/- {rawPeriodError:.5}')
    normPeriod = 2*np.pi/paramsNorm[1]
    normPeriodError = 2*np.pi*np.sqrt(np.diag(normCov))[1]/(paramsNorm[1])
    print(f'1st norm period: {normPeriod:.7} +/- {normPeriodError:.5}')

    ''' fit to parabola for hopefully more accurate estimate of zero positions '''
    as1idx = (angles>rawZero1-2) & (angles<rawZero1+2)
    pRaw1, cRaw1 = fitParabola(angles[as1idx],powRaw[as1idx],[1,rawZero1,np.min(powRaw)])
    as2idx = (angles>rawZero2-2) & (angles<rawZero2+2)
    pRaw2, cRaw2 = fitParabola(angles[as2idx],powRaw[as2idx],[1,rawZero2,np.min(powRaw)])

    estRawZeroErr = np.sqrt(cRaw1[1,1]+cRaw2[1,1])
    print(f'parab raw period: {np.abs(pRaw1[1]-pRaw2[1]):.7} +/- {estRawZeroErr:.10}')

    ''' fit to parabola for hopefully more accurate estimate of zero positions '''
    as1idx = (angles>normZero1-1) & (angles<normZero1+1)
    pNorm1, cNorm1 = fitParabola(angles[as1idx],powNorm[as1idx],[1,rawZero1,np.min(powNorm)])
    as2idx = (angles>rawZero2-1) & (angles<rawZero2+1)
    pNorm2, cNorm2 = fitParabola(angles[as2idx],powNorm[as2idx],[1,rawZero2,np.min(powNorm)])

    estNormZeroErr = np.sqrt(cNorm1[1,1]+cNorm2[1,1])
    print(f'parab norm period: {np.abs(pNorm1[1]-pNorm2[1]):.7} +/- {estNormZeroErr:.10}')

    print(pNorm1[1])
    print(pNorm2[1])
    print(np.abs(pNorm1[1]-pNorm2[1]))


    plt.figure(1)
    plt.plot(angles,powRaw,'.',label='raw data')
    plt.plot(angles,powNorm,'.',label='normalized data')
    plt.plot(denseAngles,fitted_powRaw,label='fit to raw data')
    plt.plot(denseAngles,fitted_powNorm,label='fit to normalized data')

    plt.plot(denseAngles,parabola(denseAngles,*pRaw1),label='raw parabola')
    plt.plot(denseAngles,parabola(denseAngles,*pRaw2),label='raw parabola')
    plt.plot(denseAngles,parabola(denseAngles,*pNorm1),label='norm parabola')
    plt.plot(denseAngles,parabola(denseAngles,*pNorm2),label='norm parabola')

    plt.ylim([-0.05*np.max(powRaw),np.max(powRaw)*1.05])
    plt.ylabel('power (W)')
    plt.xlabel('stage angle (degrees)')

    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(angles,powRaw-sine_function(angles,*paramsRaw),'.',label='raw')
    plt.plot(angles,powNorm-sine_function(angles,*paramsNorm),'.',label='normalized')
    plt.plot(angles,np.zeros_like(angles))
    plt.legend()
    plt.show()

    return(data,paramsRaw,paramsNorm,rawCov,normCov)

def disconnect_stages(stages):
    for stage in stages.values():
        stage.stop()
        stage.mClose()


def main():

    pmeter = setup_powermeter(pmAddr,1000)
    print(pmeter)
    stages = initRotationStages(rm,[stages_info[1]],[rot_stage_addresses[1]])
    print(stages)
    stages['pol_1_stage'].mHome()
    #print('pol 1 stage homed, starting measZerosScan')
    #time.sleep(1)
    #d=measZerosLoop('../data/2024_09_23/powerCycles_'+str(int(time.time()))+'.csv',stages['pol_1_stage'],116.02,296.02,5,0.1,pmeter,1000,1)
    #stages['pol_1_stage'].mAbs(5)
    #stages['pol_1_stage'].close()
    #pmeter.close()

    # stages_prel_steps()

    # plt.ion()
    # fig, ax = plt.subplots()

    # measure_round(degrees, powers_H, powers_V, ax)

    # # Fit and plot the sine function at the end
    # fit_and_plot(degrees, powers_H, powers_V, ax)

    # # Calculate total HWP phase errors (real and ideal)
    # errors = calc_error_est(powers_H, powers_V)

    # #errors_ideal = calc_error_ideal(powers_H)
    # plot_error(degrees, errors)

    # # Read the calibration chartlsblk
    # cal_degrees, percentage_differences = read_calibration_chart(cal_chart_path)

    # # Correct the power values
    # corrected_powers_H, corrected_powers_V = correct_QWP(powers_H, powers_V, degrees, cal_degrees, percentage_differences)

    # plot_corrected_power(corrected_powers_H, corrected_powers_V)
    # save_to_file(data_H_corr, corrected_powers_H)
    # save_to_file(data_V_corr, corrected_powers_V)

    # disconnect_stages()
    # plt.ioff() 
    # plt.show()
    # rm.close()

    # rot_stages

#if __name__ == "__main__":
#    main()
