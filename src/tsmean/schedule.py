import numpy as np
import matplotlib.pyplot as plt


def linear_schedule(size, eta_init=0.1, eta=0.01):
    return np.linspace(eta_init,eta,size)


def sine_schedule(size, cycles=5, eta_init = 1, exponent=1, warmup=True):
    if warmup:
        x = np.linspace(np.pi/2,(cycles+0.5)*np.pi,size)
        y = np.cos(x)/(x**exponent)
    else:
        x = np.linspace(np.pi/1.5,cycles*np.pi,size)
        y = np.sin(x)/(x**exponent)
    # normalize
    y = np.sign(y)*y
    y = eta_init*y/np.max(y)
    return y


def sine_schedule_by_width(size, cycle_width=500, eta_init = 1, exponent=1, warmup=False):

    if warmup:
        x = np.linspace(np.pi/2, (size/cycle_width+0.5)*np.pi, size+1)
        y = np.cos(x)/(x**exponent)
    else:
        x = np.linspace(np.pi/1.5,size/cycle_width*np.pi,size+1)
        y = np.sin(x)/(x**exponent)
    # normalize
    y = np.sign(y)*y
    y = eta_init*y/np.max(y)
    return y


def sawtooth_schedule(size, cycles=5, eta_init=1, exponent=1):
    y = []
    cycle_size = int(size//cycles)
    Y0 = np.linspace(1,0, cycles+1)
    for i in range(cycles):
        y0 = Y0[i]**exponent
        y1 = 0.1*y0
        yx = np.linspace(y0,y1,cycle_size)
        y.extend(yx)
    return np.array(y)*eta_init


def sawtooth_schedule_by_width(size, cycle_width=500, eta_init=1, exponent=1, cutoff=False):
    """if cutoff==True the last incomplete cycle is cut off. This ensures that the last 
    value of the schedule is small"""
    y = []
    cycles = size//cycle_width
    remaining_size = size - cycle_width*cycles

    if remaining_size > 0 and not cutoff:
        Y0 = np.linspace(1,0, cycles+2)
    else:
        Y0 = np.linspace(1,0, cycles+1)
    for i in range(cycles):
        y0 = Y0[i]**exponent
        y1 = 0.1*y0
        yx = np.linspace(y0,y1,cycle_width)
        y.extend(yx)
    
    if remaining_size > 0 and not cutoff:
        y0 = Y0[cycles]**exponent
        y1 = 0.1*y0/(remaining_size/cycle_width)
        yx = np.linspace(y0,y1,remaining_size)
        y.extend(yx)

    return np.array(y)*eta_init


