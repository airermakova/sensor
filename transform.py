import getopt
import math
import os
from random import randint
import sys
import threading
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTMCell, Masking
from keras.layers.rnn import ConvLSTM1D
import numpy
from matplotlib import pyplot
import time
from keras.layers import LSTM
import json
from sklearn.model_selection import train_test_split
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal.windows import blackman
import psutil
import scipy

## ARRAYS TO GET FOURIER TRANSFORM
fs = 1100
dft = []
signal = []
t_f = []
koef = []
t = []
cnt = 0
# VALUES DEFINED
tc = 1 / 1e6
tau = 70e-6
ndft = 4000
f00 = 50e3
Vmax = 0.5
freq = [0] * 20
freq2 = [0] * 20
meanfreq = [0] * 20
meanfreq2 = [0] * 20
meanmeanfreq = [0] * 20
stdmeanfreq = [0] * 20
meanmeanfreq2 = [0] * 20
stdmeanfreq2 = [0] * 20
meanf = [0] * 20
stdfreq = [0] * 20
meanf2 = [0] * 20
stdfreq2 = [0] * 20
f0v = [0] * 20
vv = []
Npower = []

n = numpy.minimum(math.floor(5 * tau / tc), ndft)
t = numpy.multiply(numpy.linspace(0, n - 1, n), tc)
noise = [1e-3, 5e-3, 2e-2, 1e-1]
nn = ndft - n


finvv = []
finmeanfr = []
finmeanfr2 = []


###SERVICE FUNCTIONS AREA###


def generate_v():
    global f0
    global fas
    retval = []
    for f in f0:
        v = numpy.multiply(
            Vmax,
            numpy.exp(-t / tau),
            numpy.sin((2 * numpy.pi * f * t) + fas * (randint(0, 1) - 0.5 * 0)),
        )
        retval.append(v)
    return retval


def generate_npower():
    retval = []
    for i in range(0, len(noise)):
        retval.append(Vmax * noise[i])
    return retval


###SERVICE FUNCTIONS AREA###


# f0 frequency calculation. To speed up further calculation it will be as an array of 20 elements, each element is a f0 calculated for range from 1 to 20
f0 = numpy.fromfunction(
    lambda kk, j: f00 + (kk - 10) * 250 / 20 + 3.333, (20, 1), dtype=float
)
print(f"f0: {f0}")

# zeta calculation. To speed up further calculation it will be as an array of 20 elements, each element is a zeta value calculated for each value of f0
zeta_koef = 1 / tau / 2 / numpy.pi
zeta = numpy.divide(zeta_koef, f0)
print(f"zeta: {zeta}")
f0v = f0

# phase error due to frequency peak slide. To speed up further calculation it will be as an array of 20 elements, each element is a value calculated for each value of f0
fas = numpy.multiply(tc * 2 * numpy.pi, f0)
print(f"fas: {zeta}")

# s power calculation
v = generate_v()
# print(v)
Spower = numpy.sum(numpy.power(v, 2)) / n
print(f"Spower: {Spower}")

# n power calculation. To speed up further calculation it will be as an array of 4 elements, each element is a value calculated for each value from noise array
Npower = generate_npower()
print(f"Npower: {Npower}")

# rn calculation. To speed up further calculation it will be as an array of 20 elements, each element counted as a nn+i
rn = numpy.fromfunction(lambda i, j: nn + i, (20, 1), dtype=float)

for kplot in range(0, len(noise) - 1):
    for j in range(0, 20 - 1):
        for i in range(0, 20 - 1):
            vv = v[i][0] + numpy.random.randn(1, n) * Vmax * noise[kplot]
            v1 = numpy.concatenate((vv[0], rn[i]), axis=0) + 1.5
            v2 = numpy.pad(vv[0], (0, nn)) + 1.5


# for kplot in range(0, 4 - 1):
#    Vmax = 0.5
#    for kk in range(0, 20 - 1):
#        f0 = f00 + (kk - 10) * 250 / 20 + 3.333
#        zeta = 1 / tau / 2 / numpy.pi / f0
#        f0v[kk] = f0
#        fas = (
#            tc * f0 * 2 * numpy.pi
#        )  # errore di fase dovuto alla determinazione del picco
#        v = numpy.multiply(
#            Vmax,
#            numpy.exp(-t / tau),
#            numpy.sin(2 * numpy.pi * f0 * t),  # + fas * (numpy.random(1) - 0.5) * 0
#        )
#        Spower = numpy.sum(numpy.power(v, 2)) / n
#        print(f"Spower - {Spower}")
#        for j in range(0, 20 - 1):
#            for i in range(0, 20 - 1):
#                vv = v + numpy.random.randn(1, n) * Vmax * noise[kplot]
#                Npower = Vmax * noise[kplot]
#                nn = ndft - n
#                rn = nn + i
#                v1 = numpy.pad(vv, (0, rn))
#                v2 = numpy.pad(vv, (0, nn))
#                v1 = v1 + 1.5
#                v2 = v2 + 1.5
#                V = fft(v1 - numpy.mean(v1))
#                V2 = fft(v2 - numpy.mean(v2))
#                vm = v1 - numpy.mean(v1)
#                vm2 = v2 - numpy.mean(v2)
#                NN = len(vm2)
#                for fk in range(0, (NN - 1)):
#                    print(fk)
#                    for ti in range(0, (NN - 1)):
#                        V[fk] = numpy.sum(
#                            vm[ti] * numpy.cos(ti * fk * 2 * numpy.pi / NN)
#                        ) + j * numpy.sum(
#                            vm[ti] * numpy.sin(ti * fk * 2 * numpy.pi / NN)
#                        )
#                        V2[fk] = numpy.sum(
#                            vm2[ti] * numpy.cos(ti * fk * 2 * numpy.pi / NN)
#                        ) + j * numpy.sum(
#                            vm2[ti] * numpy.sin(ti * fk * 2 * numpy.pi / NN)
#                        )
#                ini = 1
#                a = numpy.max(
#                    numpy.abs(numpy.power(V[ini : math.floor((n + rn) / 2)], 2))
#                )
#                b = numpy.max(
#                    numpy.abs(numpy.power(V[ini : math.floor((n + rn) / 2)], 2))
#                )
#                a2 = numpy.max(
#                    numpy.abs(numpy.power(V[ini : math.floor((n + nn / 2) / 2)], 2))
#                )
#                b2 = numpy.max(
#                    numpy.abs(numpy.power(V2[ini : math.floor((n + nn / 2) / 2)], 2))
#                )
#                f = numpy.multiply(numpy.linspace(0, n + rn, n + rn), 1 / tc / (n + rn))
#                f2 = numpy.multiply(
#                    numpy.linspace(0, n + nn, n + nn), 1 / tc / (n + nn)
#                )
#                freq[i] = (
#                    f.item(int(b + ini - 1))
#                    / numpy.sqrt(1 - 2 * numpy.power(zeta, 2))
#                    * numpy.sqrt(1 - numpy.power(zeta, 2))
#                    - f0
#                )
#                freq2[i] = (
#                    f2.item(int(b2 + ini - 1))
#                    / numpy.sqrt(1 - 2 * numpy.power(zeta, 2))
#                    * numpy.sqrt(1 - numpy.power(zeta, 2))
#                    - f0
#                )
#                # cos
#                freq[i] = (
#                    f.item(int(b + ini - 1)) * numpy.sqrt(1 - 0 * numpy.power(zeta, 2))
#                    - f0
#                )
#                freq2[i] = (
#                    f2.item(int(b2 + ini - 1))
#                    * numpy.sqrt(1 - 0 * numpy.power(zeta, 2))
#                    - f0
#                )
#                print(f"frequency - {freq2[i]}")
#            SNR = 10 * numpy.log10(Spower / Npower)
#            print(f"SNR - {SNR}")
#            meanfreq[j] = numpy.mean(freq)
#            meanfreq2[j] = numpy.mean(freq2)
#        meanmeanfreq[kk] = numpy.mean(meanfreq)
#        stdmeanfreq[kk] = numpy.std(meanfreq)
#        meanmeanfreq2[kk] = numpy.mean(meanfreq2)
#        stdmeanfreq2[kk] = numpy.std(meanfreq2)
#        meanf[kk] = numpy.mean(freq)
#        stdfreq[kk] = numpy.std(freq)
#        meanf2[kk] = numpy.mean(freq2)
#        stdfreq2[kk] = numpy.std(freq2)

#    finvv.append(f0v)
#    finmeanfr.append(meanmeanfreq)
#    finmeanfr2.append(meanmeanfreq2)
