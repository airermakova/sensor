###############################################################################
# Company: University of Siena                                                #
# Engineer: Riccardo Moretti                                                  #
# Project: QCMOscillator                                                      #
# Description: Application main scheduler.                                    #
# Revision: v1.01 - MainSchedulerSignal, MainSchedulerException,              #
#                   MainScheduler classes added. run, connection,             #
#                   configutation, measurement methods in MainScheduler class #
#                   added.                                                    #
#           v0.01 - File created.                                             #
###############################################################################

from PySide6.QtCore import QObject,Signal,QRunnable,Slot
from c_fpga import FPGA,FPGAException
from serial.tools import list_ports
import csv
from datetime import datetime
import time
import os

class MainSchedulerSignal(QObject):
    '''
    Application main scheduler signals.
    '''
    connected = Signal() # Connected to FPGA architecture signal
    configured = Signal() # FPGA architecture configured signal
    connectionlost = Signal() # Connection to FPGA architecture lost signal
    newdata = Signal(float,float) # New data (frequency,resistance) available signal
    signallost = Signal()

class MainSchedulerException(Exception):
    def __init__(self,code:int):
        '''
        Initialize the structure of an exception generated by the application
        main scheduler.

        Args:
        - code: Event code
        '''
        match code:
            case 1:
                message = 'No serial ports available' # Event message
            case 2:
                message = 'FPGA architecture not found'
            case 3:
                message = 'Connection to FPGA architecture lost'
            case 4:
                message = 'Oscillator trap enabled'
            case _:
                message = 'Undefined error'
        super().__init__(message)
        self.code = code

class MainScheduler(QRunnable):
    def __init__(self) -> None:
        '''
        Initialize the application main scheduler.
        '''
        super().__init__()
        self.signal = MainSchedulerSignal()
        self.ddsfrequency = 4970000 # DDS frequency to set [Hz]
        self.fcaperture = 0.01 # Frequency counter aperture time to set [s]
        self.intermeasuretime = 1 # Time between measurements to set [s]
        self.CLOSE = False # Close the scheduler command
        self.STOP = False # Stop measurement command
        self.CONNECT = True # Connect to the FPGA architecture command
        self.START = False
        self.CONFIGURE = False # Configure the FPGA architecture command
        self.MEASURE = False # Measure data command
        self.SAVEDATA = False # Save data command
        self.LOST = False
        self.fpga = None # FPGA architecture
        self.savefile = None # Save file identifier
        self.savedir = None

    @Slot()
    def run(self) -> None:
        '''
        Run the scheduler.
        '''
        while True:
            if self.CLOSE:
                try:
                    self.fpga.close()
                except Exception as error:
                    print('CLOSE - ',error)
                break
            elif self.STOP:
                self.ddsfrequency = 4970000
                if self.SAVEDATA:
                    self.savefile.close()
                self.CONFIGURE = False
                self.MEASURE = False
                self.STOP = False
                self.LOST = False
            elif self.CONNECT:
                try:
                    self.connection()
                    self.CONNECT = False
                    self.signal.connected.emit()
                except Exception as error:
                    g=1
                    #print('CONNECT - ',error)
            elif self.CONFIGURE:
                try:
                    self.configuration()
                    self.CONFIGURE = False
                    self.MEASURE = True
                    self.signal.configured.emit()
                except Exception as error:
                    print('CONFIGURE - ',error)
                    self.CONNECT = True
                    self.signal.connectionlost.emit()
            elif self.MEASURE:
                try:
                    [qcmfrequency,qcmresistance] = self.measurement()
                    self.signal.newdata.emit(qcmfrequency,qcmresistance)
                except MainSchedulerException as error:
                    if error.code==4:
                        if not self.LOST:
                            self.LOST = True
                            self.signal.signallost.emit()
                        time.sleep(10)
                    else:
                        print('MEASURE - ',error)
                        self.CONNECT = True
                        self.CONFIGURE = True
                        self.signal.connectionlost.emit()
                except Exception as error:
                    print('MEASURE - ',error)
                    self.CONNECT = True
                    self.CONFIGURE = True
                    self.signal.connectionlost.emit()

    def connection(self) -> None:
        '''
        Connect to the FPGA architecture.
        '''
        devicefound = False # FPGA architecture found condition
        ports = list_ports.comports() # List of available serial ports
        if len(ports)==0:
            raise MainSchedulerException(1)
        else:
            for port in ports:
                try:
                    self.fpga = FPGA(port.device)
                    self.fpga.interrogation()
                    devicefound = True
                    break
                except FPGAException:
                    self.fpga.close()
                except:
                    pass
            if not devicefound:
                raise MainSchedulerException(2)

    def configuration(self) -> None:
        '''
        Configure the FPGA architecture.
        '''
        try:
            if self.START:
                self.fpga.configuration(ddsfrequency=self.ddsfrequency,fcaperture=self.fcaperture,intermeasuretime=self.fcaperture)
                [qcmfrequency,qcmresistance] = self.fpga.measurement()
                self.ddsfrequency = self.ddsfrequency-qcmfrequency+10000
                self.START = False
            self.fpga.configuration(ddsfrequency=self.ddsfrequency,fcaperture=self.fcaperture,intermeasuretime=self.intermeasuretime)
        except FPGAException:
            self.fpga.close()
            raise MainSchedulerException(3)

    def measurement(self) -> tuple[float,float]:
        '''
        Measure the oscillator frequency and resistance.

        Returns:
        - qcmfrequency : Measured QCM frequency [Hz]
        - qcmresistance : Measured QCM resistance [Ohm]
        '''
        try:
            [qcmfrequency,qcmresistance] = self.fpga.measurement()
            if self.SAVEDATA:
                csv.writer(self.savefile,delimiter=';').writerow([datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),self.ddsfrequency-qcmfrequency,qcmresistance])
            return qcmfrequency,qcmresistance
        except FPGAException as error:
            self.fpga.close()
            if error.code==6:
                raise MainSchedulerException(4)
            else:
                raise MainSchedulerException(3)
            
    def savefile(self):
        try:
            sz = os.path.getsize(self.savefile)/1048576
            if(sz>10):
                self.savefile.close()
                self.savefile = open(self.savedir +'\\'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv','w',newline='')
                csv.writer(self.savefile,delimiter=';').writerow(['Time','Frequency [Hz]','Resistance [Ohm]'])
            csv.writer(self.savefile,delimiter=';').writerow(['Time','Frequency [Hz]','Resistance [Ohm]'])
        except:
            print(f'file {self.savefile} does not exists')

