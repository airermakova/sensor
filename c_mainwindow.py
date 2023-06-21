###############################################################################
# Company: University of Siena                                                #
# Engineer: Riccardo Moretti                                                  #
# Project: QCMOscillator                                                      #
# Description: Application main window.                                       #
# Revision: v1.01 - FrequencyAxisItem, MainWindow classes added. tickStrings  #
#                   method of FrequencyAxisItem added. savecheckstatechanged, #
#                   changefolderpressed, timercheckstatechanged,              #
#                   timerchanged, startmeasurement, stopmeasurement,          #
#                   updatetime, fpgaconnected, fpgaconfigured,                #
#                   fpgaconnectionlost, fpganewdata, closeEvent methods of    #
#                   MainWindow class added.                                   #
#           v0.01 - File created.                                             #
###############################################################################

from PySide6.QtWidgets import QMainWindow,QCheckBox,QSizePolicy,QPushButton,\
                              QLabel,QHBoxLayout,QSpacerItem,QTimeEdit,\
                              QVBoxLayout,QWidget,QFileDialog, QListWidget, QFrame, QLineEdit
from PySide6.QtCore import QTime,QTimer,QThreadPool,Qt
from PySide6.QtGui import QCloseEvent
from pyqtgraph import AxisItem,PlotWidget,DateAxisItem,PlotItem
from c_mainscheduler import MainScheduler
import os
from datetime import datetime
import csv
import glob
import time
import argparse
import pandas as pd
from datetime import datetime
import pytz
from PySide6 import QtGui, QtCore


class FrequencyAxisItem(AxisItem):
    def __init__(self,*args,**kwargs):
        '''
        Define the plot frequency axis.
        '''
        super(FrequencyAxisItem,self).__init__(*args,**kwargs)

    def tickStrings(self, values, scale, spacing):
        '''
        Set the frequency axis strings.
        '''
        return [f'{v:.3f}' for v in values] 

class MainWindow(QMainWindow):
    filepath=""

    def __init__(self) -> None:
        '''
        Initialize the application main window.
        '''
        super().__init__()
        self.setWindowTitle('QCM Oscillator')

        #WIDGETS TO SAVE DATA IN CSV FILE
        self.savecheck = QCheckBox('Save data')
        self.savecheck.setStyleSheet('font-size:14px')
        self.savecheck.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.savecheck.stateChanged.connect(self.savecheckstatechanged)
        self.savecheck.setChecked(True)

        self.changefolderbutton = QPushButton('...')
        self.changefolderbutton.setFixedSize(30,30)
        self.changefolderbutton.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.changefolderbutton.setEnabled(True)
        self.changefolderbutton.clicked.connect(self.changefolderpressed)

        self.path = QLabel()
        self.path.setStyleSheet('font-size:12px; color: blue')
        self.path.setText("Folder to save data: ")
        self.savepath = QLabel()
        self.savepath.setStyleSheet('font-size:12px; color: blue')
        self.savepath.setText(os.path.join(os.getcwd(),"Data"))

        savelayout = QHBoxLayout()
        savelayout.addWidget(self.savecheck)
        savelayout.addWidget(self.path)
        savelayout.addWidget(self.savepath)
        savelayout.addWidget(self.changefolderbutton)
        savelayout.addItem(QSpacerItem(0,0,QSizePolicy.Expanding,QSizePolicy.Fixed))

        frame = QFrame()
        frame.setStyleSheet("background-color: lightblue;")
        frame.setFrameStyle(1)
        frame.setLayout(savelayout)

        #TIMER WIDGETS
        self.timercheck = QCheckBox('Set timer')
        self.timercheck.setStyleSheet('font-size:20px')
        self.timercheck.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.timercheck.stateChanged.connect(self.timercheckstatechanged)

        self.timerbox = QTimeEdit()
        self.timerbox.setDisplayFormat('hh:mm:ss')
        self.timerbox.setMinimumTime(QTime(0,0,1))
        self.timerbox.setFixedSize(130,50)
        self.timerbox.setStyleSheet('font-size:20px')
        self.timerbox.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.timerbox.setEnabled(False)
        self.timerbox.timeChanged.connect(self.timerchanged)

        self.remainingtime = QLabel()
        self.remainingtime.setStyleSheet('font-size:20px')

        timelayout = QHBoxLayout()
        timelayout.addWidget(self.timercheck)
        timelayout.addWidget(self.timerbox)
        timelayout.addWidget(self.remainingtime)
        timelayout.addItem(QSpacerItem(0,0,QSizePolicy.Expanding,QSizePolicy.Fixed))     
    
        #GRAPH SCALER WIDGET
        scaleflayout = QHBoxLayout()
        self.scaleaxisLabel = QLabel()
        self.scaleaxisLabel.setText("Scale Y Axis. From To")
        self.scaleaxisLabel.setStyleSheet("color:white")
        self.scalerxaxischeckbox = QCheckBox()
        self.scalerxaxischeckbox.setEnabled(False)
        self.scalerxaxischeckbox.clicked.connect(self.scalexfrequency)
        self.yresvaluefromlineedit = QLineEdit()
        self.yresvaluefromlineedit.textChanged.connect(self.setscaleractive)
        self.yresvaluefromlineedit.setStyleSheet("color:white")
        self.yresvaluetolineedit = QLineEdit()
        self.yresvaluetolineedit.textChanged.connect(self.setscaleractive)
        self.yresvaluetolineedit.setStyleSheet("color:white")
        scaleflayout.addWidget(self.scalerxaxischeckbox)
        scaleflayout.addWidget(self.scaleaxisLabel)
        scaleflayout.addWidget(self.yresvaluefromlineedit)
        scaleflayout.addWidget(self.yresvaluetolineedit)
        scaleflayout.addItem(QSpacerItem(0,0,QSizePolicy.Expanding,QSizePolicy.Fixed))  

        scalerlayout = QHBoxLayout()
        self.scalefaxis = QLabel()
        self.scalefaxis.setText("Scale Y Axis. From To")
        self.scalefaxis.setStyleSheet("color:white")
        self.scalefxaxis = QCheckBox()
        self.scalefxaxis.setEnabled(False)
        self.scalefxaxis.clicked.connect(self.scalexresistance)
        self.yfrvalue = QLineEdit()
        self.yfrvalue.setStyleSheet("color:white")
        self.yfrvalue.textChanged.connect(self.setscalefactive)
        self.yfrmaxvalue = QLineEdit()
        self.yfrmaxvalue.setStyleSheet("color:white")
        self.yfrmaxvalue.textChanged.connect(self.setscalefactive)
        scalerlayout.addWidget(self.scalefxaxis)
        scalerlayout.addWidget(self.scalefaxis)
        scalerlayout.addWidget(self.yfrvalue)
        scalerlayout.addWidget(self.yfrmaxvalue)
        scalerlayout.addItem(QSpacerItem(0,0,QSizePolicy.Expanding,QSizePolicy.Fixed)) 

        scalerfinallayout = QHBoxLayout()
        scalerfinallayout.addLayout(scaleflayout)
        scalerfinallayout.addLayout(scalerlayout)        
        scaleframe = QFrame()
        scaleframe.setStyleSheet("background-color: black;")
        scaleframe.setFrameStyle(1)
        scaleframe.setLayout(scalerfinallayout)

        #GRAPH PLOT WIDGET
        self.resistanceplot = PlotWidget(axisItems={'bottom':DateAxisItem()})
        self.resistanceplot.showGrid(x=True,y=True)
        self.resistanceplot.setTitle('Resistance [Ohm]')
        self.resistanceplot.autoPixelRange
        self.resistancex = []
        self.resistancey = []
        self.resistanceline = self.resistanceplot.plot(self.resistancex,self.resistancey)
        
        self.frequencyplot = PlotWidget(axisItems={'bottom':DateAxisItem(),'left': FrequencyAxisItem(orientation='left')})
        self.frequencyplot.showGrid(x=True,y=True)
        self.frequencyplot.setTitle('Frequency [Hz]')
        self.frequencyx = []
        self.frequencyy = []
        self.frequencyline = self.frequencyplot.plot(self.frequencyx,self.frequencyy)    

        plotlayout = QHBoxLayout()
        plotlayout.addWidget(self.frequencyplot)
        plotlayout.addWidget(self.resistanceplot)

        #BUTTONS START AND STOP
        self.startbutton = QPushButton('START')
        self.startbutton.setFixedSize(100,50)
        self.startbutton.setStyleSheet('font-size:22px;font-weight:semibold;color:darkgreen')
        self.startbutton.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.startbutton.setEnabled(False)
        self.startbutton.clicked.connect(self.startmeasurement)

        self.stopbutton = QPushButton('STOP')
        self.stopbutton.setFixedSize(100,50)
        self.stopbutton.setStyleSheet('font-size:22px;font-weight:semibold;color:darkred')
        self.stopbutton.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.stopbutton.setEnabled(False)
        self.stopbutton.clicked.connect(self.stopmeasurement)

        #TIMER AND START STOP BUTTON LAYOUT
        buttonlayout = QHBoxLayout()
        buttonlayout.addLayout(timelayout)
        buttonlayout.addWidget(self.startbutton)
        buttonlayout.addWidget(self.stopbutton)

        #DEVICE CONNECTED AND MEASUREMENT STARTED LABELS
        self.connectedlabel = QLabel('Device not connected')
        self.connectedlabel.setStyleSheet('font-size:20px;color:darkred;font-weight:bold')
        self.connectedlabel.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)

        self.measurelabel = QLabel('Measurement stopped')
        self.measurelabel.setStyleSheet('font-size:20px;color:darkred;font-weight:bold')
        self.measurelabel.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)

        labellayout = QVBoxLayout()
        labellayout.addWidget(self.connectedlabel)
        labellayout.addWidget(self.measurelabel)

        #FINAL LAYOUTS FOR MEASUREMENTS AND GRAPHS
        bottomlayout = QHBoxLayout()
        bottomlayout.addLayout(timelayout)
        bottomlayout.addLayout(buttonlayout)
        bottomlayout.addLayout(labellayout)
        bottomframe = QFrame()
        bottomframe.setStyleSheet("background-color: lightgrey;")
        bottomframe.setFrameStyle(1)
        bottomframe.setLayout(bottomlayout)

        layout = QVBoxLayout()
        layout.addWidget(frame)
        layout.addWidget(bottomframe)
        layout.addWidget(scaleframe)
        layout.addLayout(plotlayout)

        #WIDGETS WITH PREVIOUS RECORDINGS
        self.oldrecs = QLabel("Previous records")
        self.oldrecs.setStyleSheet('font-size:15px; background-color: lightgrey; border: 1px solid black;')
        self.prevlist = QListWidget(self)
        self.prevlist.setStyleSheet('font-size:15px; background-color: lightblue; border: 1px solid black;')
        if os.path.isdir(self.savepath.text()):            
            self.prevlist.addItems([os.path.basename(x) for x in glob.glob(f"{self.savepath.text()}/*.csv")])
        self.prevlist.itemDoubleClicked.connect(self.graphSelected)
        oldpicslayout = QVBoxLayout()
        oldpicslayout.addWidget(self.oldrecs)
        oldpicslayout.addWidget(self.prevlist)

        #FINAL LAYOUTS
        finlayout = QHBoxLayout()
        finlayout.addLayout(oldpicslayout)
        finlayout.addLayout(layout, 1)
        finlayout.addStretch(0)

        widget = QWidget()
        widget.setLayout(finlayout)
        self.setCentralWidget(widget)
        #widget.setStyleSheet('background-color: lightgrey;')

        self.show()

        self.timer = QTimer()
        self.timer.timeout.connect(self.updatetime)

        self.mainscheduler = MainScheduler()
        self.mainscheduler.signal.connected.connect(self.fpgaconnected)
        self.mainscheduler.signal.configured.connect(self.fpgaconfigured)
        self.mainscheduler.signal.connectionlost.connect(self.fpgaconnectionlost)
        self.mainscheduler.signal.newdata.connect(self.fpganewdata)
        self.mainscheduler.signal.signallost.connect(self.fpgasignallost)

        self.connectedstate = False
        self.idlestate = True

        self.threadpool = QThreadPool()
        self.threadpool.start(self.mainscheduler)

    def savecheckstatechanged(self,state:int) -> None:
        '''
        Enable or disable the save commands and indicators. If enabled, select
        the data destination folder.
        
        Args:
        - state: Button state
        '''
        if state==Qt.CheckState.Checked.value:
            self.changefolderbutton.setEnabled(True)
            self.savepath.setText(QFileDialog.getExistingDirectory(caption='Select the data destination folder'))
            if self.savepath.text()=='':
                self.savepath.setText(os.getcwd())
        else:
            self.changefolderbutton.setEnabled(False)
            self.savepath.setText('')

    def changefolderpressed(self) -> None:
        '''
        Update the data destination folder.
        '''
        oldpath = self.savepath.text()
        self.savepath.setText(QFileDialog.getExistingDirectory())
        if self.savepath.text()=='':
            self.savepath.setText(oldpath)
        elif os.path.isdir(self.savepath.text()): 
            self.prevlist.clear()           
            self.prevlist.addItems([os.path.basename(x) for x in glob.glob(f"{self.savepath.text()}/*.csv")])

    def timercheckstatechanged(self,state:int) -> None:
        '''
        Enable or disable the execution time commands and indicators.
        '''
        if state==Qt.CheckState.Checked.value:
            self.timerbox.setEnabled(True)
            self.remainingtime.setText(self.timerbox.time().toString())
        else:
            self.timerbox.setEnabled(False)
            self.remainingtime.setText('')

    def timerchanged(self) -> None:
        '''
        Update the execution time.
        '''
        self.remainingtime.setText(self.timerbox.time().toString())

    def startmeasurement(self) -> None:
        '''
        Start the measurement operations.
        '''
        self.idlestate = False
        if self.savecheck.isChecked():
            if not os.path.isdir(self.savepath.text()):
                os.mkdir(self.savepath.text())
            self.writeCsvFile(self.savepath.text())
        else:
            self.mainscheduler.SAVEDATA = False
        self.savecheck.setEnabled(False)
        self.changefolderbutton.setEnabled(False)
        self.timercheck.setEnabled(False)
        self.timerbox.setEnabled(False)
        self.startbutton.setEnabled(False)
        self.stopbutton.setEnabled(True)
        self.frequencyx = []
        self.frequencyy = []
        self.frequencyline.setData(self.frequencyx,self.frequencyy)
        self.resistancex = []
        self.resistancey = []
        self.resistanceline.setData(self.resistancex,self.resistancey)
        if self.timercheck.isChecked():
            self.timer.start(1000)
        self.mainscheduler.START = True
        self.mainscheduler.CONFIGURE = True

    def stopmeasurement(self) -> None:
        '''
        Stop the measurement operations.
        '''
        self.timer.stop()
        self.mainscheduler.STOP = True
        self.mainscheduler.CONNECT = not self.connectedstate
        self.idlestate = True
        self.savecheck.setEnabled(True)
        if self.savecheck.isChecked():
            self.changefolderbutton.setEnabled(True)
        self.timercheck.setEnabled(True)
        if self.timercheck.isChecked():
            self.timerbox.setEnabled(True)
            self.remainingtime.setText(self.timerbox.time().toString())
        self.startbutton.setEnabled(self.connectedstate)
        self.stopbutton.setEnabled(False)
        self.measurelabel.setText('Measurement stopped')
        self.measurelabel.setStyleSheet('font-size:24px;color:red;font-weight:bold')
        self.setFileList()

    def updatetime(self) -> None:
        '''
        Update the execution time countdown.
        '''
        remainingtime = QTime.fromString(self.remainingtime.text(),'hh:mm:ss').addSecs(-1)
        self.remainingtime.setText(remainingtime.toString())
        if remainingtime==QTime(0,0,0):
            self.stopmeasurement()

    def fpgaconnected(self) -> None:
        '''
        Enable the measurement operations.
        '''
        self.connectedstate = True
        self.startbutton.setEnabled(self.idlestate)
        self.connectedlabel.setText('Device connected')
        self.connectedlabel.setStyleSheet('font-size:24px;color:darkgreen;font-weight:bold')

    def fpgaconfigured(self) -> None:
        '''
        Update the measurement label.
        '''
        self.measurelabel.setText('Measurement running')
        self.measurelabel.setStyleSheet('font-size:24px;color:darkgreen;font-weight:bold')

    def fpgaconnectionlost(self) -> None:
        '''
        Disable the measurement start command. Update the connection label.
        '''
        self.connectedstate = False
        self.startbutton.setEnabled(False)
        self.connectedlabel.setText('Device not connected')
        self.connectedlabel.setStyleSheet('font-size:24px;color:red;font-weight:bold')

    def fpganewdata(self,qcmfrequency:float,qcmresistance:float) -> None:
        '''
        Update the plots.

        Args:
        - qcmfrequency : Measured QCM frequency [Hz]
        - qcmresistance : Measured QCM resistance [Ohm]
        '''
        if self.mainscheduler.LOST:
            self.measurelabel.setText('Measurement running')
            self.measurelabel.setStyleSheet('font-size:24px;color:green;font-weight:bold')
            self.mainscheduler.LOST = False
        x = time.time()
        self.frequencyx.append(x)
        self.frequencyy.append(qcmfrequency)
        self.frequencyx = self.frequencyx[max(len(self.frequencyx)-1000,0):1000]
        self.frequencyy = self.frequencyy[max(len(self.frequencyy)-1000,0):1000]
        self.frequencyline.setData(self.frequencyx,self.frequencyy)
        self.resistancex.append(x)
        self.resistancey.append(qcmresistance)
        self.resistancex = self.resistancex[max(len(self.resistancex)-1000,0):1000]
        self.resistancey = self.resistancey[max(len(self.resistancey)-1000,0):1000]
        self.resistanceline.setData(self.resistancex,self.resistancey)

    def fpgasignallost(self) -> None:
        self.measurelabel.setText('Recovering input signal...')
        self.measurelabel.setStyleSheet('font-size:24px;color:brown;font-weight:bold')

    def closeEvent(self,event:QCloseEvent) -> None:
        '''
        Close the application process.
        '''
        self.mainscheduler.CLOSE = True
        return super().closeEvent(event)
    
    def setFileList(self) -> None:
        self.prevlist.clear()
        if os.path.isdir(self.savepath.text()):            
            self.prevlist.addItems([os.path.basename(x) for x in glob.glob(f"{self.savepath.text()}/*.csv")])

    def scalexfrequency(self) -> None:
        if self.scalerxaxischeckbox.isChecked():
            valfrom = self.yresvaluefromlineedit.text()
            valto = self.yresvaluetolineedit.text()
            if(valto.isdigit() and valfrom.isdigit()):
                self.frequencyplot.setYRange(float(valto), float(valfrom))
        else:
            self.frequencyplot.autoRange()
            self.frequencyplot.enableAutoRange()

    def scalexresistance(self):
        if self.scalefxaxis.isChecked():
            valfrom = self.yfrvalue.text()
            valto = self.yfrmaxvalue.text()
            if(valto.isdigit() and valfrom.isdigit()):
                self.resistanceplot.autoPixelRange = False
                self.resistanceplot.setYRange(float(valto), float(valfrom))
        else:
            self.resistanceplot.autoRange()
            self.resistanceplot.enableAutoRange()


    def setscaleractive(self):
        if len(self.yresvaluefromlineedit.text())>0 and len(self.yresvaluetolineedit.text())>0:
            self.scalerxaxischeckbox.setEnabled(True)
        else:
            self.scalerxaxischeckbox.setChecked(False)
            self.scalerxaxischeckbox.setEnabled(False)

    def setscalefactive(self):
        if len(self.yfrvalue.text())>0 and len(self.yfrmaxvalue.text())>0:
            self.scalefxaxis.setEnabled(True)
        else:
            self.scalefxaxis.setChecked(False)
            self.scalefxaxis.setEnabled(False)
   
    def graphSelected(self):
        file = self.prevlist.currentItem().text()
        filepath=os.path.join(self.savepath.text(),file)
        self.frequencyx = []
        self.frequencyy = []
        self.resistancex = []
        self.resistancey = []
        self.resistanceline.setData(self.resistancex,self.resistancey)
        self.frequencyline.setData(self.frequencyx,self.frequencyy)
        with open(filepath) as csvfile:
            print(f"READING SCV FILE {filepath}")
            csvReader = csv.reader(csvfile, delimiter=';')
            for row in csvReader:
                try:
                    tm = self.convert_string_to_time(row[0])
                    self.frequencyx.append(tm)
                    self.frequencyy.append(float(row[1]))
                    self.frequencyx = self.frequencyx[max(len(self.frequencyx)-1000,0):1000]
                    self.frequencyy = self.frequencyy[max(len(self.frequencyy)-1000,0):1000]
                    self.resistancex.append(tm)
                    self.resistancey.append(float(row[2]))
                    self.resistancex = self.resistancex[max(len(self.resistancex)-1000,0):1000]
                    self.resistancey = self.resistancey[max(len(self.resistancey)-1000,0):1000]
                except:
                    continue
            self.resistanceline.setData(self.resistancex,self.resistancey)
            self.frequencyline.setData(self.frequencyx,self.frequencyy)

    def convert_string_to_time(self, date_string):
        date_obj = datetime.strptime(date_string, '%d/%m/%Y, %H:%M:%S')
        return date_obj.timestamp()
    
    def writeCsvFile(self, dirpath):
        self.mainscheduler.savedir = dirpath
        self.filepath = os.path.join(dirpath, datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv')
        self.mainscheduler.savefile = open(self.filepath,'w',newline='')
        csv.writer(self.mainscheduler.savefile,delimiter=';').writerow(['Time','Frequency [Hz]','Resistance [Ohm]'])
        self.mainscheduler.SAVEDATA = True
                




