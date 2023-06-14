################################################################################
# Company: University of Siena                                                 #
# Engineer: Riccardo Moretti                                                   #
# Project: QCMOscillator                                                       #
# Description: Application main function.                                      #
# Revision: v1.01 - Application execution added.                               #
#           v0.01 - File created.                                              #
################################################################################

import sys
from PySide6.QtWidgets import QApplication
from c_mainwindow import MainWindow

app = QApplication(sys.argv)
window = MainWindow()
sys.exit(app.exec())
