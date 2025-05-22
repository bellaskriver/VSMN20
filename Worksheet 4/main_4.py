# -*- coding: utf-8 -*-
import sys
import main_window as mw
from qtpy.QtWidgets import QApplication

if __name__ == '__main__':

    # Create application instance
    app = QApplication(sys.argv)

    # Create and show main window
    widget = mw.MainWindow()
    widget.show()

    # Start main event loop
    sys.exit(app.exec_())