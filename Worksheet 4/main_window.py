# -*- coding: utf-8 -*-
import sys

from qtpy.QtCore import QThread
from qtpy.QtWidgets import QApplication, QDialog, QWidget, QMainWindow, QFileDialog
from qtpy.uic import loadUi

import calfem.ui as cfui
import flowmodel_4 as fm

class SolverThread(QThread):
    """Klass för att hantera beräkning i bakgrunden"""

    def __init__(self, solver, param_study = False):
        """Klasskonstruktor"""
        QThread.__init__(self)
        self.solver = solver
        self.param_study = param_study

    def __del__(self):
        self.wait()

    def run(self):
        self.solver.execute()

class MainWindow(QMainWindow):
    """MainWindow-klass som hanterar vårt huvudfönster"""

    def __init__(self):
        """Constructor"""
        super(QMainWindow, self).__init__()

        # --- Load user interface description

        loadUi('mainwindow.ui', self)

        # --- Show the window

        self.show()
        self.raise_()
        

        # --- Connect controls to event methods

        self.new_action.triggered.connect(self.on_new_action)

        # --- Connect controls to event methods

        self.new_action.triggered.connect(self.on_new_action)
        
        self.show_geometry_button.clicked.connect(self.on_show_geometry) # <---


    def on_show_geometry(self):
        """Visa geometrifönster"""

        print("on_show_geometry")

    def on_new_action(self):
        """Skapa en ny modell"""
        print("on_new_action")

    def update_controls(self):
        """Fyll kontrollerna med värden från modellen"""
        self.w_edit.setText(str(self.model_params.w))
        self.h_edit.setText(str(self.model_params.h))
        self.d_edit.setText(str(self.model_params.d))
        self.t_edit.setText(str(self.model_params.t))
        self.kx_edit.setText(str(self.model_params.kx))
        self.ky_edit.setText(str(self.model_params.ky))
        self.el_size_factor_edit.setText(str(self.model_params.el_size_factor))


    def update_model(self):
        """Hämta värden från kontroller och uppdatera modellen"""
        self.model_params.w = float(self.w_edit.text())
        self.model_params.h = float(self.h_edit.text())
        self.model_params.d = float(self.d_edit.text())
        self.model_params.t = float(self.t_edit.text())
        self.model_params.kx = float(self.kx_edit.text())
        self.model_params.ky = float(self.ky_edit.text())
        self.model_params.el_size_factor = float(self.el_size_factor_edit.text())

    def on_open_action(self):
        """Öppna in indata fil"""

        filename, _ = QFileDialog.getOpenFileName(self, 
            "Öppna modell", "", "Modell filer (*.json *.jpg *.bmp)")

        if filename!="":
            self.filename = filename

            # --- Open ModelParams instance

    def on_save_action(self):
        """Spara modell"""

        self.update_model()

        if self.filename == "":
            filename, _  = QFileDialog.getSaveFileName(self, 
                "Spara modell", "", "Modell filer (*.json)")

            if filename!="":
                self.filename = filename

        # --- Save ModelParams instance

        def on_action_execute(self):
            """Kör beräkningen"""

            # --- Disable user interface during calculation     

            self.setEnabled(False)

            # --- Update model from user interface

            self.update_model()

            # --- Create a solver

            self.solver = fm.ModelSolver(self.model_params, self.model_results)

            # --- Create a thread with the calculation, so that the 
            #     user interface doesn't freeze.

            self.solver_thread = SolverThread(self.solver)   
            self.solver_thread.finished.connect(self.on_model_solver_finished)  
            self.solver_thread.start()

        def on_solver_finished(self):
            """Anropas när beräkningstråden avslutas"""

            # --- Activate user interface       

            self.setEnabled(True)

            # --- Generate result report     