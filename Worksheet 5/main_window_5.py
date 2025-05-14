# -*- coding: utf-8 -*-
from qtpy.QtCore import Qt, QThread
from qtpy.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QProgressDialog
from qtpy.uic import loadUi
from qtpy.QtGui import QFont

import os
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import calfem.vis_mpl as cfv
import numpy as np

import flowmodel_5 as fm

def clean_ui(uifile):
    """Fix issues with Orientation:Horizontal/Vertical by creating _cleaned_mainwindow_5.ui"""
    tree = ET.parse(uifile)
    root = tree.getroot()
    for enum in root.findall(".//property[@name='orientation']/enum"):
        txt = enum.text or ''
        if 'Orientation::Horizontal' in txt:
            enum.text = 'Horizontal'
        elif 'Orientation::Vertical' in txt:
            enum.text = 'Vertical'
    clean_file = os.path.join(os.path.dirname(uifile), '_cleaned_mainwindow_5.ui')
    tree.write(clean_file, encoding='utf-8', xml_declaration=True)
    return clean_file

class SolverThread(QThread):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def run(self):
        self.solver.execute()

class MainWindow(QMainWindow):
    """Main window for the application."""
    def __init__(self):
        
        """Constructor for the main window."""
        super(QMainWindow, self).__init__()
   
        # Set up
        self.model_params = fm.ModelParams() # Model parameters
        self.visualization = None # Visualization object
        self.calc_done = False # Calculation status
       
        # Clean UI file and load interface description
        ui_path = os.path.join(os.path.dirname(__file__), 'mainwindow5.ui')
        loadUi(clean_ui(ui_path), self) #loads cleaned_mainwindow_5.ui

        # Font settings
        mono = QFont("ComicSans", 12)
        self.plainTextEdit.setFont(mono)

        # Export VTK
        if hasattr(self, 'actionExport_VTK'):
            self.actionExport_VTK.triggered.connect(self.on_export_vtk)
        elif hasattr(self, 'actionExportVTK'):
            self.actionExportVTK.triggered.connect(self.on_export_vtk)
        else:
            print("Warning: could not find Export VTK action in UI!")

        # Menu placement in ui window
        self.menuBar().setNativeMenuBar(False)

        # Element size slider
        self.element_size_label.setText('Element size:')
        self.element_size_slider.setRange(50, 100)

        # Set placeholders
        placeholders = {
            'w_text': '100.0 m', 
            'h_text': '10.0 m', 
            'd_text': '5.0 m', 
            't_text': '0.5 m',
            'kx_text': '20.0 m/day', 
            'ky_text': '20.0 m/day',
            'left_bc_text': '60.0 mvp', 
            'right_bc_text': '0.0 mvp'
        }

        # Set placeholder text
        for attr, text in placeholders.items():
            if hasattr(self, attr):
                widget = getattr(self, attr)
                widget.clear()
                widget.setPlaceholderText(text)

        # Set default values
        defaults = {
            'w_text':           str(self.model_params.w),
            'h_text':           str(self.model_params.h),
            'd_text':           str(self.model_params.d),
            't_text':           str(self.model_params.t),
            'kx_text':          str(self.model_params.kx),
            'ky_text':          str(self.model_params.ky),
            'left_bc_text':     str(self.model_params.bc_values['left_bc']),
            'right_bc_text':    str(self.model_params.bc_values['right_bc']),
            'dEndEdit':         str(self.model_params.d),
            'tEndEdit':         str(self.model_params.t),
        }

        # Set default values in UI
        for attr, val in defaults.items():
            if hasattr(self, attr):
                getattr(self, attr).setText(val)

            self.element_size_slider.setValue(int(self.model_params.el_size_factor * 100))
        
        # Set default values for the parameter study
        if hasattr(self, 'paramStep'):
            self.paramStep.setValue(4)

        # Clear checked radio buttons
        if hasattr(self, 'paramVaryDRadio'):
            self.paramVaryDRadio.setChecked(False)
        if hasattr(self, 'paramVaryTRadio'):
            self.paramVaryTRadio.setChecked(False)

        # Disable buttons initially
        for btn in (self.show_geometry_button, 
                    self.show_mesh_button,
                    self.show_nodal_values_button, 
                    self.show_element_values_button):
            btn.setEnabled(False)

        # Connect menu actions
        self.new_action.triggered.connect(self.on_new)
        self.open_action.triggered.connect(self.on_open)
        self.save_action.triggered.connect(self.on_save)
        self.save_as_action.triggered.connect(self.on_save_as)
        self.exit_action.triggered.connect(self.close)
        self.execute_action.triggered.connect(self.on_execute)
        self.paramButton.clicked.connect(self.on_execute_param_study)

        # Connect visualization buttons
        self.show_geometry_button.clicked.connect(self.on_show_geometry)
        self.show_mesh_button.clicked.connect(self.on_show_mesh)
        self.show_nodal_values_button.clicked.connect(self.on_show_nodal_values)
        self.show_element_values_button.clicked.connect(self.on_show_element_values)

        # Slider only updates element_size_factor
        self.element_size_slider.valueChanged.connect(self.on_element_size_change)

        self.model_params = None
        self.model_results = None

        self.show()
        self.raise_()

    def update_model(self):
        """Read UI fields into model_params and update boundary conditions."""

        # Ensure we have a ModelParams to write into
        if not self.model_params:
            self.model_params = fm.ModelParams()

        # Define the mapping
        fields = [
            ('w_text',       'w',        'Width of domain (w)'),
            ('h_text',       'h',        'Height of domain (h)'),
            ('d_text',       'd',        'Depth of barrier (d)'),
            ('t_text',       't',        'Thickness of barrier (t)'),
            ('kx_text',      'kx',       'Permeability in x (kx)'),
            ('ky_text',      'ky',       'Permeability in y (ky)'),
            ('left_bc_text', 'left_bc',  'Left surface pressure (mvp)'),
            ('right_bc_text','right_bc', 'Right surface pressure (mvp)'),
        ]

        invalid = []
       
        # Read values from UI fields and set them in model_params
        for widget_name, param_name, label in fields:
            widget = getattr(self, widget_name, None)
            txt = widget.text().strip() if widget else ''
            try:
                value = float(txt)
            except Exception:
                invalid.append(label)
            else:
                setattr(self.model_params, param_name, value)

        # Warnings for invalid inputs
        if invalid:
            QMessageBox.warning(
                self,
                'Invalid input',
                'Please enter valid numbers for:\n' + '\n'.join(invalid)
            )
            return False

        # Propagate into bc_values
        if hasattr(self.model_params, 'bc_values'):
            self.model_params.bc_values['left_bc']  = self.model_params.left_bc
            self.model_params.bc_values['right_bc'] = self.model_params.right_bc

        # Properties
        mp = self.model_params
        mp.D = np.array([[mp.kx, 0], [0, mp.ky]])
        mp.el_size_factor = self.element_size_slider.value() / 100.0

        return True

    def on_new(self):
        self.__init__()
        self.paramButton.setEnabled(True)

    def on_open(self):
        """Open a model file and load its parameters into the UI."""

        # Open file dialog to select a model file
        fn, _ = QFileDialog.getOpenFileName(self, 'Open model', '', 'Model files (*.json)')
        if not fn: return
        mp = fm.ModelParams()
        try:
            mp.load(fn)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Load failed: {e}')
            return
       
        # Set the model parameters in the UI
        for param, attr in [('w','w_text'), 
                            ('h','h_text'), 
                            ('d','d_text'), 
                            ('t','t_text'),
                             ('kx','kx_text'), 
                             ('ky','ky_text'),
                             ('left_bc','left_bc_text'), 
                             ('right_bc','right_bc_text')]:
                
                if not hasattr(self, attr):
                    continue
                if param in mp.bc_values:
                    text = str(mp.bc_values[param])
                else:
                    text = str(getattr(mp, param, ''))
                getattr(self, attr).setText(text)

        self.model_params = mp
        self.model_results = None
        for btn in (self.show_geometry_button, self.show_mesh_button,
                    self.show_nodal_values_button, self.show_element_values_button):
            btn.setEnabled(False)

    def on_save(self):
        """Save model parameters to the current file or prompt for a new file."""
        # Check if model_params is None or if update_model() fails
        if not self.model_params:
            QMessageBox.warning(self, 'Warning', 'Nothing to save or invalid data.')
            return
        fn = getattr(self.model_params, 'filename', None)
        if fn:
            try:
                self.model_params.save(fn)
            except Exception:
                self.on_save_as()
        else:
            self.on_save_as()

    def on_export_vtk(self):
        # Ensure we've actually run a solve (and have a solver to export from)
        if not hasattr(self, 'solver') or self.solver is None:
            QMessageBox.warning(self, "Nothing to export",
                                 "Please run the simulation first.")
            return

        # Prompt user for filename
        fn, _ = QFileDialog.getSaveFileName(
            self,
            "Export VTK File",
            "",
            "VTK files (*.vtk);; All files (*)"
        )
        if not fn:
            return

        # Delegate to the solver's export method
        try:
            self.solver.export_vtk(fn)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed",
                                 f"Could not write VTK:\n{e}")
            return

        QMessageBox.information(self, "Export Successful",
                                f"Wrote VTK file:\n{fn}")

    def on_save_as(self):
        """Prompt for a file name and save model parameters to that file."""
        # Ensure we have a model_params to save into
        if not self.model_params:
            self.model_params = fm.ModelParams()

        # Prompt for filename
        fn, _ = QFileDialog.getSaveFileName(self, 'Save As', '', 'Model files (*.json)')

        # If Cancel is pressed or no filename is provided
        if not fn:
            return

        # Save the model parameters to the specified file
        try:
            _ = self.update_model()
            self.model_params.save(fn)
            self.model_params.filename = fn
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Save failed: {e}')

    def on_execute(self):
        """Run solver unless already executed; prompt to start new model if so"""
        # Prevent re-execution
        if self.model_results is not None:
            QMessageBox.warning(
                self,
                'Execution Already Run',
                'To generate another domain create a new file.'
            )
            return

        # UI values into model_params
        if not self.update_model():
            return

        # Disable UI until solver finishes
        self.calc_done = False
        self.setEnabled(False)

        # Create fresh result & solver
        self.model_results = fm.ModelResult()
        self.solver = fm.ModelSolver(self.model_params, self.model_results)

        # Show “please wait” dialog
        progress = QProgressDialog("Running simulation…", None, 0, 0, self)
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()

        # Launch the solver in its thread
        self.solverThread = SolverThread(self.solver)
        self.solverThread.finished.connect(progress.close)
        self.solverThread.finished.connect(self.on_solver_finished)
        self.solverThread.start()

    def on_solver_finished(self):
        """Handle completion of the solver thread."""
        self.setEnabled(True)
        
        # Caluclation finished
        self.calc_done = True

        # Recreate visualization object
        self.visualization = fm.ModelVisualization(self.model_params, self.model_results)

        for btn in (self.show_geometry_button, self.show_mesh_button,
                    self.show_nodal_values_button, self.show_element_values_button):
            btn.setEnabled(True)

                # --- Build a neat, monospaced summary

        self.paramButton.setEnabled(False)

        mp = self.model_params
        mr = self.model_results

        lines = []
        lines.append("=== Model Inputs ===")
        # list of (label, value) tuples in desired order
        inputs = [
            ("Width (w)"       , mp.w),
            ("Height (h)"      , mp.h),
            ("Barrier depth (d)", mp.d),
            ("Thickness (t)"   , mp.t),
            ("Permeability kx" , mp.kx),
            ("Permeability ky" , mp.ky),
            ("Element size"    , mp.el_size_factor),
            ("Left BC"         , mp.bc_values["left_bc"]),
            ("Right BC"        , mp.bc_values["right_bc"]),
        ]
        # align the colon at column  twenty for neatness
        for label, val in inputs:
            lines.append(f"{label:<20s}: {val}")

        lines.append("")  # blank separator
        lines.append("=== Model Results ===")
        results = [
            ("Max nodal pressure"  , mr.max_nodal_pressure),
            ("Max nodal residual"  , mr.max_nodal_flow),
            ("Max element pressure", mr.max_element_pressure),
            ("Max element flow"    , mr.max_element_flow),
            ("Max element gradient", mr.max_element_gradient),
        ]
        for label, val in results:
            lines.append(f"{label:<20s}: {val:.4f}")

        # set into the UI
        self.plainTextEdit.setPlainText("\n".join(lines))


    def on_show_geometry(self):
        """Display the geometry of the model."""

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No Data', 'Please run the calculation first.')
            return
        self.visualization.show_geometry()

    def on_show_mesh(self):
        """Display the finite element mesh of the model."""

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No Data', 'Please run the calculation first.')
            return
        self.visualization.show_mesh()

    def on_show_nodal_values(self):
        """Display nodal values of the model."""

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No Data', 'Please run the calculation first.')
            return
        self.visualization.show_nodal_values()

    def on_show_element_values(self):
        """Display element values of the model."""

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No Data', 'Please run the calculation first.')
            return
        self.visualization.show_element_values()

    def on_element_size_change(self, value):
        """Update the element size factor based on the slider value."""
        if self.model_params is None:
            self.model_params = fm.ModelParams()
        self.model_params.el_size_factor = value / 100.0

    def on_execute_param_study(self):
        """Run a parameter study, either on depth (d) or thickness (t), and log results."""
        # Params from the UI
        if not self.update_model():
            return

        # Decide which parameter to vary
        if self.paramVaryDRadio.isChecked():
            var_name = 'd'
            start_val = self.model_params.d
            try:
                end_val = float(self.dEndEdit.text())
            except ValueError:
                QMessageBox.warning(self, 'Invalid Input',
                                    'Depth end‐value must be a number.')
                return
            xlabel = 'Barrier Depth d'
        elif self.paramVaryTRadio.isChecked():
            var_name = 't'
            start_val = self.model_params.t
            try:
                end_val = float(self.tEndEdit.text())
            except ValueError:
                QMessageBox.warning(self, 'Invalid Input',
                                    'Thickness end‐value must be a number.')
                return
            xlabel = 'Barrier Thickness t'
        else:
            QMessageBox.warning(self, 'Parameter Study',
                                'Please check “Vary d” or “Vary t” to enable a sweep.')
            return

        # Steps from the UI
        n_steps = self.paramStep.value()
        if n_steps < 2:
            QMessageBox.warning(self, 'Invalid Input',
                                'Number of steps must be at least 2.')
            return

        vals = np.linspace(start_val, end_val, n_steps)

        # Progress dialog
        progress = QProgressDialog(f"Running parameter study of {var_name}…",
                                   "Abort", 0, n_steps-1, self)
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        # Prepare for plotting
        self.plainTextEdit.clear()
        flows = []

        #
        for i, v in enumerate(vals):
            # allow cancellation
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled():
                break

            # Copy all other params into a fresh ModelParams
            base = self.model_params
            p = fm.ModelParams()
            p.w = base.w
            p.h = base.h
            p.d = v if var_name == 'd' else base.d
            p.t = v if var_name == 't' else base.t
            p.kx = base.kx
            p.ky = base.ky
            p.bc_markers = base.bc_markers
            p.bc_values= base.bc_values.copy()
            p.load_markers = base.load_markers
            p.load_values = base.load_values.copy()
            p.el_size_factor = base.el_size_factor

            # Solve the model
            mr = fm.ModelResult()
            solver = fm.ModelSolver(p, mr)
            solver.execute()

            mf = mr.max_element_flow
            flows.append(mf)

            # Log into the plainTextEdit
            self.plainTextEdit.appendPlainText(
                f"{var_name} = {v:.4g}  →  max element-flow = {mf:.4g}"
            )

        progress.setValue(n_steps-1)
        progress.close()

        cfv.figure()
        plt.clf()
        plt.plot(vals[:len(flows)], flows)
        plt.xlabel(xlabel)
        plt.ylabel('Max Element Flow')
        plt.title(f'Parameter Study: {xlabel} vs Max Element Flow')
        plt.grid(True)
        cfv.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
