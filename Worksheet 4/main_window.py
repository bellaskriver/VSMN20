# -*- coding: utf-8 -*-
from qtpy.QtCore import QThread
from qtpy.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from qtpy.uic import loadUi
import xml.etree.ElementTree as ET
import os
import numpy as np
import flowmodel_4 as fm


def clean_ui(uifile):
    # Fix Qt4 enum issues for orientation properties
    tree = ET.parse(uifile)
    root = tree.getroot()
    for enum in root.findall(".//property[@name='orientation']/enum"):
        txt = enum.text or ''
        if 'Orientation::Horizontal' in txt:
            enum.text = 'Horizontal'
        elif 'Orientation::Vertical' in txt:
            enum.text = 'Vertical'
    clean_file = os.path.join(os.path.dirname(uifile), '_cleaned_mainwindow.ui')
    tree.write(clean_file, encoding='utf-8', xml_declaration=True)
    return clean_file


class SolverThread(QThread):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver
    def run(self):
        self.solver.execute()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Clean UI file to replace Qt4 enum values
        ui_path = os.path.join(os.path.dirname(__file__), 'mainwindow.ui')
        clean_path = clean_ui(ui_path)
        loadUi(clean_path, self)

        # Embed menu on macOS
        self.menuBar().setNativeMenuBar(False)

        # Slider configuration
        self.element_size_label.setText('Element size (0.5 - 1.0)')
        self.element_size_slider.setRange(50, 100)

        # Set input placeholders including boundary fields
        placeholders = {
            'w_text': '100.0', 'h_text': '10.0', 'd_text': '5.0', 't_text': '0.5',
            'kx_text': '20.0', 'ky_text': '20.0',
            'left_bc_text': '0.0', 'right_bc_text': '0.0'
        }
        for attr, text in placeholders.items():
            if hasattr(self, attr):
                widget = getattr(self, attr)
                widget.clear()
                widget.setPlaceholderText(text)

        # Disable visualization buttons initially
        for btn in (self.show_geometry_button, self.show_mesh_button,
                    self.show_nodal_values_button, self.show_element_values_button):
            btn.setEnabled(False)

        # Connect menu actions
        self.new_action.triggered.connect(self.on_new)
        self.open_action.triggered.connect(self.on_open)
        self.save_action.triggered.connect(self.on_save)
        self.save_as_action.triggered.connect(self.on_save_as)
        self.exit_action.triggered.connect(self.close)
        self.execute_action.triggered.connect(self.on_execute)

        # Connect visualization buttons
        self.show_geometry_button.clicked.connect(self.on_show_geometry)
        self.show_mesh_button.clicked.connect(self.on_show_mesh)
        self.show_nodal_values_button.clicked.connect(self.on_show_nodal_values)
        self.show_element_values_button.clicked.connect(self.on_show_element_values)

        # Slider update
        self.element_size_slider.valueChanged.connect(self.update_model)

        self.model_params = None
        self.model_results = None

        self.show()
        self.raise_()

    def update_model(self):
        """Read UI fields into model_params and update boundary conditions."""
        if not self.model_params:
            self.model_params = fm.ModelParams()
        def parse_attr(attr_name, param_name):
            if hasattr(self, attr_name):
                txt = getattr(self, attr_name).text().strip()
                if txt:
                    try:
                        return float(txt)
                    except ValueError:
                        QMessageBox.warning(self, 'Invalid input', f'Enter valid {param_name}')
                        raise
            return getattr(self.model_params, param_name)
        try:
            # Physical parameters
            self.model_params.w = parse_attr('w_text', 'w')
            self.model_params.h = parse_attr('h_text', 'h')
            self.model_params.d = parse_attr('d_text', 'd')
            self.model_params.t = parse_attr('t_text', 't')
            self.model_params.kx = parse_attr('kx_text', 'kx')
            self.model_params.ky = parse_attr('ky_text', 'ky')
            # Boundary conditions
            self.model_params.left_bc = parse_attr('left_bc_text', 'left_bc')
            self.model_params.right_bc = parse_attr('right_bc_text', 'right_bc')
            # Propagate into bc_values dict if present
            if hasattr(self.model_params, 'bc_values'):
                self.model_params.bc_values['left_bc'] = self.model_params.left_bc
                self.model_params.bc_values['right_bc'] = self.model_params.right_bc
        except ValueError:
            return False
        # Derived properties
        mp = self.model_params
        mp.D = np.array([[mp.kx, 0], [0, mp.ky]])
        mp.el_size_factor = self.element_size_slider.value() / 100.0
        return True

    def on_new(self):
        self.__init__()

    def on_open(self):
        fn, _ = QFileDialog.getOpenFileName(self, 'Open model', '', 'Model files (*.json)')
        if not fn: return
        mp = fm.ModelParams()
        try:
            mp.load(fn)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Load failed: {e}')
            return
        for param, attr in [('w','w_text'), ('h','h_text'), ('d','d_text'), ('t','t_text'),
                             ('kx','kx_text'), ('ky','ky_text'),
                             ('left_bc','left_bc_text'), ('right_bc','right_bc_text')]:
            if hasattr(self, attr):
                getattr(self, attr).setText(str(getattr(mp, param)))
        self.model_params = mp
        self.model_results = None
        for btn in (self.show_geometry_button, self.show_mesh_button,
                    self.show_nodal_values_button, self.show_element_values_button):
            btn.setEnabled(False)

    def on_save(self):
        if not self.model_params or not self.update_model():
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

    def on_save_as(self):
        if not self.model_params or not self.update_model():
            QMessageBox.warning(self, 'Warning', 'Nothing to save or invalid data.')
            return
        fn, _ = QFileDialog.getSaveFileName(self, 'Save As', '', 'Model files (*.json)')
        if fn:
            try:
                self.model_params.save(fn)
                self.model_params.filename = fn
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Save failed: {e}')

    def on_execute(self):
        """Run solver unless already executed; prompt to start new model if so"""
        # Prevent re-execution on the same parameters
        if self.model_results is not None:
            QMessageBox.warning(
                self,
                'Execution Already Run',
                'To generate another domain create a new file.'
            )
            return
        # Ensure all parameters are entered
        if not self.update_model():
            QMessageBox.warning(
                self,
                'Missing Parameters',
                'Please fill in all parameters before execution.'
            )
            return
        self.setEnabled(False)
        # Start solver thread
        self.model_results = fm.ModelResult()
        self.solver_thread = SolverThread(
            fm.ModelSolver(self.model_params, self.model_results)
        )
        self.solver_thread.finished.connect(self.on_solver_finished)
        self.solver_thread.start()

    def on_solver_finished(self):
        self.setEnabled(True)
        for btn in (self.show_geometry_button, self.show_mesh_button,
                    self.show_nodal_values_button, self.show_element_values_button):
            btn.setEnabled(True)

    def on_show_geometry(self):
        import calfem.vis_mpl as cfv
        cfv.figure(); cfv.clf()
        cfv.draw_geometry(self.model_params.geometry(), draw_points=True,
                          label_points=True, label_curves=True)
        cfv.show_and_wait()

    def on_show_mesh(self):
        import calfem.vis_mpl as cfv
        cfv.figure(); cfv.clf()
        cfv.draw_mesh(coords=self.model_results.coords,
                      edof=self.model_results.edof,
                      dofs_per_node=self.model_results.dofs_per_node,
                      el_type=self.model_results.el_type,
                      filled=True)
        cfv.show_and_wait()

    def on_show_nodal_values(self):
        vis = fm.ModelVisualization(self.model_params, self.model_results)
        vis.show_nodal_values(); vis.wait()

    def on_show_element_values(self):
        vis = fm.ModelVisualization(self.model_params, self.model_results)
        vis.show_element_values(); vis.wait()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
