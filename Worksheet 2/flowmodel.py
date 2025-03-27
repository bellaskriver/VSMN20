# -*- coding: utf-8 -*-

import numpy as np
import calfem.core as cfc
import calfem.utils as cfu
import tabulate as tab
import flowmodel as fm

if __name__ == "__main__":

    model_params = fm.ModelParams()
    model_result = fm.ModelResult()

    solver = fm.ModelSolver(model_params, model_result)
    solver.execute()

    report = fm.ModelReport(model_params, model_result)
    print(report)

class ModelParams:
    """Class defining the model parameters"""
    def __init__(self):

        self.version = 1

        self.t = 1
        self.ep = [self.t]

        # --- Element properties

        ...

        # --- Create input for the example use cases

        self.coord = np.array([
            [0.0, 0.0],
            [0.0, 0.12],
            ...
            [0.24, 0.12]
        ])

        # --- Element topology

        self.edof = np.array([
            [...],
            ...
            [...]
        ])

        # --- Loads

        self.loads = [
            [5, 6.0],
            [6, 6.0]
        ]

        # --- Boundary conditions

        self.bcs = [
            [1, -15.0],
            [2, -15.0]
        ]


class ModelSolver:
    
    """Class for performing the model computations."""
    def __init__(self, model_params, model_result):
        self.model_params = model_params
        self.model_result = model_result

    def execute(self):
     # --- Assign shorter variable names from model properties
        edof = self.model_params.edof
        coord = self.model_params.coord
        dof = self.model_params.dof
        ep = self.model_params.ep
        loads = self.model_params.loads
        bcs = self.model_params.bcs 

        # Beräkningskod


        # --- Store results in model_results

        self.model_result.a = a
        self.model_result.r = r
        self.model_result.ed = ed
        self.model_result.qs = qs
        self.model_result.qt = qt


class ModelResult:
    """Class for storing results from calculations."""
    def __init__(self):
        self.a = None
        self.r = None
        self.ed = None
        self.qs = None
        self.qt = None

        if self.a == None:
            self.a = np.array(...) # Tilldela en riktigt datatyp om self.a == None


class ModelReport:
    """Class for presenting input and output parameters in report form."""
    def __init__(self, model_params, model_result):
        self.model_params = model_params
        self.model_result = model_result
        self.report = ""

    def clear(self):
        self.report = ""

    def add_text(self, text=""):
        self.report+=str(text)+"\n"

    def __str__(self):
        self.clear()
        self.add_text()
        self.add_text("-------------- Model input ----------------------------------")
        ...
        self.add_text("Coordinates:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.params.coords, headers=["x", "y"], tablefmt="psql")
        )
        ...
        return self.report