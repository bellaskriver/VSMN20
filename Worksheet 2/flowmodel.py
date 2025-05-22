# -*- coding: utf-8 -*-
import numpy as np
import calfem.core as cfc
import tabulate as tab
import json

class ModelParams:
    """Class defining the model parameters"""
    def __init__(self):

        self.version = 1
        
        self.t = 1 # Thickness
        self.ep = [self.t] # Element properties
        
        # --- Element properties
        self.k_x = 50  # Permeability in x-direction
        self.k_y = 50  # Permeability in y-direction
        self.D = np.array([
            [self.k_x, 0],
            [0, self.k_y]
        ])

        # --- Element coordinates
        self.coord = np.array([
            [0.0, 0.0],
            [0.0, 600.0],
            [600.0, 0.0],
            [600.0, 600.0],
            [1200.0, 0.0],
            [1200.0, 600.0]
        ])

        # --- Element coordinates in x
        self.ex = np.array([
            [0.0, 600.0, 0.0],
            [0.0, 600.0, 600.0],
            [600.0, 1200.0, 600.0],
            [600.0, 1200.0, 1200.0]
        ])

        # --- Element coordinates in y
        self.ey = np.array([
            [0.0, 600.0, 600.0],
            [0.0, 0.0, 600.0],
            [0.0, 600.0, 600.0],
            [0.0, 0.0, 600.0]
        ])

        # --- Element topology 
        self.edof = np.array([
            [1, 4, 2],
            [1, 3, 4],
            [3, 6, 4],
            [3, 5, 6]
        ])

        # --- Loads
        self.loads = [
            [6, -400]
        ]

        # --- Boundary conditions
        self.bcs = [
            [2, 60.],
            [4, 60.]
        ]

        # --- Nodal degrees of freedom
        self.dof = np.array([1, 2, 3, 4, 5, 6])

        # --- Element numbers
        self.elem = np.array([1, 2, 3, 4])

    def save(self, filename):
        """Save input to file."""
        model_params = {}
        model_params["version"] = self.version
        model_params["t"] = self.t
        model_params["ep"] = self.ep

        model_params["coord"] = self.coord.tolist()  # Convert NumPy array to list for JSON compatibility

        ofile = open(filename, "w")
        json.dump(model_params, ofile, sort_keys = True, indent = 4)
        ofile.close()

    def load(self, filename):
        """Read input from file."""

        ifile = open(filename, "r")
        model_params = json.load(ifile)
        ifile.close()

        self.version = model_params["version"]
        self.t = model_params["t"]
        self.ep = model_params["ep"]
        self.coord = np.asarray(model_params["coord"])

class ModelResult:
    """Class for storing results from calculations."""
    def __init__(self):
        self.a = None
        self.r = None
        self.ed = None

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
        ex = self.model_params.ex
        ey = self.model_params.ey
        loads = self.model_params.loads
        bcs = self.model_params.bcs
        D = self.model_params.D

        # --- Create global stiffness matrix and load vector
        K = np.zeros((6, 6))
        f = np.zeros((6, 1))

        f[5] = -400

        # --- Calculate element stiffness matrices and assemble global stiffness matrix
        ke1 = cfc.flw2te(ex[0,:], ey[0,:], ep, D)
        ke2 = cfc.flw2te(ex[1,:], ey[1,:], ep, D)
        ke3 = cfc.flw2te(ex[2,:], ey[2,:], ep, D)
        ke4 = cfc.flw2te(ex[3,:], ey[3,:], ep, D)
        
        # --- Assemble global stiffness matrix
        cfc.assem(edof[0, :], K, ke1, f)
        cfc.assem(edof[1, :], K, ke2, f)
        cfc.assem(edof[2, :], K, ke3, f)
        cfc.assem(edof[3, :], K, ke4, f)

        # --- Calculate element flow and gradient vectors
        for load in loads:
                dof = load[0]
                mag = load[1]
                f[dof - 1] = mag

        bc_prescr = []
        bc_value = []

        for bc in bcs:
            dof = bc[0]
            value = bc[1]
            bc_prescr.append(dof)
            bc_value.append(value)

        bc_prescr = np.array(bc_prescr)
        bc_value = np.array(bc_value)

        a, r = cfc.solveq(K, f, bc_prescr, bc_value)

        ed = cfc.extractEldisp(edof, a) 

        n_el = edof.shape[0]  # 4
        es = np.zeros((n_el, 2))
        et = np.zeros((n_el, 2))

        # --- Combinine multiple arrays
        a_and_r = np.hstack((a, r))

        temp_table = tab.tabulate(
            np.asarray(a_and_r),
            headers=["D.o.f.", "Phi [m]", "q [m^2/day]"],
            numalign="right",
            floatfmt=".4f",
            tablefmt="psql",
            showindex=range(1, len(a_and_r) + 1),
        )

        # --- Calculate element flows and gradients
        es = np.zeros([n_el, 2])
        et = np.zeros([n_el, 2])

        for elx, ely, eld, eles, elet in zip(ex, ey, ed, es, et):
            es_el, et_el = cfc.flw2ts(elx, ely, D, eld)
            eles[:] = es_el[0, :]
            elet[:] = et_el[0, :]

        # --- Store results in model_results

        self.model_result.a = a
        self.model_result.r = r
        self.model_result.ed = ed
        self.model_result.es = es
        self.model_result.et = et  

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
        self.add_text("-------------- Model Inputs ----------------------------------")
        self.add_text()
        self.add_text("Input parameters:")
        self.add_text()
        self.add_text(tab.tabulate(np.asarray([np.hstack((self.model_params.t, self.model_params.k_x))]), 
            headers=["t", "k"],
            numalign="right",
            floatfmt=".0f", 
            tablefmt="psql",
        ))

        self.add_text()
        self.add_text("Coordinates:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.coord, headers=["x", "y"], tablefmt="psql")
        )

        self.add_text()
        self.add_text("Dofs:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.dof.reshape(-1, 1), headers=["D.o.f."], tablefmt="psql")
        )

        self.add_text()
        self.add_text("Element topology:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.edof, headers=["Node 1", "Node 2", "Node 3"], tablefmt="psql")
        )

        self.add_text()
        self.add_text("Loads:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.loads, headers=["D.o.f.", "Value"], tablefmt="psql")
        )

        self.add_text()
        self.add_text("Boundary conditions:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.bcs, headers=["D.o.f.", "Value"], tablefmt="psql")
        )
       
        self.add_text()
        self.add_text("-------------- Model results --------------------------------")
        self.add_text()
        self.add_text("Nodal pressure and flows (a and r):")
        self.add_text()
        dof = self.model_params.dof.flatten().reshape(-1, 1)
        a = np.array(self.model_result.a).flatten().reshape(-1, 1)
        r = np.array(self.model_result.r).flatten().reshape(-1, 1)
        self.add_text(tab.tabulate(np.hstack((dof, a, r)),
            headers=["D.o.f.", "Phi [m]", "q [m^2/day]"],
            numalign="right",
            floatfmt=(".0f", ".4f", ".4f"),
            tablefmt="psql",
            ))

        self.add_text()
        self.add_text("Element flows (es):")
        self.add_text()
        self.add_text(tab.tabulate(np.asarray(np.hstack((self.model_params.elem.reshape(-1, 1), self.model_result.es))),
            headers=["Element", "q_x [m^2/day]", "q_y [m^2/day]"],
            numalign="right",
            floatfmt=(".0f", ".4f", ".4f"),
            tablefmt="psql",
            ))

        self.add_text()
        self.add_text("Element gradients (et):")
        self.add_text()
        self.add_text(tab.tabulate(np.asarray(np.hstack((self.model_params.elem.reshape(-1, 1), self.model_result.et))),
            headers=["Element", "g_x [-]", "g_y [-]"],
            numalign="right",
            floatfmt=(".0f", ".4f", ".4f"),
            tablefmt="psql",
            ))
        
        self.add_text()
        self.add_text("Element pressure (ed):")
        self.add_text()
        self.add_text(tab.tabulate(np.asarray(np.hstack((self.model_params.elem.reshape(-1, 1), self.model_result.ed))),
            headers=["Element", "Phi_1 [m]", "Phi_2 [m]", "Phi_3 [m]"],
            numalign="right",
            floatfmt=(".0f", ".4f", ".4f", ".4f"),
            tablefmt="psql",
            ))

        return self.report