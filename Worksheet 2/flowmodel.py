# -*- coding: utf-8 -*-

import numpy as np
import calfem.core as cfc
import calfem.utils as cfu
import tabulate as tab
import json

class ModelParams:
    """Class defining the model parameters"""
    def __init__(self):

        self.version = 1
        
        self.t = 1
        self.ep = [self.t]
        
        # --- Element properties

        self.k_x = 50  # m/day
        self.k_y = 50  # m/day
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

        '''
        self.dof = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12]
        ])
        '''

        self.ex = np.array([
            [0.0, 0.0, 600.0],
            [0.0, 600.0, 600.0],
            [600.0, 600.0, 1200.0],
            [600.0, 1200.0, 1200.0]
        ])

        self.ey = np.array([
            [0.0, 600.0, 600.0],
            [0.0, 600.0, 0.0],
            [0.0, 600.0, 600.0],
            [0.0, 600.0, 0.0]
        ])
        # --- Element topology 
               
        self.edof = np.array([
            [1, 2, 4],
            [1, 4, 3],
            [3, 4, 6],
            [3, 6, 5]
        ])

        # --- Loads

        self.loads = [
            [6, -400]
        ]

        # --- Boundary conditions

        self.bcs = [
            [2, 60.],
            [4, 60.],
            [1, 0.],  # Added constraint to fix the first degree of freedom
            [3, 0.]   # Added constraint to fix the third degree of freedom
        ]

        self.dof = None

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
        self.qs = None
        self.qt = None

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

        # --- Calculate element stiffness matrices and assemble global stiffness matrix
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
                print(load)
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

        dof = np.array([1, 2, 3, 4, 5, 6])

        bc_prescr = np.array(bc_prescr)
        bc_value = np.array(bc_value)

        a, r = cfc.solveq(K, f, bc_prescr, bc_value)
    
        print(a)

        ed = cfc.extractEldisp(edof, a) 

        n_el = edof.shape[0]  # 4
        es = np.zeros((n_el, 2))
        et = np.zeros((n_el, 2))

        for i in range(n_el):
            es[i,:], et[i,:] = cfc.flw2ts(ex[i,:], ey[i,:], D, ed[i,:])

    # Combinine multiple arrays
        a_and_r = np.hstack((a, r))

        temp_table = tab.tabulate(
            np.asarray(a_and_r),
            headers=["D.o.f.", "Phi [m]", "q [m^2/day]"],
            numalign="right",
            floatfmt=".4f",
            tablefmt="psql",
            showindex=range(1, len(a_and_r) + 1),
        )

        # Calculate element flows and gradients
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
        #self.model_result.qs = qs #problem här
        #self.model_result.qt = qt #problem här


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
        self.add_text("Coordinates:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.coord, headers=["x", "y"], tablefmt="psql")
        )
        
        self.add_text("Element topology:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.edof, headers=["Node 1", "Node 2", "Node 3"], tablefmt="psql")
        )
        self.add_text()
        self.add_text("Boundary conditions:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.bcs, headers=["D.o.f.", "Value"], tablefmt="psql")
        )
        self.add_text()
        self.add_text("Loads:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.loads, headers=["D.o.f.", "Value"], tablefmt="psql")
        )
        self.add_text()
        self.add_text("-------------- Model results --------------------------------")
        self.add_text()
        self.add_text("Nodal values:")
        self.add_text()
        self.add_text(tab.tabulate(np.asarray(np.hstack((self.model_result.a, self.model_result.r))),
            headers=["Phi [m]", "q [m^2/day]"],
            numalign="right",
            floatfmt=".4f",
            tablefmt="psql",
           # showindex=range(1, len(a_and_r) + 1),
            ))

        '''
        self.add_text()
        self.add_text("Reaction forces:")
        self.add_text()
        self.add_text(
            tab.tabulate(np.column_stack((np.arange(1, len(self.model_result.r) + 1), self.model_result.r)),
            headers=["Reaction [m^2/day]"],
            tablefmt="psql",
            floatfmt=".4f"
            )
        )'''

        '''
        self.add_text()
        self.add_text("Element flows and gradients:")
        self.add_text()
        self.add_text(
            tab.tabulate(np.column_stack((np.arange(1, len(self.model_result.qs) + 1), self.model_result.qs, self.model_result.qt)),
            headers=["Element", "Flow [m^2/day]", "Gradient [m/m]"],
            tablefmt="psql",
            floatfmt=".4f"
            )
        )'
        '''

        return self.report