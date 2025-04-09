# -*- coding: utf-8 -*-

import json
import math
import sys

import calfem.core as cfc
import calfem.geometry as cfg  # For geometric modeling
import calfem.mesh as cfm      # For mesh generation
import calfem.vis_mpl as cfv   # For visualization
import calfem.utils as cfu     # Utility functions

import matplotlib as plt
import PyQt5 as qt 
import tabulate as tab
import numpy as np

# --- Class definitions

# --- Class for defining model parameters
class ModelParams:
    """Class defining parametric model properties"""
    def __init__(self):

        # Version tracking

        self.version = 1

        # Geometric parameters (example for groundwater problem)

        self.w = 100.0  # Width of domain
        self.h = 10.0   # Height of domain
        self.d = 5.0    # Depth of barrier
        self.t = 0.5    # Thickness of barrier

        self.ep = [self.t]  # Element properties (thickness)

        # Material properties (example for groundwater problem)

        self.kx = 20.0  # Permeability in x-direction
        self.ky = 20.0  # Permeability in y-direction
        self.D = np.array([[self.kx, 0], [0, self.ky]]) #bella
    
        # Mesh control

        self.el_size_factor = 1  # Controls element size in mesh generation

        # Boundary conditions and loads will now reference markers 
        # instead of node numbers or degrees of freedom

        self.bc_markers = {
            "left_bc": 10,    # Marker for left boundary
            "right_bc": 20   # Marker for right boundary
        }

        self.bc_values = {
            "left_bc": 60.0,  # Value for left boundary
            "right_bc": 60.0  # Value for right boundary
        }

        self.load_markers = {
            #"bottom_load": 30,  # Marker for bottom boundary load
           # "top_load": 40      # Marker for top boundary load
        }

        self.load_values = {
            #"bottom_load": -400.0,  # Load value for bottom boundary
           # "top_load": 0.0        # Load value for top boundary
        }

    def geometry(self):
        """Create and return a geometry instance based on defined parameters"""

        # Create a geometry instance

     

        # Use shorter variable names for readability

        w = self.w
        h = self.h
        t = self.t
        d = self.d

        g = cfg.Geometry()

        # Define points for the geometry
        # Point indices start at 0

        g.point([0, 0])          # Point 0: Bottom left corner
        g.point([w, 0])          # Point 1: Bottom right corner
        g.point([w, h])          # Point 2: Top right corner
        g.point([w/2+t/2, h])          # Point 3: Top left corner
        g.point([w/2+t/2, h-d])          # Point 4: Top left corner
        g.point([w/2-t/2, h-d])          # Point 5: Bottom left corner
        g.point([w/2-t/2, h])          # Point 6: Bottom right corner
        g.point([0, h])          # Point 7: Top left corner

        # Define splines (lines) connecting the points
        # Use markers for boundaries with conditions

        left_side = 80
        right_side = 90

        g.spline([0, 1])                         # Bottom boundary
        g.spline([1, 2])                         # Right boundary, marker for fixed value
        g.spline([2, 3], marker=self.bc_markers["right_bc"])
        g.spline([3, 4])                         # Top of barrier
        g.spline([4, 5])  # Left boundary, marker for fixed value
        g.spline([5, 6])
        g.spline([6, 7], marker=self.bc_markers["left_bc"])                         # Left side of barrier
        g.spline([7, 0])                         # Right side of barrier

        # Define the surface (domain) using the spline indices
        # Surface is defined by a list of spline indices that form a closed loop


        g.surface([0, 1, 2, 3, 4, 5, 6, 7])

        # ----- Mesh generation ------------------------------------------

        el_type = 3  # quadrature element
        dofs_per_node = 1  # 1 dof

        # ----- Set mesh paramters ---------------------------------------

        mesh = cfm.GmshMeshGenerator(g)
        mesh.el_size_factor = 1.0
        mesh.el_type = el_type
        mesh.dofs_per_node = dofs_per_node

        # ----- Create mesh ----------------------------------------------

        coords, edof, dofs, bdofs, elementmarkers = mesh.create()

        # ----- Assemble elements ----------------------------------------

        nDofs = np.size(dofs)
        ex, ey = cfc.coordxtr(edof, coords, dofs)
        K = np.zeros([nDofs, nDofs])

        for eltopo, elx, ely in zip(edof, ex, ey):
            Ke = cfc.flw2i4e(elx, ely, ep, D)
            cfc.assem(eltopo, K, Ke)

        # ----- Force vector ---------------------------------------------

        f = np.zeros([nDofs, 1])

        # ----- Boundary conditions --------------------------------------

        bc = np.array([], int)
        bcVal = np.array([], int)

        bc, bcVal = cfu.applybc(bdofs, bc, bcVal, left_side, 0.0)
        bc, bcVal = cfu.applybc(bdofs, bc, bcVal, right_side, 10.0)

        # ----- Solve equation system ------------------------------------

        a, r = cfc.solveq(K, f, bc, bcVal)
        ed = cfc.extractEldisp(edof, a)

        # ----- Calculating element forces -------------------------------

        maxFlow = []  # empty list to store flow

        for i in range(edof.shape[0]):
            es, et, eci = cfc.flw2i4s(ex[i, :], ey[i, :], ep, D, ed[i, :])
            maxFlow.append(np.sqrt(pow(es[0, 0], 2) + pow(es[0, 1], 2)))

        # ----- Visualize results ----------------------------------------

        cfv.figure()
        cfv.draw_geometry(g, title="Geometry")

        cfv.figure()
        cfv.draw_element_values(
            maxFlow, coords, edof, dofs_per_node, el_type, None, title="Max flows"
        )

        cfv.figure()
        cfv.draw_nodal_values(a, coords, edof, dofs_per_node=dofs_per_node, el_type=el_type)

        cfv.showAndWait()
   
        # Return the complete geometry
        return g
    
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
    
# --- Class for storing results from calculations
class ModelResult:
    """Class for storing results from calculations."""
    def __init__(self):
        self.a = None
        self.r = None
        self.ed = None
        self.es = None
        self.et = None
        self.geometry = None
        self.coords = None
        self.edof = None
        self.dofs = None
        self.bdofs = None
        self.element_markers = None
        self.boundary_elements = None
        self.el_type = None
        self.dofs_per_node = None
        self.max_value = None
        self.min_value = None
        self.element_values = None
        self.nodal_values = None

# --- Class for visualizing model geometry, mesh, and results   
class ModelVisualization:
    """Class for visualizing model geometry, mesh, and results"""

    def __init__(self, model_params, model_result):
        """Constructor"""
        self.model_params = model_params
        self.model_result = model_result

        # Store references to visualization windows

        self.geom_fig = None
        self.mesh_fig = None
        self.nodal_val_fig = None
        self.element_val_fig = None
        self.deformed_fig = None

    def show_geometry(self):
        """Display model geometry"""

        # Get the geometry from results

        geometry = self.model_result.geometry

        # Create a new figure

        cfv.figure()
        cfv.clf()

        # Draw geometry

        cfv.draw_geometry(geometry, draw_points=True, label_points=True, 
                         draw_line_numbers=True, title="Model Geometry")

    def show_mesh(self):
        """Display finite element mesh"""

        # Create a new figure

        cfv.figure()
        cfv.clf()

        # Draw mesh

        cfv.draw_mesh(
            coords=self.model_result.coords,
            edof=self.model_result.edof,
            dofs_per_node=self.model_result.dofs_per_node,
            el_type=self.model_result.el_type,
            filled=True,
            title="Finite Element Mesh"
        )

    def show_nodal_values(self):
        """Display nodal values (e.g., temperature, pressure)"""

        cfv.figure()
        cfv.clf()

        # Draw nodal values
        cfv.draw_nodal_values(
            self.model_result.a,
            coords=self.model_result.coords,
            title="Nodal Values"
        )

    def show_element_values(self):
        """Display element values (e.g., flows, stresses)"""
        cfv.figure()
        cfv.clf()

        # Calculate element values (e.g., magnitude of flow vectors)
        element_values = np.sqrt(np.sum(self.model_result.es**2, axis=1))

        # Draw element values
        cfv.draw_element_values(
            element_values,
            coords=self.model_result.coords,
            edof=self.model_result.edof,
            dofs_per_node=self.model_result.dofs_per_node,
            el_type=self.model_result.el_type,
            title="Element Values"
        )

    def wait(self):
        """Wait for user to close all visualization windows"""
        cfv.show_and_wait()

# --- Class for solving the finite element model
class ModelSolver:
    """Class for solving the finite element model"""
    def __init__(self, model_params, model_result):
        self.model_params = model_params
        self.model_result = model_result

    def execute(self):
        """Perform mesh generation and finite element computation"""

        # Create shorter references to input variables

        ep = self.model_params.ep
        kx = self.model_params.kx
        ky = self.model_params.ky
        D = self.model_params.D

        # Get geometry from model_params

        geometry = self.model_params.geometry()

        # Store geometry in results for visualization

        self.model_result.geometry = geometry

        # Set up mesh generation

        el_type = 3        # 3 = 4-node quadrilateral element
        dofs_per_node = 1  # 1 for scalar problem (flow, heat), 2 for vector (stress)

        # Create mesh generator

        mesh = cfm.GmshMeshGenerator(geometry)

        # Configure mesh generator

        mesh.el_type = el_type
        mesh.dofs_per_node = dofs_per_node
        mesh.el_size_factor = self.model_params.el_size_factor
        mesh.return_boundary_elements = True

        # Generate mesh
        
        try:
            coords, edof, dofs, bdofs, element_markers, boundary_elements = mesh.create()
        except TypeError as e:
            print(f"Error during mesh creation: {e}")
            print("Please check the geometry definition and mesh generator configuration.")
            sys.exit(1)

        # Store mesh data in results

        self.model_result.coords = coords
        self.model_result.edof = edof
        self.model_result.dofs = dofs
        self.model_result.bdofs = bdofs
        self.model_result.element_markers = element_markers
        self.model_result.boundary_elements = boundary_elements
        self.model_result.el_type = el_type
        self.model_result.dofs_per_node = dofs_per_node
         
        # --- Create global stiffness matrix and load vector
        n_dofs = np.max(dofs)
        K = np.zeros((n_dofs, n_dofs))
        f = np.zeros((n_dofs, 1))

        # --- Apply loads to the load vector
        for marker_name, marker_id in self.model_params.load_markers.items():
            if marker_name in self.model_params.load_values:
                value = self.model_params.load_values[marker_name]
                cfu.apply_force_from_markers(
                    bdofs,
                    boundary_elements,
                    marker_id,
                    f,
                    value
                )

        # --- Calculate element stiffness matrices and assemble global stiffness matrix
        ke_list = []
        for coord in coords:
            ke = cfc.flw2te(coord, coord, ep, D)
            ke_list.append(ke)
        
        # --- Assemble global stiffness matrix
        for i, ke in enumerate(ke_list):
            cfc.assem(edof[i, :], K, ke, f)

        # --- Calculate element flow and gradient vectors
        '''
        for load in loads: #bella
                dof = load[0]
                mag = load[1]
                f[dof - 1] = mag

        bc_prescr = []
        bc_values = []

        for bc in bcs: #bella
            dof = bc[0]
            value = bc[1]
            bc_prescr.append(dof)
            bc_value.append(value) 

        bc_prescr = np.array(bc_prescr)
        bc_value = np.array(bc_value)

        '''

        bc_prescr = []
        bc_values = []

        # For each boundary condition marker in model_params

        for marker_name, marker_id in self.model_params.bc_markers.items():
            if marker_name in self.model_params.bc_values:
                value = self.model_params.bc_values[marker_name]
                cfu.apply_bc_from_markers(
                    bdofs,
                    boundary_elements,
                    marker_id,
                    bc_prescr,
                    bc_values,
                    value
                )

        # Convert to numpy arrays

        bc_prescr = np.array(bc_prescr)
        bc_values = np.array(bc_values)

        # Apply loads based on markers

        for marker_name, marker_id in self.model_params.load_markers.items():
            if marker_name in self.model_params.load_values:
                value = self.model_params.load_values[marker_name]
                cfu.apply_force_from_markers(
                    bdofs,
                    boundary_elements,
                    marker_id,
                    f,
                    value
                )

        # Solve equation system

        a, r = cfc.solveq(K, f, bc_prescr, bc_values)

        # Store displacement and reaction forces

        self.model_result.a = a
        self.model_result.r = r

        ed = cfc.extractEldisp(edof, a) 

        '''maxflow = []
        for i in range(edof.shape[0]):
            es, et, eci = cfc.flw2ts(ex[i, :], ey[i, :], D, ed[i, :])


            cfv.figure()
            cfv.draw_geometry(g, title="Geometry)

            '''

        n_el = edof.shape[0]  # 4
        es = np.zeros((n_el, 2))
        et = np.zeros((n_el, 2))

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

        #
        ex, ey = cfu.coord_extract(edof, coords)

        for elx, ely, eld, eles, elet in zip(ex, ey, ed, es, et): #bella
            es_el, et_el = cfc.flw2ts(elx, ely, D, eld)
            eles[:] = es_el[0, :]
            elet[:] = et_el[0, :]

        element_values = np.sqrt(np.sum(self.model_result.es**2, axis=1))
        self.model_result.max_value = np.max(element_values)

# --- Class for generating a report of model parameters and results
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
        '''
        self.add_text()
        self.add_text("Element topology:")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.edof, headers=["Node 1", "Node 2", "Node 3"], tablefmt="psql")
        ) '''

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
        self.add_text("Nodal temps and flows (a and r):")
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
        '''
        self.add_text()
        self.add_text("Element temps (ed):")
        self.add_text()
        self.add_text(tab.tabulate(np.asarray(np.hstack((self.model_params.elem.reshape(-1, 1), self.model_result.ed))),
            headers=["Element", "Phi_1 [m]", "Phi_2 [m]", "Phi_3 [m]"],
            numalign="right",
            floatfmt=(".0f", ".4f", ".4f", ".4f"),
            tablefmt="psql",
            )) '''

        return self.report
    