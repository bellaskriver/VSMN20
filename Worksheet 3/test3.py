# -*- coding: utf-8 -*-

import json
import math
import sys

import calfem.core as cfc
import calfem.geometry as cfg  # For geometric modeling
import calfem.mesh as cfm      # For mesh generation
import calfem.vis_mpl as cfv   # For visualization
import calfem.utils as cfu     # Utility functions

import tabulate as tab
import numpy as np
import tabulate as tb

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

        # Material properties (example for groundwater problem)

        self.kx = 20.0  # Permeability in x-direction
        self.ky = 20.0  # Permeability in y-direction
        self.D = np.array([[self.kx, 0], [0, self.ky]]) #bella
    
        # Mesh control

        self.el_size_factor = 0.5  # Controls element size in mesh generation

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
            "bottom_load": 30,  # Marker for bottom boundary load
            "top_load": 40      # Marker for top boundary load
        }

        self.load_values = {
            "bottom_load": -400.0,  # Load value for bottom boundary
            "top_load": 0.0        # Load value for top boundary
        }

    def geometry(self):
        """Create and return a geometry instance based on defined parameters"""

        # Create a geometry instance

        g = cfg.Geometry()

        # Use shorter variable names for readability

        w = self.w
        h = self.h
        t = self.t
        d = self.d

        # Define points for the geometry
        # Point indices start at 0

        g.point([0, 0])          # Point 0: Bottom left corner
        g.point([w, 0])          # Point 1: Bottom right corner
        g.point([w, h])          # Point 2: Top right corner
        g.point([0, h])          # Point 3: Top left corner

        # Add points for the barrier

        g.point([w/2-t/2, h])    # Point 4: Top left of barrier
        g.point([w/2+t/2, h])    # Point 5: Top right of barrier
        g.point([w/2-t/2, h-d])  # Point 6: Bottom left of barrier
        g.point([w/2+t/2, h-d])  # Point 7: Bottom right of barrier

        # Define splines (lines) connecting the points
        # Use markers for boundaries with conditions

        g.spline([0, 1])                         # Bottom boundary
        g.spline([1, 2])                         # Right boundary, marker for fixed value
        g.spline([2, 5], marker=self.bc_markers["right_bc"])
        g.spline([5, 4])                         # Top of barrier
        g.spline([4, 3], marker=self.bc_markers["left_bc"]) # Left boundary, marker for fixed value
        g.spline([3, 0])
        g.spline([4, 6])                         # Left side of barrier
        g.spline([5, 7])                         # Right side of barrier
        g.spline([6, 7])                         # Bottom of barrier

        # Define the surface (domain) using the spline indices
        # Surface is defined by a list of spline indices that form a closed loop

        g.surface([0, 1, 2, 3, 4, 5, 6, 7, 8])

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

        coords, edof, dofs, bdofs, element_markers, boundary_elements = mesh.create()

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

        for elx, ely, eld, eles, elet in zip(ex, ey, ed, es, et): #bella
            es_el, et_el = cfc.flw2ts(elx, ely, D, eld)
            eles[:] = es_el[0, :]
            elet[:] = et_el[0, :]

        element_values = np.sqrt(np.sum(self.model_result.es**2, axis=1))
        self.model_result.max_value = np.max(element_values)

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