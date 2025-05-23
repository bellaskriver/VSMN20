# -*- coding: utf-8 -*-
import json
import pyvtk as vtk

import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu

import matplotlib.pylab as plt
import tabulate as tab
import numpy as np

class ModelParams:
    """Class defining parametric model properties"""
    def __init__(self):

        # Version tracking
        self.version = 1

        # Geometric parameters
        self.w = 100.0 # Width of domain
        self.h = 10.0 # Height of domain
        self.d = 5.0 # Depth of barrier
        self.t = 0.5 # Thickness of barrier
        self.ep = [self.t, int(2)] # Element properties

        # Material properties
        self.kx = 20.0 # Permeability in x-direction
        self.ky = 20.0 # Permeability in y-direction
    
        # Mesh control
        self.el_size_factor = 0.5 # Elements size in mesh

        # Boundary conditions and loads
        self.bc_markers = {
            "left_bc": 10, # Marker for left boundary
            "right_bc": 20 # Marker for right boundary
        }

        self.bc_values = {
            "left_bc": 60.0, # Value for left boundary
            "right_bc": 0.0  # Value for right boundary
        }

    def geometry(self):
        """Create and return a geometry instance based on defined parameters"""

        # Use shorter variable names for readability
        w = self.w
        h = self.h
        t = self.t
        d = self.d

        # Create a geometry object
        g = cfg.Geometry()

        # Define points for the geometry
        g.point([0, 0]) # Point 0: Bottom left corner
        g.point([w, 0]) # Point 1: Bottom right corner
        g.point([w, h]) # Point 2: Top right corner
        g.point([w/2+t/2, h]) # Point 3: Top right corner of barrier
        g.point([w/2+t/2, h-d]) # Point 4: Bottom right corner of barrier
        g.point([w/2-t/2, h-d]) # Point 5: Bottom left corner of barrier
        g.point([w/2-t/2, h]) # Point 6: Top left corner of barrier
        g.point([0, h]) # Point 7: Top left corner

        # Define splines connecting the points
        g.spline([0, 1])                       
        g.spline([1, 2])
        g.spline([2, 3], marker=self.bc_markers["right_bc"])
        g.spline([3, 4])          
        g.spline([4, 5]) 
        g.spline([5, 6])
        g.spline([6, 7], marker=self.bc_markers["left_bc"])                    
        g.spline([7, 0])               

        # Define the surface using the spline indices
        g.surface([0, 1, 2, 3, 4, 5, 6, 7])
  
        return g
    
    def save(self, filename):
        """Save input to file."""

        model_params = {} # Create a dictionary to store model parameters
        model_params["version"] = self.version
        model_params["t"] = self.t
        model_params["ep"] = self.ep
        model_params["w"] = self.w
        model_params["h"] = self.h
        model_params["d"] = self.d
        model_params["kx"] = self.kx
        model_params["ky"] = self.ky
        model_params["el_size_factor"] = self.el_size_factor
        model_params["bc_markers"] = self.bc_markers
        model_params["bc_values"] = self.bc_values

        # Write the model parameters to a JSON file
        ofile = open(filename, "w")
        json.dump(model_params, ofile, sort_keys = True, indent = 4)
        ofile.close()

    def load(self, filename):
        """Read input from file."""

        # Read the model parameters from a JSON file
        ifile = open(filename, "r")
        model_params = json.load(ifile)
        ifile.close()

        self.version = model_params["version"]
        self.t = model_params["t"]
        self.ep = model_params["ep"]
        self.w = model_params["w"]
        self.h = model_params["h"]
        self.d = model_params["d"]
        self.kx = model_params["kx"]
        self.ky = model_params["ky"]
        self.el_size_factor = model_params["el_size_factor"]
        self.bc_markers = model_params["bc_markers"]
        self.bc_values = model_params["bc_values"]
    
class ModelResult:
    """Class for storing results from calculations."""

    def __init__(self):

        # Initialize attributes for mesh and geometry
        self.loads = None
        self.bcs = None
        self.edof = None
        self.coords = None
        self.dofs = None
        self.bdofs = None
        self.boundary_elements = None
        self.geometry = None

        # Initialize attributes for results
        self.a = None
        self.r = None
        self.ed = None
        self.es = None
        self.et = None

        self.flow = None
        self.pressure = None
        self.gradient = None

        self.max_nodal_flow = None
        self.max_nodal_pressure = None
        self.max_element_flow = None
        self.max_elemetn_pressure = None
        self.max_element_gradient = None

class ModelVisualization:
    """Class for visualizing model geometry, mesh, and results"""

    def __init__(self, model_params, model_result):
        """Constructor"""

        # Store references to model parameters and results
        self.model_params = model_params
        self.model_result = model_result

        # Store references to visualization windows
        self.geom_fig = None
        self.mesh_fig = None
        self.node_value_fig = None
        self.element_value_fig = None

    def show_geometry(self):
        """Display model geometry"""

        # Create a new figure
        cfv.figure()
        cfv.clf()

        # Draw Geometry§
        cfv.draw_geometry(
            geometry = self.model_params.geometry(),
            draw_points=True,
            label_points=True,
            label_curves=True,
            title="Model Geometry"
        )

        cfv.show_and_wait()

    def show_mesh(self):
        """Display Finite Element Mesh"""

        # Create a new figure
        cfv.figure()
        cfv.clf()

        # Draw Mesh
        cfv.draw_mesh(
            coords=self.model_result.coords,
            edof=self.model_result.edof,
            dofs_per_node=self.model_result.dofs_per_node,
            el_type=self.model_result.el_type,
            filled=True,
            title="Finite Element Mesh"
        )
        
        cfv.show_and_wait()

    def show_nodal_values(self):
        """Display Nodal Pressure"""

        # Create a new figure
        cfv.figure()
        cfv.clf()

        # Draw Nodal Pressure
        cfv.draw_nodal_values(
            self.model_result.a,
            coords=self.model_result.coords,
            edof=self.model_result.edof,
            title="Nodal Pressure"
        )

        cfv.show_and_wait()

    def show_element_values(self):
        """Display Element Flows"""
        
        # Create a new figure
        cfv.figure()
        cfv.clf()

        # Draw element flows
        cfv.draw_element_values(
            self.model_result.flow,
            coords=self.model_result.coords,
            edof=self.model_result.edof,
            dofs_per_node=self.model_result.dofs_per_node,
            el_type=self.model_result.el_type,
            title="Element Flows"
        )

        cfv.show_and_wait()

    def wait(self):
        """Wait for user to close all visualization windows"""

        cfv.show_and_wait()

class ModelSolver:
    """Class for solving the finite element model"""

    def __init__(self, model_params, model_result):
        self.model_params = model_params
        self.model_result = model_result

    def execute(self):
        """Perform mesh generation and finite element computation"""

        # Create shorter references to input variables
        ep = self.model_params.ep
        D = np.array([[self.model_params.kx, 0], [0, self.model_params.ky]]) # Permeability matrix

        # Get geometry and store it
        geometry = self.model_params.geometry()
        self.model_result.geometry = geometry

        # Set up mesh generation
        el_type = 3
        dofs_per_node = 1

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
         
        # Create global stiffness matrix and load vector
        n_dofs = np.max(dofs)
        K = np.zeros((n_dofs, n_dofs))
        f = np.zeros((n_dofs, 1))

        # Global stiffness matrix
        nDofs = np.size(dofs) # Number of global degrees of freedom
        ex, ey = cfc.coordxtr(edof, coords, dofs) # Extract coordinates of elements
        K = np.zeros([nDofs, nDofs]) # Global stiffness matrix
        n_el = edof.shape[0] # Number of elements
        ep = np.tile(self.model_params.ep, (n_el, 1)).astype(object) # Element properties

        # Assemble global stiffness matrix
        for i, (eltopo, elx, ely) in enumerate(zip(edof, ex, ey)):
            thickness = float(ep[i][0])
            integration_rule = int(ep[i][1])
            el_ep = [thickness, integration_rule]
            Ke = cfc.flw2i4e(elx, ely, el_ep, D)
            cfc.assem(eltopo, K, Ke)

        # Global load vector
        f = np.zeros([nDofs, 1])

        # Boundary conditions
        bc = np.array([], int)
        bcVal = np.array([], int)

        # Apply boundary conditions
        for name, marker in self.model_params.bc_markers.items():
            value = self.model_params.bc_values.get(name, 0.0)
            bc, bcVal = cfu.applybc(bdofs, bc, bcVal, marker, value)

        # Solve the system of equations
        a, r = cfc.solveq(K, f, bc, bcVal)
        ed = cfc.extractEldisp(edof, a)

        # Calculate element flows
        flow = [] # List to store flow values
        gradient = [] # List to store gradient values

        for i in range(edof.shape[0]):
            el_ep = [float(ep[i][0]), int(ep[i][1])]
            es, et, eci = cfc.flw2i4s(ex[i, :], ey[i, :], el_ep, D, ed[i, :])
            flow.append(np.sqrt(es[0, 0]**2 + es[0, 1]**2))
            gradient.append(np.sqrt(et[0, 0]**2 + et[0, 1]**2))

        # Maximal flow, pressure, gradient for nodes and elements
        max_nodal_pressure = np.max(np.abs(a))
        max_nodal_flow = np.max(np.abs(r))
        max_element_pressure = np.max(np.abs(ed))
        max_element_flow = np.max(np.abs(flow))
        max_element_gradient = np.max(np.abs(gradient))

        # Store results in model_result and model_params
        self.model_result.loads = list(zip(bc, bcVal)) 
        self.model_result.bcs = list(zip(bc, bcVal))
        self.model_result.edof = edof
        self.model_params.coord = coords
        self.model_params.dof = dofs
        self.model_params.elem = np.arange(edof.shape[0])

        self.model_result.a = a
        self.model_result.r = r
        self.model_result.ed = ed
        self.model_result.es = es
        self.model_result.et = et

        self.model_result.flow = flow
        self.model_result.gradient = gradient

        self.model_result.max_nodal_flow = max_nodal_flow
        self.model_result.max_nodal_pressure = max_nodal_pressure
        self.model_result.max_element_flow = max_element_flow
        self.model_result.max_element_pressure = max_element_pressure
        self.model_result.max_element_gradient = max_element_gradient

    def run_parameter_study(self):
        """Run a parameter study by varying the barrier depth"""

        # Parameters to vary
        d_values = np.linspace(1.0, 9.0, 9)
        max_flow_values = []

        # Run simulation for each value
        for d in d_values:
            print(f"Simulating with barrier depth d = {d:.2f}...")
            # Create model with current parameter
            model_params = ModelParams()
            model_params.d = d  # Set current barrier depth

            # Other parameters remain constant
            model_params.w = 100.0
            model_params.h = 10.0
            model_params.t = 0.5
            model_params.kx = 20.0
            model_params.ky = 20.0

            # Create result storage and solver
            model_result = ModelResult()
            model_solver = ModelSolver(model_params, model_result)

            # Run the simulation
            model_solver.execute()

            # Compute the norm of the flow vector for each element
            flow_norms = np.linalg.norm(model_result.es, axis=1)

            # Store the maximum flow for this configuration
            max_flow_values.append(np.max(flow_norms))
            print(f"Maximum flow value: {np.max(flow_norms):.4f}")


        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(d_values, max_flow_values, 'o-', linewidth=2)
        plt.grid(True)
        plt.xlabel('Barrier Depth (d)')
        plt.ylabel('Maximum Flow')
        plt.title('Parameter Study: Effect of Barrier Depth on Maximum Flow')
        plt.savefig('parameter_study.png')
        plt.show()

        # Return results for further analysis
        return d_values, max_flow_values
         
    def export_vtk(self, filename):
        """Export node‐pressures and cell‐flows to a VTK file."""

        print(f"Exporting results to {filename!r}...")

        # Import geometry, mesh and flow
        points = self.model_result.coords.tolist()
        polygons = (self.model_result.edof[:, 1:] - 1).tolist()
        point_data = vtk.PointData(
            vtk.Scalars(self.model_result.a.tolist(), name="pressure")
        )
        cell_data = vtk.CellData(
            vtk.Scalars(self.model_result.flow, name="flow")
        )

        # Create VTK structure
        structure = vtk.PolyData(points=points, polygons=polygons)
        vtk_data  = vtk.VtkData(structure, point_data, cell_data)
        vtk_data.tofile(filename, "ascii")

        print("VTK export complete.")

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
        self.add_text(
            tab.tabulate([
            ["t", self.model_params.t],
            ["w", self.model_params.w],
            ["h", self.model_params.h],
            ["d", self.model_params.d],
            ["kx", self.model_params.kx],
            ["ky", self.model_params.ky],
            ["Element size", self.model_params.el_size_factor],
            ["Left boundary", self.model_params.bc_values.get("left_bc", "N/A")],
            ["Right boundary", self.model_params.bc_values.get("right_bc", "N/A")]
            ],
            headers=["Parameter", "Value"],
            numalign="right",
            floatfmt=".1f",
            tablefmt="psql"
            )
        )
       
        self.add_text()
        self.add_text("-------------- Model results --------------------------------")
        self.add_text()
        self.add_text(
            tab.tabulate(
            [[
                self.model_result.max_nodal_pressure,
                self.model_result.max_nodal_flow,
                self.model_result.max_element_pressure,
                self.model_result.max_element_flow,
                self.model_result.max_element_gradient
            ]],
            headers=[
                "Max Nodal Pressure",
                "Max Nodal Flow",
                "Max Element Pressure",
                "Max Element Flow",
                "Max Element Gradient"
            ],
            numalign="right",
            floatfmt=".4f",
            tablefmt="psql"
            )
        )

        return self.report