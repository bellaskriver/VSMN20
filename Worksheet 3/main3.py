# -*- coding: utf-8 -*-

import test3 as fm  # Import test3.py

if __name__ == "__main__":

    # --- Initialization ---
    model_params = fm.ModelParams() # Initiate class ModelParams
    model_result = fm.ModelResult() # Initiate class ModelResult
    model_solver = fm.ModelSolver(model_params, model_result) # Initiate class ModelSolver
    
    # --- Save and Load ---
    #model_params.load("test_json1") # Load file
    model_params.save("test_json1") # Save file
    
    # --- Calculations ---
    model_solver.execute() # Execute ModelSolver
    model_solver.run_parameter_study() # Run parameter study

    # --- Results ---
    report = fm.ModelReport(model_params, model_result) # Initiate class ModelReport
    print(str(report)) # Print results

    # --- Visualization ---
    model_visualization = fm.ModelVisualization(model_params, model_result) # Initiate class ModelVisualization
    model_visualization.show_mesh()
    model_visualization.show_geometry()
    model_visualization.show_nodal_values()
    model_visualization.show_element_values()
    model_visualization.wait()
    