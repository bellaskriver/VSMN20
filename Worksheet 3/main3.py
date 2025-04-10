# -*- coding: utf-8 -*-

import test3 as fm  # Import test3.py

if __name__ == "__main__":

    model_params = fm.ModelParams() # Initiate class ModelParams
    model_result = fm.ModelResult() # Initiate class ModelResult
    
    #model_params.load("test_json1") # Load file
    #model_params.save("test_json1") # Save file
    
    solver = fm.ModelSolver(model_params, model_result) # Initiate class ModelSolver
    solver.execute() # Execute ModelSolver

    report = fm.ModelReport(model_params, model_result) # Initiate class ModelReport
    print(str(report)) # Print results

    vis = fm.ModelVisualization(model_params, model_result) # Initiate class ModelVisualization
    print(vis) # Print visualization