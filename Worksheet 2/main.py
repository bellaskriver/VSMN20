# -*- coding: utf-8 -*-

import flowmodel as fm # Import flowmodel.py

if __name__ == "__main__":

    model_params = fm.ModelParams() # Initiate class ModelParams
    model_result = fm.ModelResult() # Initiate class ModelResult
    
    #model_params.load("test_json") # Load file
    model_params.save("test_json") # Save file
    
    solver = fm.ModelSolver(model_params, model_result) # Initiate class ModelSolver
    solver.execute() # Execute ModelSolver

    report = fm.ModelReport(model_params, model_result) # Initiate class ModelReport
    print(report) # Print results