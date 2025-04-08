# -*- coding: utf-8 -*-

import test3 as fm  # Import test3.py

if __name__ == "__main__":

    model_params = fm.ModelParams()  # Initiate class ModelParams
    model_visualization = fm.ModelVisualization()  # Initiate class ModelVisualization
    
    # Uncomment the following lines if you want to load or save parameters
    # model_params.load("test_json")  # Load file
    # model_params.save("test_json")  # Save file
    
    solver = fm.ModelSolver(model_params, model_visualization)  # Initiate class ModelSolver
    solver.execute()  # Execute ModelSolver
    
    visualize = fm.ModelVisualization(model_params, model_visualization)  # Initiate class ModelVisualization
    print(visualize)  # Print results


'''# -*- coding: utf-8 -*-

import flowmodel as fm # Import flowmodel.py

if __name__ == "__main__":

    model_params = fm.ModelParams() # Initiate class ModelParams
    model_result = fm.ModelResult() # Initiate class ModelResult
    
    #model_params.load("test_json") # Load file
    #model_params.save("test_json") # Save file
    
    solver = fm.ModelSolver(model_params, model_result) # Initiate class ModelSolver
    solver.execute() # Execute ModelSolver

    report = fm.ModelReport(model_params, model_result) # Initiate class ModelReport
    print(report) # Print results'''