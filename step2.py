from pathlib import Path
from models.StepTwo import CVaRBuilder, ALSOXBuilder
from models.OptimizationClasses import LP_OptimizationProblem

def solve_cvar():
       
    builder_cvar = CVaRBuilder(alpha=0.90)
    problem_cvar = LP_OptimizationProblem(builder_cvar)
    problem_cvar.model.setParam('OutputFlag', 0) 
    problem_cvar.run()
    
    c_up_opt = problem_cvar.results.variables['c_up']
    print(f"Optimal Bid (CVaR): {c_up_opt:.2f} kW")
    return problem_cvar