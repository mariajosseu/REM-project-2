import gurobipy as gp
from gurobipy import GRB
import re
from collections import defaultdict
from data.data import unit_cost_G, demand, load_curve, technical_data_G, technical_data_W
from models.AnalyzerClasses import SensitivityAnalyzer


class Results:
    """Container for optimization results"""
    def __init__(self):
        self.objective_value = None
        self.variables = {}
        self.optimal_duals = {}

class LP_InputData:

    def __init__(
        self, 
        VARIABLES: list[str],
        CONSTRAINTS: list[str],
        objective_coeff: dict[str, float],               # Coefficients in objective function
        constraints_coeff: dict[str, dict[str,float]],   # Linear coefficients of constraints
        constraints_rhs: dict[str, float],              # Right hand side coefficients of constraints
        constraints_sense: dict[str, int],              # Direction of constraints
        objective_sense: int,                           # Direction of op2timization
        model_name: str                                 # Name of model
    ):
        self.VARIABLES = VARIABLES
        self.CONSTRAINTS = CONSTRAINTS
        self.objective_coeff = objective_coeff
        self.constraints_coeff = constraints_coeff
        self.constraints_rhs = constraints_rhs
        self.constraints_sense = constraints_sense
        self.objective_sense = objective_sense
        self.model_name = model_name

class LP_OptimizationProblem():

    def __init__(self, builder): # initialize class
        self.builder = builder
        self.data = builder.build_input_data() # define data attributes
        self.results = Results() # initialize results container
        self._build_model() # build gurobi model
    
    def _build_variables(self):
        self.variables = {
            v: self.model.addVar(
                lb=-GRB.INFINITY if v.startswith('theta') else 0,
                name=f'{v}'
            )
            for v in self.data.VARIABLES
        }
    
    def _build_constraints(self):
        self.constraints = {c:
                self.model.addLConstr(
                        gp.quicksum(self.data.constraints_coeff[c][v] * self.variables[v] for v in self.data.constraints_coeff[c]),
                        self.data.constraints_sense[c],
                        self.data.constraints_rhs[c],
                        name = f'{c}'
                ) for c in self.data.CONSTRAINTS
        }

    def _build_objective_function(self):
        objective = gp.quicksum(self.data.objective_coeff[v] * self.variables[v] for v in self.data.VARIABLES)
        self.model.setObjective(objective, self.data.objective_sense)

    def _build_model(self):
        self.model = gp.Model(name=self.data.model_name)
        self._build_variables()
        self._build_objective_function()
        self._build_constraints()
        self.model.update()
    
    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        self.results.variables = {v.VarName:v.x for v in self.model.getVars()}
        self.results.optimal_duals = {f'{c.ConstrName}':c.Pi for c in self.model.getConstrs()}
    
    def get_results(self):
        """Return optimization results as a dictionary"""
        return {
            "objective_value": self.results.objective_value,
            "variables": self.results.variables,
            "optimal_duals": self.results.optimal_duals
        }

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print(f"optimization of {self.model.ModelName} was not successful")
    
    def display_results(self):
        print()
        print("="*70)
        print(" "*20 + "OPTIMIZATION RESULTS")
        print("="*70)
        print(f"Model: {self.data.model_name}")
        print(f"Objective Value: €{self.results.objective_value:,.2f}")
        print(f"Total operating cost: €{sum(self.data.objective_coeff[v] * self.results.variables[v] for v in self.data.VARIABLES if v.startswith('g'))}")

