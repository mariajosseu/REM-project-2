#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gurobipy import GRB
import numpy as np


from models.OptimizationClasses import LP_InputData
from data.data import scenarios

#%%

class DayAheadOnePriceBuilder:
    """Builder for Offering Strategy Under a One-price Balancing Scheme constraints and coefficients for 24 hour"""
    
    def __init__(self, scenario_list, model_name="Day-Ahead One-Price Model"):
        self.P_max = 500 # wind farm installed capacity [MW]
        self.num_hours = 24
        self.wind_scenarios = 20
        self.price_scenarios = 20
        self.imbalance_scenarios = 4
        self.scenario_list = list(scenario_list)
        self.num_scenarios = len(self.scenario_list)
        self.model_name = model_name

        # Build variable names once
        self._build_names()
        
    
    def _build_names(self):
        """Build all variable and constraint names"""
        self.variables = (
            [f"p_DA_{i+1}" for i in range(self.num_hours)] +
            [f"delta_{i+1}_{w+1}" for i in range(self.num_hours) for w in range(self.num_scenarios)]
        )
        
        self.u_keys = [f"u_p_DA_{t+1}" for t in range(self.num_hours)]
        self.l_keys = [f"l_p_DA_{i+1}" for i in range(self.num_hours)]
        self.balance = [f"bal_{t+1}_{w+1}" for t in range(self.num_hours) for w in range(self.num_scenarios)]
        
    
    def build_objective_coefficients(self):
        """Build objective coefficients from data"""
        obj_coeff = {}
        for hour in range(self.num_hours):
            count = 0
            #print(hour)
            for w in self.scenario_list:
                #print(w)
                exp_price_da = float(w.prices[hour]/ self.num_scenarios)
                #print(exp_price_da)
                obj_coeff[f"p_DA_{hour+1}"] = sum(float(s.prices[hour] / self.num_scenarios) for s in self.scenario_list)
                obj_coeff[f"delta_{hour+1}_{count+1}"] = -exp_price_da
                count += 1
        return obj_coeff
    
    def build_constraint_coefficients(self):
        """Build constraint coefficients"""
        coeff = {}
        # capacity: p_DA <= P_max
        # Upper bounds: x <= max
        for i in range(self.num_hours):
            coeff[self.u_keys[i]] = self._one_hot_vector(i, sign=1)
        # Lower bounds: -x <= 0
        for i in range(self.num_hours):
            coeff[self.l_keys[i]] = self._one_hot_vector(i, sign=-1)
            
        # balancing: p_DA + delta = p_real
        for hour in range(self.num_hours):
            for w in range(self.num_scenarios):
                c_name = f"bal_{hour+1}_{w+1}"
                coeff[c_name] = {
                    f"p_DA_{hour+1}": 1,
                    f"delta_{hour+1}_{w+1}": 1
                }
        return coeff
    
    def build_constraint_rhs(self):
        """Build right-hand side values"""
        rhs = {}

        # upper bounds
        for t in range(self.num_hours):
            rhs[self.u_keys[t]] = self.P_max

        # lower bounds
        for t in range(self.num_hours):
            rhs[self.l_keys[t]] = 0

        # balance constraints
        for t in range(self.num_hours):
            for w in range(self.num_scenarios):
                rhs[f"bal_{t+1}_{w+1}"] = float(self.scenario_list[w].wind[t])
        return rhs
    
    def build_constraint_sense(self):
        """Build constraint senses"""
        sense = {}
        for key in self.u_keys + self.l_keys:
            sense[key] = GRB.LESS_EQUAL
        for key in self.balance:
            sense[key] = GRB.EQUAL
        return sense
    
    def _one_hot_vector(self, idx, sign=1):
        """Helper: create one-hot coefficient vector"""
        coeff = {v: 0 for v in self.variables}
        coeff[self.variables[idx]] = sign
        return coeff
    
    def build_input_data(self):
        """Build complete LP_InputData object"""
        return LP_InputData(
            VARIABLES=self.variables,
            CONSTRAINTS=self.u_keys + self.l_keys + self.balance,
            objective_coeff=self.build_objective_coefficients(),
            constraints_coeff=self.build_constraint_coefficients(),
            constraints_rhs=self.build_constraint_rhs(),
            constraints_sense=self.build_constraint_sense(),
            objective_sense=GRB.MAXIMIZE,
            model_name=self.model_name
        )


