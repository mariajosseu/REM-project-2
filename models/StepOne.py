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
            [f"delta_{i+1}_{w+1}" for i in range(self.num_hours) for w in range(self.num_scenarios)])
        
        self.u_keys = [f"u_p_DA_{t+1}" for t in range(self.num_hours)]
        self.l_keys = [f"l_p_DA_{i+1}" for i in range(self.num_hours)]
        self.balance = [f"bal_{t+1}_{w+1}" for t in range(self.num_hours) for w in range(self.num_scenarios)]
        
    
    def build_objective_coefficients(self):
        """Build objective coefficients from data.

        Imbalance contribution changes sign with system state:
        - helps system -> earning (positive coefficient)
        - hurts system -> cost (negative coefficient)
        """
        obj_coeff = {}
        for hour in range(self.num_hours):
            obj_coeff[f"p_DA_{hour+1}"] = sum(
                                        float(s.prices[hour] / self.num_scenarios)
                                        for s in self.scenario_list
                                    )
            count = 0
            for w in self.scenario_list:
                da_price = float(w.prices[hour])
                prob = 1 / self.num_scenarios

                if int(w.imbalance[hour]) == 1:
                    balancing_price = 1.25 * da_price   # deficit
                else:
                    balancing_price = 0.85 * da_price   # surplus (excess)
                obj_coeff[f"delta_{hour+1}_{count+1}"] = prob * balancing_price
                count += 1
        for hour in range(self.num_hours):
            avg_da = sum(s.prices[hour] / self.num_scenarios for s in self.scenario_list)

            avg_bp = 0
            for w in self.scenario_list:
                da_price = float(w.prices[hour])
                if int(w.imbalance[hour]) == 1:
                    bp = 1.25 * da_price
                else:
                    bp = 0.85 * da_price
                avg_bp += bp / self.num_scenarios

            print(hour + 1, "E[DA] =", avg_da, "E[BP] =", avg_bp, "E[DA-BP] =", avg_da - avg_bp)
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
class DayAheadTwoPriceBuilder:
    """Builder for Offering Strategy Under a One-price Balancing Scheme constraints and coefficients for 24 hour"""
    
    def __init__(self, scenario_list=None, model_name="Day-Ahead Two-Price Model"):
        self.P_max = 500 # wind farm installed capacity [MW]
        self.num_hours = 24
        self.wind_scenarios = 20
        self.price_scenarios = 20
        self.imbalance_scenarios = 4
        if scenario_list is None:
            scenario_list = list(scenarios.values())[:200]
        self.scenario_list = list(scenario_list)
        self.num_scenarios = len(self.scenario_list)
        self.model_name = model_name

        # Build variable names once
        self._build_names()
        
    
    def _build_names(self):
        """Build all variable and constraint names"""
        self.variables = (
            [f"p_DA_{i+1}" for i in range(self.num_hours)] +
            [f"delta_{i+1}_{w+1}" for i in range(self.num_hours) for w in range(self.num_scenarios)]+
            [f"delta_up_{i+1}_{w+1}" for i in range(self.num_hours) for w in range(self.num_scenarios)]+
            [f"delta_down_{i+1}_{w+1}" for i in range(self.num_hours) for w in range(self.num_scenarios)]
        )
        
        self.u_keys = [f"u_p_DA_{t+1}" for t in range(self.num_hours)]
        self.l_keys = [f"l_p_DA_{i+1}" for i in range(self.num_hours)]
        self.balance = [f"bal_{t+1}_{w+1}" for t in range(self.num_hours) for w in range(self.num_scenarios)]
        self.delta_def = [f"delta_def_{t+1}_{w+1}" for t in range(self.num_hours) for w in range(self.num_scenarios)]
        self.delta_up_nonneg = [f"delta_up_nonneg_{t+1}_{w+1}" for t in range(self.num_hours) for w in range(self.num_scenarios)]
        self.delta_down_nonneg = [f"delta_down_nonneg_{t+1}_{w+1}" for t in range(self.num_hours) for w in range(self.num_scenarios)]
        
    def build_objective_coefficients(self):
        obj_coeff = {}

        for hour in range(self.num_hours):

            obj_coeff[f"p_DA_{hour+1}"] = sum(
                float(s.prices[hour] / self.num_scenarios)
                for s in self.scenario_list
            )

            for count, w in enumerate(self.scenario_list):
                da_price = float(w.prices[hour])
                prob = 1 / self.num_scenarios
                exp_price_da = prob * da_price

                obj_coeff[f"delta_{hour+1}_{count+1}"] = 0.0

                if int(w.imbalance[hour]) == 1:
                    # System deficit: BP = 1.25 * DA
                    # Upward deviation helps the system -> settled at DA
                    # Downward deviation hurts the system -> settled at BP
                    obj_coeff[f"delta_up_{hour+1}_{count+1}"] = exp_price_da
                    obj_coeff[f"delta_down_{hour+1}_{count+1}"] = -1.25 * exp_price_da

                else:
                    # System surplus: BP = 0.85 * DA
                    # Upward deviation hurts the system -> settled at BP
                    # Downward deviation helps the system -> settled at DA
                    obj_coeff[f"delta_up_{hour+1}_{count+1}"] = 0.85 * exp_price_da
                    obj_coeff[f"delta_down_{hour+1}_{count+1}"] = -exp_price_da

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
        # delta - delta_up + delta_down = 0
        for hour in range(self.num_hours):
            for w in range(self.num_scenarios):
                c_name = f"delta_def_{hour+1}_{w+1}"
                coeff[c_name] = {
                    f"delta_{hour+1}_{w+1}": 1,
                    f"delta_up_{hour+1}_{w+1}": -1,
                    f"delta_down_{hour+1}_{w+1}": 1
                }
        # -delta_up <= 0 and -delta_down <= 0;
        for hour in range(self.num_hours):
            for w in range(self.num_scenarios):
                c_name_up = f"delta_up_nonneg_{hour+1}_{w+1}"
                c_name_down = f"delta_down_nonneg_{hour+1}_{w+1}"
                coeff[c_name_up] = {f"delta_up_{hour+1}_{w+1}": -1}
                coeff[c_name_down] = {f"delta_down_{hour+1}_{w+1}": -1}
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
        
        # delta - delta_up + delta_down = 0
        for hour in range(self.num_hours):
            for w in range(self.num_scenarios):
                rhs[f"delta_def_{hour+1}_{w+1}"] = 0
        
        # -delta_up <= 0 and -delta_down <= 0;
        for hour in range(self.num_hours):
            for w in range(self.num_scenarios):
                rhs[f"delta_up_nonneg_{hour+1}_{w+1}"] = 0
                rhs[f"delta_down_nonneg_{hour+1}_{w+1}"] = 0
        return rhs
    
    def build_constraint_sense(self):
        """Build constraint senses"""
        sense = {}
        for key in self.u_keys + self.l_keys + self.delta_up_nonneg + self.delta_down_nonneg:
            sense[key] = GRB.LESS_EQUAL
        for key in self.balance + self.delta_def:
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
            CONSTRAINTS=self.u_keys + self.l_keys + self.balance + self.delta_def + self.delta_up_nonneg + self.delta_down_nonneg,
            objective_coeff=self.build_objective_coefficients(),
            constraints_coeff=self.build_constraint_coefficients(),
            constraints_rhs=self.build_constraint_rhs(),
            constraints_sense=self.build_constraint_sense(),
            objective_sense=GRB.MAXIMIZE,
            model_name=self.model_name
        )

class RiskAverseBuilder:
    """Builder for Offering Strategy Under a One-price Balancing Scheme constraints and coefficients for 24 hour"""
    def __init__(self, model_name="Day-Ahead One-Price Model"):
        self.P_max = 500 # wind farm installed capacity [MW]
        self.num_hours = 24
        self.wind_scenarios = 20
        self.price_scenarios = 20
        self.imbalance_scenarios = 4
        self.num_scenarios = 200
        self.scenario_list = list(scenarios.values())[:200]
        self.alpha = 0.90
        self.beta = 0.95
        self.model_name = model_name

        # Build variable names once
        self._build_names()
        
    
    def _build_names(self):
        """Build all variable and constraint names"""
        self.variables = (
            [f"p_DA_{i+1}" for i in range(self.num_hours)] +
            [f"delta_{i+1}_{w+1}" for i in range(self.num_hours) for w in range(self.num_scenarios)]+
            [f"VaR"]+
            [f"eta_{w+1}" for w in range(self.num_scenarios) ]
            )
        
        self.u_keys = [f"u_p_DA_{t+1}" for t in range(self.num_hours)] 
        self.l_keys = [f"l_p_DA_{i+1}" for i in range(self.num_hours)]
        + [f"l_eta_{w+1}" for w in range(self.num_scenarios)]
        self.balance = [f"bal_{t+1}_{w+1}" for t in range(self.num_hours) for w in range(self.num_scenarios)]
        self.max_eta = [f"max_eta_{w+1}" for w in range(self.num_scenarios)]
        
    
    def build_objective_coefficients(self):
        """Build objective coefficients from data.

        Imbalance contribution changes sign with system state:
        - helps system -> earning (positive coefficient)
        - hurts system -> cost (negative coefficient)
        """
        obj_coeff = {}
        for hour in range(self.num_hours):
            obj_coeff[f"p_DA_{hour+1}"] = sum(
                                        float(s.prices[hour] / self.num_scenarios)
                                        for s in self.scenario_list
                                    )
            count = 0
            for w in self.scenario_list:
                da_price = float(w.prices[hour])
                prob = 1 / self.num_scenarios

                if int(w.imbalance[hour]) == 1:
                    balancing_price = 1.25 * da_price   # deficit
                else:
                    balancing_price = 0.85 * da_price   # surplus (excess)
                obj_coeff[f"delta_{hour+1}_{count+1}"] = prob * balancing_price
                count += 1
        for hour in range(self.num_hours):
            avg_da = sum(s.prices[hour] / self.num_scenarios for s in self.scenario_list)

            avg_bp = 0
            for w in self.scenario_list:
                da_price = float(w.prices[hour])
                if int(w.imbalance[hour]) == 1:
                    bp = 1.25 * da_price
                else:
                    bp = 0.85 * da_price
                avg_bp += bp / self.num_scenarios

            print(hour + 1, "E[DA] =", avg_da, "E[BP] =", avg_bp, "E[DA-BP] =", avg_da - avg_bp)
        for w in self.scenario_list:
            obj_coeff[f"eta_{w+1}"] = -self.beta * (1 / self.num_scenarios) * (1/(1-self.alpha))
        obj_coeff["VaR"] = self.beta
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
        for w in range(self.num_scenarios):
            coeff[self.l_keys[self.num_hours + w]] = self._one_hot_vector(self.num_hours + w, sign=-1)
            
        # balancing: p_DA + delta = p_real
        for hour in range(self.num_hours):
            for w in range(self.num_scenarios):
                c_name = f"bal_{hour+1}_{w+1}"
                coeff[c_name] = {
                    f"p_DA_{hour+1}": 1,
                    f"delta_{hour+1}_{w+1}": 1
                }
        # -Profit + VaR - eta <= 0
        for w in range(self.num_scenarios):
            for hour in range(self.num_hours):
                da_price = float(self.scenario_list[w].prices[hour])

                if int(self.scenario_list[w].imbalance[hour]) == 1:
                    bp = 1.25 * da_price
                else:
                    bp = 0.85 * da_price
                coeff[f"max_eta_{w+1}"] = {
                    f"p_DA_{hour+1}": -da_price,
                    f"delta_{hour+1}_{w+1}": -bp ,
                    f"VaR": 1,
                    f"eta_{w+1}": -1}
        return coeff
    
    def build_constraint_rhs(self):
        """Build right-hand side values"""
        rhs = {}

        # upper bounds
        for t in range(self.num_hours):
            rhs[self.u_keys[t]] = self.P_max

        # lower bounds
        for t in range(self.num_hours+ self.num_scenarios):
            rhs[self.l_keys[t]] = 0

        # balance constraints
        for t in range(self.num_hours):
            for w in range(self.num_scenarios):
                rhs[f"bal_{t+1}_{w+1}"] = float(self.scenario_list[w].wind[t])
        # -Profit + VaR - eta <= 0
        for w in range(self.num_scenarios):
            rhs[f"max_eta_{w+1}"] = 0

        return rhs
    
    def build_constraint_sense(self):
        """Build constraint senses"""
        sense = {}
        for key in self.u_keys + self.l_keys:
            sense[key] = GRB.LESS_EQUAL
        for key in self.balance:
            sense[key] = GRB.EQUAL
        for key in self.max_eta:
            sense[key] = GRB.LESS_EQUAL
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
