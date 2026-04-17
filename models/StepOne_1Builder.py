from gurobipy import GRB
from models.OptimizationClasses import LP_InputData
from data.data import prob_scenarios, prices_DA, prices_Bal, wind_production, P_max

class Step1point1Builder:
    """Builder for Offering Strategy Under a One-price Balancing Scheme constraints and coefficients for 24 hour"""
    
    def __init__(self, model_name="Task1_1_OnePrice"):
        self.P_max = P_max
        self.num_hours = 24
        self.num_scenarios = len(prob_scenarios)
        self.model_name = model_name
        
        # Build variable names once
        self._build_names()
    
    def _build_names(self):
        """Build all variable and constraint names"""
        self.p_DA_keys = [f"p_DA_{t+1}" for t in range(self.num_hours)]
        self.delta_keys = []
        self.bal_constraints = []
        
        for t in range(self.num_hours):
            for w in range(self.num_scenarios):
                self.delta_keys.append(f"delta_{t+1}_{w+1}")
                self.bal_constraints.append(f"bal_{t+1}_{w+1}")
                
        self.variables = self.p_DA_keys + self.delta_keys
        self.u_keys = [f"u_p_DA_{t+1}" for t in range(self.num_hours)]
    
    def build_objective_coefficients(self):
        """Build objective coefficients from data"""
        obj_coeff = {}
        
        # Day-Ahead: E[Price_DA] * p_DA
        for t in range(self.num_hours):
            exp_price_da = sum(1/1600 * prices_DA[t][w] for w in range(self.num_scenarios))
            obj_coeff[f"p_DA_{t+1}"] = exp_price_da
            
        # Balancing: pi_w * price_B * (delta_pos - delta_neg)
        for t in range(self.num_hours):
            for w in range(self.num_scenarios):
                obj_coeff[f"delta_{t+1}_{w+1}"] = 1/1600 * prices_Bal[t][w]
                
        return obj_coeff
    
    def build_constraint_coefficients(self):
        """Build constraint coefficients"""
        coeff = {}
        
        # capacity: p_DA <= P_max
        for t in range(self.num_hours):
            coeff[self.u_keys[t]] = {f"p_DA_{t+1}": 1}
            
        # balancing: p_DA + delta = p_real
        for t in range(self.num_hours):
            for w in range(self.num_scenarios):
                c_name = f"bal_{t+1}_{w+1}"
                coeff[c_name] = {
                    f"p_DA_{t+1}": 1,
                    f"delta_{t+1}_{w+1}": 1
                }
        return coeff
    
    def build_constraint_rhs(self):
        """Build right-hand side values"""
        rhs = {}
        for t in range(self.num_hours):
            rhs[self.u_keys[t]] = self.P_max
            for w in range(self.num_scenarios):
                rhs[f"bal_{t+1}_{w+1}"] = wind_production[t][w]
        return rhs
    
    def build_constraint_sense(self):
        """Build constraint senses"""
        sense = {k: GRB.LESS_EQUAL for k in self.u_keys}
        sense.update({k: GRB.EQUAL for k in self.bal_constraints})
        return sense
    
    # def _one_hot_vector(self, idx, sign=1):
    #     """Helper: create one-hot coefficient vector"""
    #     coeff = {v: 0 for v in self.variables}
    #     coeff[self.variables[idx]] = sign
    #     return coeff
    
    def build_input_data(self):
        """Build complete LP_InputData object"""
        return LP_InputData(
            VARIABLES=self.variables,
            CONSTRAINTS=self.u_keys + self.bal_constraints,
            objective_coeff=self.build_objective_coefficients(),
            constraints_coeff=self.build_constraint_coefficients(),
            constraints_rhs=self.build_constraint_rhs(),
            constraints_sense=self.build_constraint_sense(),
            objective_sense=GRB.MAXIMIZE,
            model_name=self.model_name
        )
