from gurobipy import GRB
from models.OptimizationClasses import LP_InputData
from data.data import unit_cost_G, demand, load_curve, technical_data_G, technical_data_W

class Step1Builder:
    """Builder for Copperplate Model constraints and coefficients for one hour"""
    
    def __init__(self, num_gen, num_wind, num_demand, model_name):
        self.num_gen = num_gen
        self.num_wind = num_wind
        self.num_demand = num_demand
        self.num_total = num_gen + num_wind + num_demand
        self.num_hours = 1
        self.model_name = model_name
        
        # Build variable names once
        self._build_names()
    
    def _build_names(self):
        """Build all variable and constraint names"""
        self.variables = (
            [f"g{i+1}" for i in range(self.num_gen)] +
            [f"w{i+1}" for i in range(self.num_wind)] +
            [f"d{i+1}" for i in range(self.num_demand)]
        )
        self.u_keys = [f"u{i+1}" for i in range(self.num_total)]
        self.l_keys = [f"l{i+1}" for i in range(self.num_total)]
    
    def build_objective_coefficients(self):
        """Build objective coefficients from data"""
        obj_coeff = {}
        
        # Generator costs
        for i in range(self.num_gen):
            obj_coeff[f"g{i+1}"] = unit_cost_G["C_i"][i]
        
        # Wind costs (zero)
        for i in range(self.num_wind):
            obj_coeff[f"w{i+1}"] = 0
        
        # Demand bid prices (negative for revenue)
        for i in range(self.num_demand):
            obj_coeff[f"d{i+1}"] = -demand['Bid_price'][i]
        
        return obj_coeff
    
    def build_constraint_coefficients(self):
        """Build constraint coefficients"""
        coeff = {}
        
        # Upper bounds: x <= max
        for i in range(self.num_total):
            coeff[self.u_keys[i]] = self._one_hot_vector(i, sign=1)
        
        # Lower bounds: -x <= 0
        for i in range(self.num_total):
            coeff[self.l_keys[i]] = self._one_hot_vector(i, sign=-1)
        
        # Balance: gen + wind - demand = 0
        balance = {v: 1 for v in self.variables[:self.num_gen + self.num_wind]}
        balance.update({v: -1 for v in self.variables[self.num_gen + self.num_wind:]})
        coeff["balance"] = balance
        
        return coeff
    
    def build_constraint_rhs(self):
        """Build right-hand side values"""
        rhs = {}
        max_load = max(load_curve['load_MW'])
        
        # Generator upper bounds
        for i in range(self.num_gen):
            rhs[self.u_keys[i]] = technical_data_G["P_max"][i]
        
        # Wind upper bounds
        for i in range(self.num_wind):
            rhs[self.u_keys[self.num_gen + i]] = technical_data_W["P_max"][i]
        
        # Demand upper bounds
        for i in range(self.num_demand):
            rhs[self.u_keys[self.num_gen + self.num_wind + i]] = (
                demand['D_max %'][i] * max_load / 100
            )
        
        # Lower bounds (all zero)
        for key in self.l_keys:
            rhs[key] = 0
        
        # Balance
        rhs["balance"] = 0
        
        return rhs
    
    def build_constraint_sense(self):
        """Build constraint senses"""
        sense = {}
        for key in self.u_keys + self.l_keys:
            sense[key] = GRB.LESS_EQUAL
        sense["balance"] = GRB.EQUAL
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
            CONSTRAINTS=self.u_keys + self.l_keys + ["balance"],
            objective_coeff=self.build_objective_coefficients(),
            constraints_coeff=self.build_constraint_coefficients(),
            constraints_rhs=self.build_constraint_rhs(),
            constraints_sense=self.build_constraint_sense(),
            objective_sense=GRB.MINIMIZE,
            model_name=self.model_name
        )
