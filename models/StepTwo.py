import numpy as np
from gurobipy import GRB
from models.OptimizationClasses import LP_InputData
# Assicurati di importare i dati dal tuo nuovo file (sostituisci l'import se necessario)
from data.data_step2 import generate_fcr_profiles

class CVaRBuilder:
    """Builder for FCR-D UP Offering Strategy using CVaR approximation"""
    
    def __init__(self, model_name="CVaR_Model"):
        # Upload data
        data = generate_fcr_profiles()
        self.f_up_in = data["f_up"][:, :100] # first 100 scenarios in sample (shape: 60x100)
        alpha=0.90
        self.num_minutes = 60
        self.num_scenarios = 100
        self.alpha = alpha
        self.model_name = model_name
        self._build_names()

    def _build_names(self):
        """Build all variable and constraint names"""
        # Variables: c_up, zeta (can be negative!), eta_w
        self.variables = (
            ["c_up"] + 
            ["zeta"] + 
            [f"eta_{w+1}" for w in range(self.num_scenarios)]
        )
        
        # Constraints
        self.def_excess = [f"def_excess_m{m+1}_w{w+1}" for m in range(self.num_minutes) for w in range(self.num_scenarios)]
        self.cvar_limit = ["cvar_limit_constraint"]

    def build_objective_coefficients(self):
        """Maximize c_up"""
        obj_coeff = {v: 0.0 for v in self.variables}
        obj_coeff["c_up"] = 1.0
        return obj_coeff

    def build_constraint_coefficients(self):
        """Build constraint coefficients"""
        coeff = {}
        
        # c_up - eta_w - zeta <= F_up[m, w]
        for m in range(self.num_minutes):
            for w in range(self.num_scenarios):
                c_name = f"def_excess_m{m+1}_w{w+1}"
                coeff[c_name] = {
                    "c_up": 1.0,
                    "zeta": -1.0,
                    f"eta_{w+1}": -1.0
                }
                
        # zeta + (1 / (1-alpha) * 1/|W| ) * sum(eta_w) <= 0
        cvar_multiplier = 1.0 / ((1.0 - self.alpha) * self.num_scenarios) # 1 / (0.1 * 100) = 0.1
        coeff["cvar_limit_constraint"] = {"zeta": 1.0}
        for w in range(self.num_scenarios):
            coeff["cvar_limit_constraint"][f"eta_{w+1}"] = cvar_multiplier
            
        return coeff
    
    def build_constraint_rhs(self):
        """Build right-hand side values"""
        rhs = {}
        for m in range(self.num_minutes):
            for w in range(self.num_scenarios):
                rhs[f"def_excess_m{m+1}_w{w+1}"] = float(self.f_up_in[m, w])
                
        rhs["cvar_limit_constraint"] = 0.0
        return rhs
    
    def build_constraint_sense(self):
        sense = {c: GRB.LESS_EQUAL for c in self.def_excess + self.cvar_limit}
        return sense
    
    def build_input_data(self):
        return LP_InputData(
            VARIABLES=self.variables,
            CONSTRAINTS=self.def_excess + self.cvar_limit,
            objective_coeff=self.build_objective_coefficients(),
            constraints_coeff=self.build_constraint_coefficients(),
            constraints_rhs=self.build_constraint_rhs(),
            constraints_sense=self.build_constraint_sense(),
            objective_sense=GRB.MAXIMIZE,
            model_name=self.model_name
        )