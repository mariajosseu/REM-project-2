import numpy as np
from gurobipy import GRB
from models.OptimizationClasses import LP_InputData

from data.data_step2 import generate_fcr_profiles

class CVaRBuilder:
    """Builder for FCR-D UP Offering Strategy using CVaR approximation"""
    
    def __init__(self, model_name="CVaR_Model", alpha=0.90):
        # Upload data
        data = generate_fcr_profiles()
        self.f_up_in = data["f_up"][:, :100] # first 100 scenarios in sample (shape: 60x100)
        
        self.num_minutes = 60
        self.num_scenarios = 100
        
        self.alpha = alpha 
        self.epsilon = 1.0 - self.alpha 
        
        self.model_name = model_name
        self._build_names()

    def _build_names(self):
        """Build all variable and constraint names"""
        # Variables: c_up , beta, zeta_{m,w} 
        self.variables = (
            ["c_up"] + 
            ["beta"] + 
            [f"zeta_{m+1}_{w+1}" for m in range(self.num_minutes) for w in range(self.num_scenarios)]
        )
        
        # Constraints
        self.def_excess = [f"def_excess_m{m+1}_w{w+1}" for m in range(self.num_minutes) for w in range(self.num_scenarios)]
        self.cvar_limit = ["cvar_limit_constraint"]
        self.beta_limits = [f"beta_limit_m{m+1}_w{w+1}" for m in range(self.num_minutes) for w in range(self.num_scenarios)]
        self.beta_sign = ["beta_leq_zero"]

    def build_objective_coefficients(self):
        """Maximize c_up"""
        obj_coeff = {v: 0.0 for v in self.variables}
        obj_coeff["c_up"] = 1.0
        return obj_coeff

    def build_constraint_coefficients(self):
        """Build constraint coefficients"""
        coeff = {}
        
        # c^up - zeta_{m,w} <= F^uparrow_{m,w}
        for m in range(self.num_minutes):
            for w in range(self.num_scenarios):
                c_name = f"def_excess_m{m+1}_w{w+1}"
                z_name = f"zeta_{m+1}_{w+1}"
                coeff[c_name] = {
                    "c_up": 1.0,
                    z_name: -1.0
                }
                
        # (1 / (|m|*|w|)) * sum(zeta_{m,w}) - (1 - epsilon)*beta <= 0
        avg_multiplier = 1.0 / (self.num_minutes * self.num_scenarios)
        coeff["cvar_limit_constraint"] = {"beta": -(1.0 - self.epsilon)}
        for m in range(self.num_minutes):
            for w in range(self.num_scenarios):
                z_name = f"zeta_{m+1}_{w+1}"
                coeff["cvar_limit_constraint"][z_name] = avg_multiplier

        # beta - zeta_{m,w} <= 0  
        for m in range(self.num_minutes):
            for w in range(self.num_scenarios):
                c_name = f"beta_limit_m{m+1}_w{w+1}"
                z_name = f"zeta_{m+1}_{w+1}"
                coeff[c_name] = {
                    "beta": 1.0,
                    z_name: -1.0
                }
                
        # beta <= 0 
        coeff["beta_leq_zero"] = {"beta": 1.0}
            
        return coeff
    
    def build_constraint_rhs(self):
        """Build right-hand side values"""
        rhs = {}

        # c^up - zeta_{m,w} <= F^uparrow_{m,w} ==> RHS = F^uparrow_{m,omega}
        for m in range(self.num_minutes):
            for w in range(self.num_scenarios):
                rhs[f"def_excess_m{m+1}_w{w+1}"] = float(self.f_up_in[m, w])
                
        # (1 / (|m|*|w|)) * sum(zeta_{m,w}) - (1 - epsilon)*beta <= 0 ==> RHS = 0
        rhs["cvar_limit_constraint"] = 0.0
        
        # beta - zeta_{m,w} <= 0 ==> RHS = 0
        for m in range(self.num_minutes):
            for w in range(self.num_scenarios):
                rhs[f"beta_limit_m{m+1}_w{w+1}"] = 0.0
                
        # beta <= 0 ==> RHS = 0
        rhs["beta_leq_zero"] = 0.0
        
        return rhs
    
    def build_constraint_sense(self):
        all_constraints = self.def_excess + self.cvar_limit + self.beta_limits + self.beta_sign
        sense = {c: GRB.LESS_EQUAL for c in all_constraints}
        return sense
    
    def build_input_data(self):
        all_constraints = self.def_excess + self.cvar_limit + self.beta_limits + self.beta_sign
        return LP_InputData(
            VARIABLES=self.variables,
            CONSTRAINTS=all_constraints,
            objective_coeff=self.build_objective_coefficients(),
            constraints_coeff=self.build_constraint_coefficients(),
            constraints_rhs=self.build_constraint_rhs(),
            constraints_sense=self.build_constraint_sense(),
            objective_sense=GRB.MAXIMIZE,
            model_name=self.model_name
        )
    


class ALSOXBuilder:
    """Builder for FCR-D UP Offering Strategy using ALSOX approximation"""
    
    def __init__(self, model_name="ALSOX_Model", alpha=0.90):
        # Upload data
        data = generate_fcr_profiles()
        self.f_up_in = data["f_up"][:, :100] # first 100 scenarios in sample (shape: 60x100)
        
        self.num_minutes = 60
        self.num_scenarios = 100
        
        self.alpha = alpha 
        self.epsilon = 1.0 - self.alpha 

        # ALSO-X parameters
        # q: max number of scenarios allowed to violate the bid (shortfall scenarios)
        self.q = int(self.epsilon * self.num_minutes * self.num_scenarios)
        # M (Big-M):  max f_up is a good candidate for big-M 
        self.big_m = float(np.max(self.f_up_in)) * 1.5

        self.model_name = model_name
        self._build_names()

    def _build_names(self):
        """Build all variable and constraint names"""
        # Variables: c_up, y_{m,w} (binary)
        self.variables = (
            ["c_up"] + 
            [f"y_m{m+1}_w{w+1}" for m in range(self.num_minutes) for w in range(self.num_scenarios)]
        )
        
        # Constraints
        self.big_m_constraints = [f"big_m_m{m+1}_w{w+1}" for m in range(self.num_minutes) for w in range(self.num_scenarios)]
        self.q_limit = ["q_limit_constraint"]
        
    def build_objective_coefficients(self):
        """Maximize c_up"""
        obj_coeff = {v: 0.0 for v in self.variables}
        obj_coeff["c_up"] = 1.0
        return obj_coeff
    
    def build_constraint_coefficients(self):
        """Build constraint coefficients"""
        coeff = {}
        
        # c_up - M*y{m,w} <= F_up{m,w}
        for m in range(self.num_minutes):
            for w in range(self.num_scenarios):
                c_name = f"big_m_m{m+1}_w{w+1}"
                y_name = f"y_m{m+1}_w{w+1}"
                coeff[c_name] = {
                    "c_up": 1.0,
                    y_name: -self.big_m
                }
                
        # sum(y_{m,omega}) <= q
        coeff["q_limit_constraint"] = {}
        for m in range(self.num_minutes):
            for w in range(self.num_scenarios):
                y_name = f"y_m{m+1}_w{w+1}"
                coeff["q_limit_constraint"][y_name] = 1.0
            
        return coeff
    
    def build_constraint_rhs(self):
        """Build right-hand side values"""
        rhs = {}
        
        # c_up - M*y{m,w} <= F_up{m,w} ==> RHS = F_up{m,w}
        for m in range(self.num_minutes):
            for w in range(self.num_scenarios):
                rhs[f"big_m_m{m+1}_w{w+1}"] = float(self.f_up_in[m, w])
                
        # sum(y_{m,omega}) <= q ==> RHS = q
        rhs["q_limit_constraint"] = float(self.q)
        
        return rhs
    
    def build_constraint_sense(self):
        all_constraints = self.big_m_constraints + self.q_limit
        sense = {c: GRB.LESS_EQUAL for c in all_constraints}
        return sense
    
    def build_input_data(self):
        all_constraints = self.big_m_constraints + self.q_limit
        return LP_InputData(
            VARIABLES=self.variables,
            CONSTRAINTS=all_constraints,
            objective_coeff=self.build_objective_coefficients(),
            constraints_coeff=self.build_constraint_coefficients(),
            constraints_rhs=self.build_constraint_rhs(),
            constraints_sense=self.build_constraint_sense(),
            objective_sense=GRB.MAXIMIZE,
            model_name=self.model_name
        )