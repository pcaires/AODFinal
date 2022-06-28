import numpy as np

import openmdao.api as om

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint


if __name__ = "__main__":

    # --- Given parameters ----
    v = 63 #m/s
    h = 2000 #m

    wing_type = 'rect'
    b = 11
    S = 16.2

    #WING SECTION = NACA 2412
    # Max thickness 12% at 30% chord.
    # Max camber 2% at 40% chord
    CL0 = 0.2183           
    CD0 = 0.018            
    #########################

    R =  287.052874 #Perfect Gas Constant

    rho = 1.225*(1- (0.0065*h)/288.15)**(4.25588) #Density [kg/m^3]
    T = 288.15 -0.0065*h                          #Temperature [K]
    M = v / ((1.4* R * (T))**(0.5))               #Mach Number

    l = S/b #mean chord
    mu = 1.458e-6 * T**(1.5) * (T+110.4)**(-1) #Dynamic viscosity [Ns/m^2]
    nu = mu/rho #kinematic viscosity [m^2/s]

    re = v/nu #Reynold Number 1/m


    #Reference CL for Cesna 172
    W = 756 #Operating weight of aircraft kg
    W = W*9.81 #N


    CL_ = (2*W)/(rho * S * v**2)
    
    
    #Define Wing
    mesh_dict = {
        "num_y": 7,
        "num_x": 2,
        "wing_type": "CRM",
        "symmetry": True,
        "num_twist_cp": 5
        }
    
    mesh,twist_cp = generate_mesh(mesh_dict)

    surface_dict = {
        "name": "wing",
        "symmetry":True,
        "S_ref_type":"wetted",
        "fem_model_type":"tube",
        "twist_cp":twist_cp,
        "mesh":mesh,
        "CL0":CL0,
        "CD0":CD0,
        "k_lam": 0.05,  
        "t_over_c_cp": np.array([0.15]),  
        "c_max_t": 0.303,  
        "with_viscous": True, 
        "with_wave": True
        }
    #-----------------------------------

    #Define Horizontal Stablizer
    mesh_dict = {
        "num_y": 7,
        "num_x": 2,
        "wing_type": "rect",
        "symmetry": True,
        "offset": np.array([50, 0.0, 0.0])
        }

    mesh = generate_mesh(mesh_dict)

    surf_dict2 = {
       
    "name": "tail", 
    "symmetry": True,
    "S_ref_type": "wetted", 
    "twist_cp": twist_cp,
    "mesh": mesh,
    "CL0": 0.0,  
    "CD0": 0.0,
    "fem_origin": 0.35,
    "k_lam": 0.05,  
    "t_over_c_cp": np.array([0.15]), 
    "c_max_t": 0.303,  
    "with_viscous": True,
    "with_wave": True,
    }

    surfaces = [surf_dict, surf_dict2]
    #-----------------------------------


    # Create the OpenMDAO problem
    prob = om.Problem()

    # Create an independent variable component that will supply the flow
    # conditions to the problem.
    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output('v', val=v, units='m/s')
    indep_var_comp.add_output('alpha', val=0, units='deg')
    indep_var_comp.add_output('Mach_number', val=M)
    indep_var_comp.add_output('re', val=re, units='1/m')
    indep_var_comp.add_output('rho', val=rho, units='kg/m**3')
    indep_var_comp.add_output('cg', val=np.zeros((num_x+1)), units='m')


    
                    
