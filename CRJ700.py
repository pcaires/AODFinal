import numpy as np

import openmdao.api as om

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint


if __name__ == "__main__":

    # --- Given parameters ----

    rng = 1685 #Range (Nautical miles)
    M = 0.78   #Cruise Mach Number
    h = 10668 #Cruise alt (m)
    
    b = 23.24  #span (m)
    
    wing_sweep = 30 #(deg)
    wing_taper = 0.3
    AR = 8
    t_c = 0.12
    Clmax = 0.6
    SFC = 0.38 #Engine specific fuel consuption (GE CF34)
    
    TOW = 323608   #Take off weigth (N)
    OEW = 193498   #Operating empty weigth (N)

    #alloy Al 7075-T6 as the material used in the manufacturing of
    #the wing spar
    E = 70e9     #Young's modulus (Pa)
    G = 26e6     #Shear modulus (Pa)
    mrho = 3e3   #Material Density (kg/m3)
    yld = 480e6  #Allowable yield stress (Pa)

    n = 2.5 #load factor

    #Wing Profile
    CL0 = 0.0
    CD0 = 0.15


    #From measurements file
    crt = 2.63 # Tail Root Chord (m)
    ctt = 1.15 # Tail tip Chord (m)
    bt = 9     # Tail Span (m)

    tail_taper = ctt/crt
    tail_sweep = 35 #deg
    tail_dihedral = 86-90 #deg

    tail_offset = np.array([15.68, 0.0, 4.76])
    
               
    #########################

    R =  287.052874 #Perfect Gas Constant

    rho = 1.225*(1- (0.0065*h)/288.15)**(4.25588) #Density [kg/m^3]
    T = 288.15 -0.0065*h                          #Temperature [K]
    v = M * ((1.4* R * (T))**(0.5))               #Velocity [m/s]

    cm = b/AR #mean chord
    S = b*cm  #wing area

    cr = 2*cm/(1+wing_taper) #root chord


    
    mu = 1.458e-6 * T**(1.5) * (T+110.4)**(-1) #Dynamic viscosity [Ns/m^2]
    nu = mu/rho #kinematic viscosity [m^2/s]

    re = v/nu #Reynold Number 1/m


    #Reference CL 
    CL_ = (2*TOW)/(rho * S * v**2)

    
    
    #Define Wing
    mesh_dict = {
        "num_y": 7,
        "num_x": 2,
        "wing_type": "rect",
        "symmetry": True,
        "num_twist_cp": 5,
        "span":b,
        "root_chord":cr,
        }
    
    mesh = generate_mesh(mesh_dict)

    surf_dict = {
        "name": "wing",
        "symmetry":True,
        "S_ref_type":"wetted",
        "fem_model_type":"tube",
        "mesh":mesh,
        "CL0":CL0,
        "CD0":CD0,
        "k_lam": 0.05,  
        "t_over_c_cp": np.array([t_c]),  
        "c_max_t": 0.303,  
        "with_viscous": True, 
        "with_wave": True,
        "E":E,
        "G":G,
        "yield":yld,
        "mrho":mrho
        }
    #-----------------------------------

    #Define Horizontal Stablizer
    mesh_dict = {
        "num_y": 7,
        "num_x": 2,
        "wing_type": "rect",
        "symmetry": True,
        "root_chord": crt,
        "span": bt,
        "offset": tail_offset
        }

    mesh = generate_mesh(mesh_dict)

    surf_dict2 = {
       
    "name": "tail", 
    "symmetry": True,
    "S_ref_type": "wetted",
    "fem_model_type":"tube",
    "mesh": mesh,
    "CL0": 0.0,  
    "CD0": 0.0,
    "fem_origin": 0.35,
    "k_lam": 0.05,  
    "t_over_c_cp": np.array([t_c]), 
    "c_max_t": 0.303,  
    "with_viscous": True,
    "with_wave": True,
    "E":E,
    "G":G,
    "yield":yld,
    "mrho":mrho
    }

    surfaces = [surf_dict, surf_dict2]
    #-----------------------------------


    # Create the OpenMDAO problem
    prob = om.Problem()

    # Independent variable component that will supply the
    # conditions to the problem.
    
    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output('v', val=v, units='m/s')
    indep_var_comp.add_output('alpha', val=0, units='deg')
    indep_var_comp.add_output('Mach_number', val=M)
    indep_var_comp.add_output('re', val=re, units='1/m')
    indep_var_comp.add_output('rho', val=rho, units='kg/m**3')
    indep_var_comp.add_output('cg', val=np.zeros((3)), units='m')

    # Aircraft parameters
    indep_var_comp.add_output('load_factor', val=n)
    indep_var_comp.add_output('sweep', val=wing_sweep, units='deg')
    indep_var_comp.add_output('taper', val=wing_taper)
    indep_var_comp.add_output("CT", val=SFC, units="1/s")
    indep_var_comp.add_output("R", val=rng, units="m")
    indep_var_comp.add_output('tail_sweep', val=tail_sweep, units='deg')
    indep_var_comp.add_output('tail_taper', val=tail_taper)
    indep_var_comp.add_output('tail_dihedral', val=tail_dihedral, units='deg')


    prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

    #------------------
    
    for surface in surfaces:

        geom_group = Geometry(surface=surface)
        prob.model.add_subsystem(surface["name"], geom_group)
        

    for i in range(1):
        aero_group = AeroPoint(surfaces=surfaces)
        point_name = "aero_point_{}".format(i)
        prob.model.add_subsystem(point_name, aero_group)

        prob.model.connect("v", point_name + ".v")
        prob.model.connect("alpha", point_name + ".alpha")
        prob.model.connect("Mach_number", point_name + ".Mach_number")
        prob.model.connect("re", point_name + ".re")
        prob.model.connect("rho", point_name + ".rho")
        prob.model.connect("cg", point_name + ".cg")
        #prob.model.connect("load_factor", point_name + ".load_factor")
        #prob.model.connect("CT", point_name + ".CT")
        #prob.model.connect("R", point_name + ".R")

        # Connect the parameters within the model for each aero point
        for surface in surfaces:
            name = surface["name"]
            prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")
            prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")
            prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")
            
    # Connect surface specific parameters

    prob.model.connect("taper", 'wing.mesh.taper.taper')
    prob.model.connect("sweep", 'wing.mesh.sweep.sweep')
    prob.model.connect("tail_taper", 'tail.mesh.taper.taper')
    prob.model.connect("tail_sweep", 'tail.mesh.sweep.sweep')
    prob.model.connect("tail_dihedral", 'tail.mesh.dihedral.dihedral')
    

            
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['tol'] = 1e-9

    recorder = om.SqliteRecorder("CRJ700baseline.db")
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['record_derivatives'] = True
    prob.driver.recording_options['includes'] = ['*']

    prob.model.add_objective(point_name + '.wing_perf.CD', scaler=1e4)
    prob.model.add_design_var('alpha',-10,10)

    prob.setup()

    #prob.run_model()
    prob.run_driver()
  


    
                    
