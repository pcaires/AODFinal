import numpy as np

import openmdao.api as om

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint


if  __name__ == "__main__":

    # --- Given parameters ----

    print('Complexity level select:')
    print('1- DV: aoa, span')
    print('2- DV: aoa, spar thickness')
    print('3- DV: aoa, chord scaling, twist, sweep')
    print('4- DV: all')
    sel = int(input('-'))
    
    rng = 1685 * 1852 #Range (m)
    M = 0.78   #Cruise Mach Number
    h = 10668  #Cruise alt (m) (ISA Atmosphere valid < 11km)
    
    b = 23.24  #span (m)
    
    wing_sweep = 30 #(deg)
    wing_taper = 0.3
    AR = 8
    t_c = 0.12
    Clmax = 0.6
    SFC = 0.38 #Engine specific fuel consuption (GE CF34) [1/h]
    
    TOW = 323608/9.81   #Take off weigth (kg)
    OEW = 193498/9.81   #Operating empty weigth (kg)

    Wing_tail_weight = 5700 #Approximation wing+tail weight (kg)

    Fuel_Capacity = 19595*0.4535924 #kg
    Payload_Useful= 18800*0.4535924 #kg
    
    Reserve_Fuel = .1*Fuel_Capacity #Approximation: 10% fuel capacity reserve
    Payload = .4*Payload_Useful #Approximation: 40% payload

    W0 = OEW - Wing_tail_weight + Reserve_Fuel + Payload #Fuselage + cargo + fuel

    
    #alloy Al 7075-T6 as the material used in the manufacturing of
    #the wing spar
    E = 70e9     #Young's modulus (Pa)
    G = 26e9     #Shear modulus (Pa)
    mrho = 3e3   #Material Density (kg/m3)
    yld = 480e6  #yield stress (Pa)
    yld = .8*yld  #Allowable yield stress 80% (Pa)

    wing_thick = np.array([0.01, 0.02])
    wing_radii = np.array([0.1, 0.2])
    tail_factor = 1/2
    
    n = np.array([1, 2.5]) #load factor

    #Wing Profile
    CL0 = 0.2
    CD0 = 0.015


    #From measurements file
    crt = 2.63 # Tail Root Chord (m)
    ctt = 1.15 # Tail tip Chord (m)
    bt = 8.54     # Tail Span (m)

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
        "fem_origin": 0.35,  
        "mesh":mesh,
        "CL0":CL0,
        "CD0":CD0,
        "k_lam": 0.05,  
        "t_over_c_cp": np.array([t_c, t_c]),  
        "c_max_t": 0.303,  
        "with_viscous": True, 
        "with_wave": True,
        "E":E,
        "G":G,
        "yield":yld,
        "mrho":mrho,
        "wing_weight_ratio": 2.0,
        "struct_weight_relief": True,  
        "distributed_fuel_weight": True,
        "Wf_reserve": Reserve_Fuel,  
        "thickness_cp":wing_thick,
        "radius_cp":wing_radii,
        "exact_failure_constraint": True,  # if false, use KS function
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
        "fem_origin": 0.35, 
        "mesh": mesh,
        "CL0": 0.0,  
        "CD0": 0.0,
        "fem_origin": 0.35,
        "k_lam": 0.05,  
        "t_over_c_cp": np.array([t_c, t_c]), 
        "c_max_t": 0.303,  
        "with_viscous": True,

        "with_wave": True,
        "E":E,
        "G":G,
        "yield":yld,
        "mrho":mrho,
        "wing_weight_ratio": 2.0,
        "struct_weight_relief": True,  # True to add the weight of the structure to the loads on the structure
        "distributed_fuel_weight": False,
        "thickness_cp":tail_factor*wing_thick,
        "radius_cp":tail_factor*wing_radii,
        "exact_failure_constraint": True,  # if false, use KS function
    }

    surfaces = [surf_dict, surf_dict2]

    
    #-----------------------------------


    # Create the OpenMDAO problem
    prob = om.Problem()

    # Independent variable component that will supply the
    # conditions to the problem.
    
    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output('v', val=v, units='m/s')
    indep_var_comp.add_output('alpha', val=2, units='deg')
    indep_var_comp.add_output('alpha_man', val=4, units='deg')
    indep_var_comp.add_output('Mach_number', val=M)
    indep_var_comp.add_output('re', val=re, units='1/m')
    indep_var_comp.add_output('rho', val=rho, units='kg/m**3')
    
    indep_var_comp.add_output('empty_cg', val=np.array([6.07,0,0]), units='m')

    # Aircraft parameters
    indep_var_comp.add_output('load_factor', val=n)
    indep_var_comp.add_output('sweep', val=wing_sweep, units='deg')
    indep_var_comp.add_output('taper', val=wing_taper)
    indep_var_comp.add_output("CT", val=SFC/3600, units="1/s")
    indep_var_comp.add_output("R", val=rng, units="m")
    indep_var_comp.add_output("W0", val=W0, units="kg")
    indep_var_comp.add_output('tail_sweep', val=tail_sweep, units='deg')
    indep_var_comp.add_output('tail_taper', val=tail_taper)
    indep_var_comp.add_output('tail_dihedral', val=tail_dihedral, units='deg')


    prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

    #------------------
    
    for surface in surfaces:
        geom_group = AerostructGeometry(surface=surface)
        prob.model.add_subsystem(surface["name"], geom_group)

    for i in range(2): 
        aero_group = AerostructPoint(surfaces=surfaces)
        point_name = "aero_point_{}".format(i)
        prob.model.add_subsystem(
               point_name,
               aero_group)
        prob.model.connect("v", point_name + ".v")
        prob.model.connect("Mach_number", point_name + ".Mach_number")
        prob.model.connect("re", point_name + ".re")
        prob.model.connect("rho", point_name + ".rho")
        prob.model.connect("CT", point_name + ".CT")
        prob.model.connect("R", point_name + ".R")
        prob.model.connect("W0", point_name + ".W0")
        prob.model.connect("empty_cg", point_name + ".empty_cg")
        prob.model.connect("load_factor", point_name + ".load_factor", src_indices=[i])

            
            
        for surface in surfaces:
            name = surface["name"]
            com_name = point_name + "." + name + "_perf"
            prob.model.connect(
                name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed"
            )
            prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")

            # Connect aerodyamic mesh to coupled group mesh
            prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")

            # Connect performance calculation variables
            prob.model.connect(name + ".radius", com_name + ".radius")
            prob.model.connect(name + ".thickness", com_name + ".thickness")
            prob.model.connect(name + ".nodes", com_name + ".nodes")
            prob.model.connect(name + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location")
            prob.model.connect(name + ".structural_mass", point_name + "." + "total_perf." + name + "_structural_mass")
            prob.model.connect(name + ".t_over_c", com_name + ".t_over_c")

    # Connect point specific parameters
    prob.model.connect("alpha", "aero_point_0.alpha")
    prob.model.connect("alpha_man", "aero_point_1.alpha")

    # Connect surface specific parameters

    prob.model.connect("taper", 'wing.geometry.mesh.taper.taper')
    prob.model.connect("sweep", 'wing.geometry.mesh.sweep.sweep')
    prob.model.connect("tail_taper", 'tail.geometry.mesh.taper.taper')
    prob.model.connect("tail_sweep", 'tail.geometry.mesh.sweep.sweep')
    prob.model.connect("tail_dihedral", 'tail.geometry.mesh.dihedral.dihedral')
            
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['tol'] = 1e-4


    recorder = om.SqliteRecorder("CRJ700optim_{}.db".format(sel))

        
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['record_derivatives'] = True
    prob.driver.recording_options['includes'] = ['*']

    # Setup problem and add design variables, constraint, and objective
    """
    Complexity level select:
    1- DV: aoa, span
    2- DV: aoa, spar thickness
    3- DV: aoa, chord scaling, twist, sweep
    4- DV: all
    """

    #Global
    prob.model.add_design_var("alpha", lower=-10.0, upper=25.0)
    prob.model.add_design_var("alpha_man", lower=-10.0, upper=25.0)
    prob.model.add_constraint("aero_point_1.wing_perf.Cl", upper=Clmax)
    prob.model.add_constraint("aero_point_1.tail_perf.Cl", upper=Clmax) 
    prob.model.add_constraint("aero_point_0.L_equals_W", equals=0.0)
    prob.model.add_constraint("aero_point_1.L_equals_W", equals=0.0)
    prob.model.add_constraint("aero_point_1.wing_perf.failure", upper=0.0)
    prob.model.add_constraint("aero_point_1.tail_perf.failure", upper=0.0)
    prob.model.add_objective("aero_point_0.fuelburn", scaler=1e-2)

    if sel == 1:
        prob.model.add_design_var("wing.geometry.mesh.stretch.span",lower = 0.7*b, upper = 2*b)
        prob.model.add_design_var("tail.geometry.mesh.stretch.span",lower = 0.7*bt, upper = 2*bt)
    elif sel == 2:
        prob.model.add_design_var("wing.thickness_cp", lower=0.001, upper=0.2, scaler = 1e4)
        prob.model.add_design_var("tail.thickness_cp", lower=0.0005, upper=0.1, scaler = 1e4)
        prob.model.add_constraint("aero_point_1.wing_perf.thickness_intersects", upper=0.0)
        prob.model.add_constraint("aero_point_1.tail_perf.thickness_intersects", upper=0.0)
    elif sel == 3:
        prob.model.add_design_var("wing.geometry.mesh.rotate.twist", lower=-10.0, upper=15.0)
        prob.model.add_design_var("tail.geometry.mesh.rotate.twist", lower=-10.0, upper=15.0)
        prob.model.add_design_var("wing.geometry.mesh.scale_x.chord", lower=0.7, upper=10.0)
        prob.model.add_design_var("tail.geometry.mesh.scale_x.chord", lower=0.7, upper=10.0)
        prob.model.add_design_var("sweep", lower=-10.0, upper=40)
        prob.model.add_design_var("tail_sweep", lower=-10.0, upper=40)
    elif sel == 4:
        prob.model.add_design_var("wing.geometry.mesh.stretch.span",lower = 0.7*b, upper = 2*b)
        prob.model.add_design_var("tail.geometry.mesh.stretch.span",lower = 0.7*bt, upper = 2*bt)
        prob.model.add_design_var("wing.thickness_cp", lower=0.001, upper=0.2, scaler = 1e4)
        prob.model.add_design_var("tail.thickness_cp", lower=0.0005, upper=0.1, scaler = 1e4)
        prob.model.add_constraint("aero_point_1.wing_perf.thickness_intersects", upper=0.0)
        prob.model.add_constraint("aero_point_1.tail_perf.thickness_intersects", upper=0.0)
        prob.model.add_design_var("wing.geometry.mesh.rotate.twist", lower=-10.0, upper=15.0)
        prob.model.add_design_var("tail.geometry.mesh.rotate.twist", lower=-10.0, upper=15.0)
        prob.model.add_design_var("wing.geometry.mesh.scale_x.chord", lower=0.7, upper=10.0)
        prob.model.add_design_var("tail.geometry.mesh.scale_x.chord", lower=0.7, upper=10.0)
        prob.model.add_design_var("sweep", lower=-10.0, upper=40)
        prob.model.add_design_var("tail_sweep", lower=-10.0, upper=40)

    
    
        
    prob.setup(check=True)
    prob.run_driver()
    

    print("Design Variables")
    print("AoA: ", prob["alpha"], "[deg]")
    print("AoA (2.5g): ", prob["alpha_man"], "[deg]")
    print("Wing Span: ", prob["wing.geometry.mesh.stretch.span"], "[m]")
    print("Tail Span: ", prob["tail.geometry.mesh.stretch.span"], "[m]")
    print("Wing Spar thickness: ", prob["wing.thickness_cp"], "[m]")
    print("Tail Spar thickness: ", prob["tail.thickness_cp"], "[m]")
    

    print("Performance Metrics")
    print("Empty CG: ", prob["aero_point_0.empty_cg"])
    print("Load factor: ",n[0])
    print("CL: ", prob["aero_point_0.CL"][0])
    print("CD: ", prob["aero_point_0.CD"][0])
    print("CM: ", prob["aero_point_0.CM"][1])
    print("Load factor: ",n[1])
    print("CL: ", prob["aero_point_1.CL"][0])
    print("CD: ", prob["aero_point_1.CD"][0])
    print("CM: ", prob["aero_point_1.CM"][1])



    #print("The range value is ", prob["R"][0], "[m]")
    print("The fuel burn value is ", prob["aero_point_0.fuelburn"][0], "[kg]")
    print("Lift to Weight difference: ", prob["aero_point_0.L_equals_W"][0])
    print("No Failure (2.5g): ", prob["aero_point_1.wing_perf.failure"], "<0")
    
    #print("No Intersects: ", prob["aero_point_0.wing_perf.thickness_intersects"], "<0")

    print("Tail section Cl (2.5g): ",prob["aero_point_1.tail_perf.Cl"])
    print("Wing section Cl (2.5g): ",prob["aero_point_1.wing_perf.Cl"])

    print("Total Weight: ",prob["aero_point_0.total_weight"][0], " [N]")
    
    
    
