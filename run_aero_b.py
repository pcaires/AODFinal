import numpy as np

import openmdao.api as om

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

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
T = 288.15 -0.0065*h                        #Temperature [K]
M = v / ((1.4* R * (T))**(0.5))             #Mach Number

l = S/b #mean chord
mu = 1.458e-6 * T**(1.5) * (T+110.4)**(-1) #Dynamic viscosity [Ns/m^2]
nu = mu/rho #kinematic viscosity [m^2/s]

re = v/nu #Reynold Number 1/m


#Reference CL for Cesna 172
W = 756 #Operating weight of aircraft kg
W = W*9.81 #N


CL_ = (2*W)/(rho * S * v**2)
#CL_ = 0.7


print('wrt: alpha-0; alpha/twist-1; alpha/chord-2; alpha/twist/chord-3')
ch = int(input('-'))


num_y = 7
num_x = 2



# Create a dictionary to store options about the mesh
mesh_dict = {'num_y' : num_y,
             'num_x' : num_x,
             'wing_type' : wing_type,
             'span' : b,
             'root_chord' : l,
             'symmetry' : True}

# Generate the aerodynamic mesh based on the previous dictionary
mesh = generate_mesh(mesh_dict)

# Create a dictionary with info and options about the aerodynamic
# lifting surface
surface = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : True,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'                                    
            'mesh' : mesh,
            
            'fem_model_type' : 'tube',
            

            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            'CL0' : CL0,            # CL of the surface at alpha=0
            'CD0' : CD0,            # CD of the surface at alpha=0

            # Airfoil properties for viscous drag calculation
            'k_lam' : 0.05,         # percentage of chord with laminar
                                    # flow, used for viscous drag
            't_over_c_cp' : np.array([0.12]),      # thickness over chord ratio (NACA2412)
            'c_max_t' : .303,       # chordwise location of maximum (NACA2412)
                                    # thickness
                                    
                                    
            'with_viscous' : True,  # if true, compute viscous drag
            'with_wave' : False,     # if true, compute wave drag
            }

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

# Add this IndepVarComp to the problem model
prob.model.add_subsystem('prob_vars',
    indep_var_comp,
    promotes=['*'])

# Create and add a group that handles the geometry for the
# aerodynamic lifting surface
geom_group = Geometry(surface=surface)
prob.model.add_subsystem(surface['name'], geom_group)

# Create the aero point group, which contains the actual aerodynamic
# analyses
aero_group = AeroPoint(surfaces=[surface])
point_name = 'aero_point_0'
prob.model.add_subsystem(point_name, aero_group,
    promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'cg'])

name = surface['name'] # = 'wing' (from surface dictionary) 

# Connect the mesh from the geometry component to the analysis point
prob.model.connect(name + '.mesh', point_name + '.' + name + '.def_mesh')

# Perform the connections with the modified names within the
# 'aero_states' group.
prob.model.connect(name + '.mesh', point_name + '.aero_states.' + name + '_def_mesh')

prob.model.connect(name + '.t_over_c', point_name + '.' + name + '_perf.' + 't_over_c')

# Import the Scipy Optimizer and set the driver of the problem to use
# it, which defaults to an SLSQP optimization method
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['tol'] = 1e-9

recorder = om.SqliteRecorder("aero.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options['includes'] = ['*']

# Setup problem and add design variables, constraint, and objective
#   prob.model.add_design_var('wing.twist_cp', lower=-10., upper=15.)

prob.model.add_design_var('alpha',-10,10)

if ch>1:
    prob.model.add_design_var('wing.mesh.scale_x.chord',0.5,3)
    print('Added chord scale design variable')
    
if ch == 1 or ch == 3:
    prob.model.add_design_var('wing.mesh.rotate.twist',-10,10)
    print('Added wing twist design variable')


prob.model.add_constraint(point_name + '.wing_perf.CL', equals=CL_)

if ch>1:
    print('Added wing area constraint')
    prob.model.add_constraint(point_name + '.wing.S_ref',equals = S)



prob.model.add_objective(point_name + '.wing_perf.CD', scaler=1e4)

# Set up and run the optimization problem
prob.setup()
# prob.check_partials(compact_print=True)
# exit()
prob.run_driver()
# prob.run_model()

print('CD: ',prob['aero_point_0.wing_perf.CD'][0])

print('CL: ',prob['aero_point_0.wing_perf.CL'][0])

print('CM: ',prob['aero_point_0.CM'][1])

print('AoA alpha: ',prob['alpha'][0])

print('Twist Dist (tip->root): ', prob['wing.mesh.rotate.twist'])

print('Chord Dist (tip->root): ', prob['wing.mesh.scale_x.chord'])
