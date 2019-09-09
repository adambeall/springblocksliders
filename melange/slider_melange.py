"""
Demonstration of a simple spring-block-slider model, in which sliding stress
is defined by an arbitrary function of strain-rate and yield stress.

The model solver is stored in lib_melange_slider and by default, slider strain-rate and stress is assumed to follow:

stress = ( c_0 * ln( strain-rate * eta_m / tau_y ) + c_0 ) / tau_y

where:
eta_m - matrix viscosity
tau_y - clast yield stress
c_0, c_1 - constants

tau_y is varied as a function of stress. At stress tau_1, tau_y is reduced to tau_y1. When stress lowers to tau_0, tau_y
is increased back to tau_y0. 

This example takes tau_y0 as 200 MPa and varies tau_y1 between 50, 60 and 80 MPa.


"""




import numpy
import matplotlib.pyplot as plt
import lib_melange_slider

# Compute models for three different values of 'weakened' tau_y
arrTauy1 = [50e6,60e6,80e6]

# 3 plots
figVel = plt.figure()
figStressTime = plt.figure()
figStressStrainrate = plt.figure()

# Solution loop
for i,tauy1 in enumerate(arrTauy1):

    # Initialise model
    s = lib_melange_slider.solver()
    # Change tauy_1
    s.tauy_1 = tauy1
    # Solve
    sol = s.solve()

    # Prepare output
    solT = sol['t'] / 3600 / 24 / 365
    solX = sol['x']
    solV = sol['v']
    solStressd = sol['nondim_stress']
    solSRd = sol['nondim_strainrate']

    # Plot bulk stress over time
    # it gets messy, so just do for the first plot
    if i == 0:
        figStressTime.gca().plot(solT,solX*s.k / 1e6 )
        figStressTime.gca().set_xlabel('Time (years)')
        figStressTime.gca().set_ylabel('Stress (MPa)')

   
    # Plot velocity over time
    figVel.gca().plot(solT[:-1],(solV[:-1]/1e-9))
    figVel.gca().set_xlabel('Time (years)')
    figVel.gca().set_ylabel('Velocity (m/s)')
    figVel.gca().set_yscale('log')
    figVel.gca().set_xlim([0,10])
    figVel.gca().set_ylim([0.5,10.0])

    # Plot stress vs strain-rate
    figStressStrainrate.gca().scatter(solSRd[:-1],solStressd[:-1],s=0.1)
    figStressStrainrate.gca().set_xlabel('Strain-rate (non-dimensional)')
    figStressStrainrate.gca().set_ylabel('Stress (non-dimensional)')



# Write plots to disk
figStressTime.savefig('stress_time.pdf')
figStressStrainrate.savefig('stress_strainrate.pdf')
figVel.savefig('velocity_time.pdf')
