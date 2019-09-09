"""
Demonstration of a simple spring-block-slider model, in which sliding stress
is defined as a Voigt mixture of a rate-and-state frictional material (of volumetric proportion phi)
and a Newtonian viscous material, which is essentially a Bingham plastic.

The model solver is stored in lib_slider and contains default parameters which are printed at start-up.
Matrix viscosity can be varied as a fraction of the critical viscosity.
"""

import numpy
import matplotlib.pyplot as plt
import lib_slider

# Initialise model
# using default model parameters
model = lib_slider.solver()

# Vary the matrix viscosity as a fraction of the critical viscosity
model.eta_m = 0.5 * model.eta_c

# Solve
sol = model.solve()

# Solution returns velocity, theta and time
solV = sol['v']
solTheta = sol['theta']
solTimeYears = sol['t']  / 3600.0 / 24.0 / 365.0

# Back calculate stress, from solution V and theta
solStress = model.fnStress(solV,solTheta)


### Plot output


## Plot phase diagram (tau vs theta) and compare to steady-state curve

# Phase diagram
arrThetaSolLog = numpy.log10(solTheta)
plt.plot(arrThetaSolLog,solStress/1e6)

# Calculate steady-state curve (calculate for nP points)
nP = int(1e2)
arrThetaLog = numpy.linspace(numpy.min(arrThetaSolLog),numpy.max(arrThetaSolLog),nP)
arrTau = numpy.zeros(nP)

for i in range(nP):
    arrTau[i] = model.phi * (model.mu_0 + (model.b - model.a)*numpy.log(model.V0*10.**arrThetaLog[i] / model.Dc)) + (1.-model.phi) * model.eta_m * model.Dc/10.**arrThetaLog[i]/model.sigNormEff/ model.Lv

# Plot steady-state curve
plt.plot(arrThetaLog,arrTau*model.sigNormEff/1e6,"--",c="black")


plt.xlabel('Log Theta')
plt.ylabel('Stress (MPa)')
plt.savefig('phase.pdf')
plt.clf()



## Plot velocity over time

plt.plot(solTimeYears,solV/1e-9,c="black")

plt.xlabel('Time (years)')
plt.ylabel('Velocity (m/s)')
plt.ylim([1e-4,1e2])
plt.xlim([0,5])
plt.yscale('log')

plt.savefig('v_time.pdf')
plt.clf()




## Plot tau and tau_y over time
tauy = (solStress - (1.-model.phi)*solV/model.Lv * model.eta_m) / model.phi
plt.plot(solTimeYears,tauy / 1e6,label=r'$\tau_y$')
plt.plot(solTimeYears,solStress / 1e6,label=r'$\tau_b$')
plt.xlabel('Time (years)')
plt.ylabel('Stress (MPa)')
plt.legend()

plt.savefig('stress_time.pdf')


