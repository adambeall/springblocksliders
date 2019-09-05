import numpy
from scipy.integrate import solve_ivp

"""
1D Spring-block slider, with sliding stress modelled as a Bingham rheology incorporating rate-and-state friction.

The spring-block slider system is modelled as:
    k * (Vp * t - x) = tau

    k: spring constant (Pa / m)
    Vp: loading velocity
    x: slider displacement
    tau: slider stress

Stress is set as a Voigt mixture:
    tau = phi * tau_f + (1 - phi) * tau_v

    phi: volume fraction
    tau_f: stress of frictional material
    tau_v: stress of viscous material

The viscous material has a Newtonian rheology:
    tau_v = V / L_v * eta_m

    V: dx_dt
    L_v: shear-zones thickness
    eta_m: matrix viscosity

The frictional material follows rate-and-state friction:

    tau_f = sigNormEff * (mu_0 + a * ln(V/V_0) + b * ln( (theta * V_0 / D_c)) ) 

    for rate-and-state paramaters mu_0, a,b, V_0,D_c 
    sigNormEff: effective normal stress
    theta: state variable, which evolves according to the aging law,  dtheta_dt = 1 - V*theta / D_c

This system is solved using radau (incorporated in solve_ivp, SciPy library), a 4th order implicit Runge-Kutta method. 


"""

class solver:


    def __init__(self):
        ## Default rate-and-state parameters

        # Reference slip
        self.Dc = 1e-3 
        # Reference velocity
        self.V0 = 1e-6
        # a and b parameters
        self.a = 0.001
        self.b = 0.006
        # Reference friction
        self.mu_0 = 0.6
        # Effective normal stress
        self.sigNormEff = 200e6


        ## Slider options

        # Calculate critical stiffness for regular rate-and-state
        # slider and set spring constant k as a fraction of it
        self.k_c = self.sigNormEff * (self.b-self.a) / self.Dc
        self.k = 0.1 * self.k_c

        # plate velocity
        self.Vp = 1e-9
        # initial velocity
        self.v_init = 1.0 * self.Vp
        # initial state variable
        self.theta_init = 0.9 / self.v_init * self.Dc
        # finish time (seconds)
        self.tf = 10.0*3600*24*365
        


        ## Default visco-brittle Bingham rheology parameters

        # shear zone thickness 
        self.Lv = 1e2
        # fraction of frictional material ( 0 <= phi <= 1)
        self.phi = 0.6
        # Critical viscosity 
        self.eta_c = ((self.b-self.a)*self.sigNormEff * self.phi - self.k * self.Dc)*self.Lv/self.Vp / (1.-self.phi)
        # matrix viscosity as a fraction of a critical viscosity eta_c
        self.eta_m = 0.5 * self.eta_c



    def fnStress(self,arrV,arrTheta):
        # Calculate stress from velocity (arrV: constant or array) 
        # and state variable (arrTheta: constant or array).
        # This can be switched out for different rheologies.

        # Insert constants into arrays
        if not hasattr(arrV,'__len__'):
            arrV = [arrV]
            arrTheta = [arrTheta]

        # Empty stress array
        arrStress = numpy.zeros(len(arrV))

        # Fill stress array
        for i,V in enumerate(arrV): 

            theta = arrTheta[i]
            mu = self.mu_0 + self.a * numpy.log(V/self.V0) + self.b * numpy.log(self.V0 * theta / self.Dc)

            # frictional component
            tau_f = self.sigNormEff * mu
            # viscous component
            tau_v = self.eta_m * V / self.Lv 

            # Voigt / Bingham mixture
            arrStress[i] = self.phi * tau_f + (1.-self.phi) * tau_v

        return arrStress


    
    def fnX(self,V,theta,t):
        # Calculate slider displacement
        x = self.Vp*t -  self.fnStress(V,theta)/self.k

        return x


    # Calculate dV_dt using the chain-rule,
    # which requires dTheta_dt, dTau_dt, dTau_dTheta,
    # dTau_dt (already known) and dTau_dV (known function of V)
    # these could be switched out in the future for different rheologies

    def fnThetaDeriv(self,theta,V):
        dTheta_dt = 1. - theta * V / self.Dc
        return dTheta_dt 


    def fnDTauDV(self,theta,V):
        dTau_dV = self.phi * self.sigNormEff * self.a / V + (1.-self.phi) * self.eta_m / self.Lv 
        return dTau_dV


    def fnDTauDTheta(self,theta,V):
        dTau_dTheta = self.phi * self.sigNormEff * self.b / theta
        return dTau_dTheta



    # function describing ODE, which calculates dV_dt and dTheta_dt
    # for a given array, arrVar = [V,theta] 
    def fnODE(self,t,arrVar):
        V,theta = arrVar
        
        dTheta_dT = self.fnThetaDeriv(theta,V)
        dTau_dV = self.fnDTauDV(theta,V)
        dTau_dTheta = self.fnDTauDTheta(theta,V)

        dV_dT = (self.k * (self.Vp - V) - dTau_dTheta * dTheta_dT) / dTau_dV 


        return [dV_dT,dTheta_dT]



    # ODE solver
    def solve(self):

        # Print set parameters
        print('-------------------------------')
        print('Using parameters:')
        print('- - - - - - - - - - - - - - - -')
        print('a = %.2e \t b = %.2e \t a-b = %.2e' %(self.a,self.b,self.a-self.b))
        print('mu_0 = %.2f \t D_c = %.2e \t V_0 = %.2e' %(self.mu_0, self.Dc, self.V0)) 
        print('sigma_eff = %.2e \t L_v = %.2e' %(self.sigNormEff, self.Lv)) 
        print('k = %.2e (%.2f%% of k_c)' %(self.k, self.k / self.k_c*100.0)) 
        print('V_p = %.2e \t phi = %.2f' %(self.Vp, self.phi))
        print('eta_m = %.2e (%.2f%% of eta_c)' %(self.eta_m, self.eta_m / self.eta_c))
        print('-------------------------------')



        # Uses 5th order implicit Runge-Kutta
        # automatically determines time-step sizes
        solODE = solve_ivp(fun=self.fnODE,t_span = [0.,self.tf],y0 =
               [self.v_init,self.theta_init],rtol=1e-12,atol=1e-10,dense_output=True,method="Radau") 
               
        dicSolve = {'v':solODE.y[0], 'theta':solODE.y[1], 't':solODE.t}

        return dicSolve


