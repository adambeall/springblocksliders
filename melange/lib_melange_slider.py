import numpy



# Default rheology function, which can be switched out
def fnRheology(self,stress,boolWeakened = False):

        # check if the yield stress is currently its upper or lower value 
        if boolWeakened: 
            tau_y = self.tauy_1
        else:
            tau_y = self.tauy_0

        # Effective rheology for melange with phi = 0.61, D=0.14 and clast sizes following power-law distribution 

        C_0 = 0.23
        C_1 = 1.21 
        # Calculate velocity
        V = numpy.exp((stress / tau_y - C_1) / C_0) * 2.0 * self.Lv * tau_y / self.eta_m 

        return V




class solver:


    def __init__(self):

        # Default slider / rheology parameters
        # Shear-zone thickness
        self.Lv = 1e2
        # Spring stiffnes (Pa / m)
        self.k = 1e8
        # Loading velocity (m / s)
        self.vDrive = 1e-9
        # Finish time (seconds)
        self.tf = 10.0*3600*24*365

        # Upper stress, at which weakening is triggered
        self.stress_1 =50e6
        # Lower stress, at which strengthening is triggered
        self.stress_0 = self.stress_1 - 1.0e6
       
        # Weakened / unweakened values of tau_y
        self.tauy_1 = 60e6
        self.tauy_0 = 200e6


        # Starting slider stress / displacement
        self.stress_init = self.stress_0
        self.x_init =  self.stress_init/self.k

        # Matrix viscosity 
        self.eta_m = 1e18

        # Chosen rheology
        self.fnRheology = fnRheology



    def fnVelocity(self,x,t):
        # Work out current stress
        stress = self.k * (self.vDrive * t - x + self.x_init)
        # Return velocity
        sol = self.fnRheology(self,stress)

        return sol



    # ODE, in terms of x and dx_dt
    def fnODE(self,t,x):
        V = self.fnVelocity(x,t)
        
        return [V]



    # Compute the model solution
    def solve(self):
        # Number of time-steps
        nSteps = 20000
        # Initialise solution arrays
        arrT, arrV, arrX, arrStressd, arrSRd = numpy.zeros((5,nSteps))
        # Time-step size
        self.tStepSize = self.tf / nSteps


        # init coniditions
        arrX[0] = self.x_init
        boolWeakened = False

        # Solution loop
        for i in range(nSteps-1):

            ## Calculate current non-dim stress and strain-rate
            # Calculate current stress, for output
            stress = self.k * arrX[i] 
            # Find current tau_y 
            if boolWeakened:
                eta_y = self.tauy_1 
            else:
                eta_y = self.tauy_0
            # Store non-dim stress
            arrStressd[i] = stress / eta_y

            # Current velocity and strain-rate
            arrV[i] = self.fnRheology(self,stress,boolWeakened)
            arrSRd[i] = 0.5* arrV[i] /self.Lv * self.eta_m / eta_y 


            # Loading velocity
            v = self.vDrive - arrV[i] 

            # Calculate stress step and check if the upper or lower stress thresholds will be met.
            #   If so, adjust step-size to avoid overshoot
            stress_grad = self.k * v 
            stress_trial = stress + stress_grad * self.tStepSize
            if stress_trial  > self.stress_1:
                deltaT = (self.stress_1 - stress) / stress_grad 
                boolWeakened = True 
            elif stress_trial < self.stress_0: 
                deltaT = (self.stress_0 - stress) / stress_grad 
                boolWeakened = False 
            else:
                deltaT = self.tStepSize
            deltaT = self.tStepSize

            # Integrate for next time and displacement
            arrT[i+1] = arrT[i] + deltaT
            arrX[i+1] = arrX[i] + v * deltaT
       
        # Dictionary of solution arrays
        sol = {'t':arrT,'x':arrX,'v':arrV,'nondim_stress':arrStressd, 'nondim_strainrate':arrSRd}

        return sol



