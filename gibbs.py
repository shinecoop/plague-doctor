import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import odeint

#vairables for our normal dist
mu = [0,0]
cov = [[1,0.5],[0.5,1]]


#SIR model

def sir_model(y, t, beta, gamma):

    S, I, R = y
    N=S+I+R
    
    #define differential equations
    dSdt = -beta *S * I/N
    dIdt = beta * S * I/N -gamma*I
    dRdt = gamma*I

    return [dSdt, dIdt, dRdt]

#SIR simulation

def sir_simulation(beta, gamma):

    S0, I0, R0 = 999,1,0  #initial conditions
    timespace = np.linspace(0,100,100) #timeline

    result = odeint(sir_model, y0=y0, t=timespace, args=(beta, gamma))
    return result[:,1] #only returning infected result

#define priors

def prior(parameter):
    if 0<parameter<1:
        return 0
    else: 
        return -np.inf

#define likelihood functions (assuming Gaussian Noise)
def loglikelihood(observed, beta, gamma):

    simulated = sir_simulation(beta=beta, gamma=gamma)
    ll = -0.5 * np.sum((observed-simulated)**2)


    return ll
        
    

def gibbs(init, samples, observed, y0):
    
    r=[]
    
    b0,g0 = init #Initial conditions

    b,g = b0, g0
    
    for i in range(samples):
        #sampling new beta

        llCurrent = loglikelihood(observed, b,g)
        lpriorCurrent= prior(b)

        #propose new beta
        b1 = np.random.normal(b, 0.1) #sus

        llNew = loglikelihood(observed, b1,g)
        lpriorNew= prior(b1)


        if llNew + lpriorNew > llCurrent + lpriorCurrent:
            print(f'BETA: {b} --> {b1}')
            b=b1 #accept new beta

        #sampling new gamma

        llCurrent = loglikelihood(observed, b,g)
        lpriorCurrent= prior(g)

        #propose new beta
        g1 = np.random.normal(g, 0.1) #sus

        llNew = loglikelihood(observed, b,g1)
        lpriorNew= prior(g1)

        if llNew + lpriorNew > llCurrent + lpriorCurrent:
            print(f'GAMMA: {g} --> {g1}')

            g=g1 #accept new beta


        r.append([b,g])


    return np.array(r[int(len(r)*.25):]) #removing burnin



#plotting
beta=0.3 #true beta
gamma=0.1 #true gamma
N=1000
y0= [0.99*N,0.1*N,0*N]

observed=sir_simulation(beta, gamma) 
observed+= np.random.normal(1,5, observed.shape)
plt.scatter(np.linspace(0,100,100),observed) #"random" observed data

#gibbs simulated data
simulated=gibbs(init=[0.2,0.2], samples=10000, observed=observed, y0=y0)
simBeta=np.mean(simulated[:,0])
simGamma=np.mean(simulated[:,1])

print(f'beta={np.mean(simulated[:,0])} || gamma={np.mean(simulated[:,1])}')

plt.plot(sir_simulation(simBeta,simGamma), color='red')

plt.show()
