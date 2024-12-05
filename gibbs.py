import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import odeint

#reading IRL data

def read_csv(file):

    r=[]

    with open(file, 'r') as f:
        r=(f.read().split('\n'))


    return [item.split(',') for item in r];

def isolate_period(start, end, data):
    if(start>end):
        print("start > end")
        return null

    r=[]

    for file in data[1:len(data)-1]:
        if(int(file[0])>=start and int(file[0])<=end):
            r.append(file)
    return r


def refine(data, loc):

    weeks=[]
    cases=[]

    for file in data:
        if file[1] == loc:
            weeks.append(file[0][:4]+"-"+file[0][4:])
            cases.append(file[4])

    return weeks, [int(c) for c in cases]




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

def sir_simulation(beta, gamma,t=np.linspace(0,53,53) ):

    N=1040000

    result = odeint(sir_model, y0=[N-10, 10,0], t=t, args=(beta, gamma))
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
    ll = -(1/len(observed)) * np.sum((observed-simulated)**2)


    return ll
        
    

def gibbs(init, samples, observed, y0):
    
    r=[]
    
    b0,g0 = init #Initial conditions

    b,g = b0, g0
    
    for i in range(samples):
        #sampling new beta

        llCurrent = loglikelihood(observed, b,g)
        lpriorCurrent= 0

        #propose new beta
        b1 = np.random.normal(b, .1) #sus

        llNew = loglikelihood(observed, b1,g)
        lpriorNew= 0


        if llNew + lpriorNew > llCurrent + lpriorCurrent:
            print(f'BETA: {b} --> {b1}')
            b=b1 #accept new beta

        #sampling new gamma

        llCurrent = loglikelihood(observed, b,g)
        lpriorCurrent= 0

        #propose new beta
        g1 = np.random.normal(g, .1) #sus

        llNew = loglikelihood(observed, b,g1)
        lpriorNew= 0

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

observed=sir_simulation(beta, gamma, t=np.linspace(0,53,53)) 
observed+= np.random.normal(2,6, observed.shape)
plt.scatter(np.linspace(0,53,53),observed) #"random" observed data

#gibbs simulated data
simulated=gibbs(init=[0.2,0.2], samples=10000, observed=observed, y0=y0)
simBeta=np.mean(simulated[:,0])
simGamma=np.mean(simulated[:,1])

print(f'beta={np.mean(simulated[:,0])} || gamma={np.mean(simulated[:,1])}')

plt.plot(sir_simulation(simBeta,simGamma, t=np.linspace(0,100,100)), color='red')
plt.title(f'true beta = {beta} : true gamma = {gamma} \n our beta = {round(simBeta,3)} : our gamma = {round(simGamma,3)}')
plt.xlabel("Time")
plt.ylabel("# of Infected People (I(t))")

plt.show()

'''''

data=read_csv('./disease_data/measles.csv')
period=isolate_period(193001,193101, data)
w,c=refine(period,loc="CO")


N=1040000
simulated=gibbs(init=[9,9], samples=10000, observed=c, y0=[N-10, 10, 0])

plt.scatter(np.linspace(0,53,53),c)
plt.plot(sir_simulation(np.mean(simulated[:,0]), np.mean(simulated[:,1])), color="red")
plt.plot(sir_simulation(8,8), color="red")
plt.title(f'true beta = ?? : true gamma = ?? \n our beta = {round(np.mean(simulated[:,0]),3)} : our gamma = {round(np.mean(simulated[:,1]),3)}')
plt.xlabel("Time")
plt.ylabel("# of Infected People (I(t))")
plt.show()

'''''
