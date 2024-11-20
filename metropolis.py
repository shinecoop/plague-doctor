import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as gkde
from scipy.stats import norm

def generate_random_dist():
    data=np.random.randn(30) #random sampling of data

    kde=gkde(data) #generating a distribution function using kde algorithm

    xs = np.linspace(min(data)-1, max(data) +1, 100) 
    r = kde(xs)

    return r/np.trapezoid(r) #turning distribution function into array of values i can plot

def generate_candidate(x):
    return np.random.normal(x, 8) #draws a random candidate from a normal distribution centered at previous state

def generate_acceptance(x,x1):

    rf = norm.pdf(x1,50,25)/norm.pdf(x,50,25)
    #rg=0 # for asymmetric candidate distributions

    return min(1,rf)

def change_states(acceptance):
    threshold = np.random.uniform(0,1)

    if acceptance>threshold:
        return True
    return False

def metropolis(start,samples):

    count=0
    r=[]

    #use mean of starting distribution as initial state
    x=50
    #1. generate random candidate based on previous state
    #2. check acceptance probability
    #3. move/not move states
    #4. repeat until stationary distribution is found (if convergence has been met)

    for i in range(samples):

        x1 = generate_candidate(x)

        acceptance = generate_acceptance(x,x1)

        cs = change_states(acceptance)

        if(cs):
            x=x1

        count+=1
        r.append(x)

    return r


def main():
    
    x=np.linspace(50-75, 50+75,100)

    start=norm.pdf(x,50,25)*4 #our starting normal distribution
    goal=norm.pdf(x,50,25)*2 #goal, unknown to algorithm
    
    met=metropolis(start,samples=10000)

    plt.plot(goal)
    plt.hist(met, bins=30, density=True)
    plt.plot(start)
    plt.show()

if __name__ == "__main__":

    main()
