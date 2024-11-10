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
    return np.random.normal(x, 1) #draws a random candidate from a normal distribution centered at previous state

def generate_acceptance(x):
    return 0

def change_states(acceptance):
    threshold = 0.5

    if acceptance>threshold:
        return True
    return False

def metropolis(start):

    #use mean of starting distribution as initial state
    x0=50
    plt.plot(50,0,'x')
    plt.plot(generate_candidate(x0),0,'x')
     
    #1. generate random candidate based on previous state
    #2. check acceptance probability
    #3. move/not move states
    #4. repeat until stationary distribution is found (if convergence has been met)

    return 0


def main():
    
    x=np.linspace(50-75, 50+75,100)

    start=norm.pdf(x,50,25)*4 #our starting normal distribution
    print(np.random.choice(start, size=1))
    goal = generate_random_dist()#our goal distribution, this is not known at the start 
    
    #print(np.trapezoid(d))

    metropolis(start)

    plt.plot(goal)
    plt.plot(start)
    plt.show()

if __name__ == "__main__":

    main()
