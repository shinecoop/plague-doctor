import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as gkde
from scipy.stats import norm, uniform, loggamma

def f(x):

    return np.abs((np.sin(5*x) + np.cos(3*x) + (1/(1+x**2))) * np.exp(-x**4))

def generate_candidate(x):
    return np.random.normal(x,1) #generate candidate from normal distribution centered around current

def generate_acceptance(x,x1):
    rf = f(x1)/f(x)
    #rg=0 # for asymmetric candidate distributions

    return min(1,rf)

def change_states(acceptance):
    
    if acceptance>=np.random.uniform(0,1):
        return True
    return False

def metropolis(samples):

    count=0
    r=[]

    x=0 #starting state

    #1. generate random candidate based on previous state
    #2. check acceptance probability
    #3. move/not move states
    #4. repeat until stationary distribution is found (if convergence has been met)

    for i in range(samples):

        s = f"{x} --> "

        x1 = generate_candidate(x)

        acceptance = generate_acceptance(x,x1,)

        cs = change_states(acceptance)

        if(cs):
            x=x1
 
        print(f"{s}{x} : acceptance = {acceptance} : change? {cs} : count = {count}")
        count+=1
        r.append(x)
    
    burn_in = int(0.1*len(r))
    return r[burn_in:]


def main():
    
    #draw target distributions
    x_vals = np.arange(-20,20,.01) 
    y_vals = [f(x) for x in x_vals] #f values

    #values used for verification in the future 
    NORM_CONST=2.12218
    TRUE_MEAN = 0.34085/ NORM_CONST

    p_vals = [f(x)/NORM_CONST for x in x_vals] #pi values
    
    #plotting 
    plt.plot(x_vals, y_vals, color="red")
    plt.plot(x_vals, p_vals, color="blue")

     
    #plot setup
    plt.xlabel('x', fontsize=20)
    plt.ylabel('density', fontsize=20)
    plt.legend(['f(x)','PI(x)'], fontsize=20)
    plt.xlim(-10,10)

    #metropolis
    met=metropolis(samples=100000)
    plt.hist(met, bins=200, density=True, color="orange")
    #verification
    plt.title(f"Expected Mean: {round(TRUE_MEAN,3)} \n Mean We Got: {round(np.mean(met),3)}")


    plt.show()




if __name__ == "__main__":

    main()
