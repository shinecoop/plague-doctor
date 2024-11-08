import numpy as np 
import matplotlib.pyplot as plt

def generate_random_data(size):
    return np.random.rand(size)*10  

def gaussian_kernel(u):
    #gaussian kernel function
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    

def get_kde_value(data, x,bw=1):

    n=len(data)
    k=np.sum(gaussian_kernel((data-x)/bw)) #KDE equation 
    return k/(n*bw)

def plot_kde(data):
    
    p = np.array([])
    for x in range(-1,20): #finding the density function over a given x range
        p=np.append(p,get_kde_value(data,x,bw=1.65)*10)

    plt.plot(p)#fhat(x)
    pp = np.gradient(p)

    plt.plot(pp)#d/dx(fhat(x)) --> for determing local mins
    plt.show()
    

    
def main():
    data = generate_random_data(10)
    plt.hist(data)
    plot_kde(data)

if __name__ == "__main__":
    main()
