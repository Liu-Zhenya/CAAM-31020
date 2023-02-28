import numpy as np

def norm(x):
    return np.linalg.norm(x)

def Q(x,mu):
    x_1 = x[0]
    x_2 = x[1]
    return x_1 + x_2 + (mu/2)*(x_1**2 + x_2**2 -2)**2

def gradient(x,mu):
    x_1 = x[0]
    x_2 = x[1]
    y_1 = 1 + 2*mu*x_1*(x_1**2 + x_2**2 -2)
    y_2 = 1 + 2*mu*x_2*(x_1**2 + x_2**2 -2)
    return np.array([y_1, y_2])


def step_length(x,p,mu):
    alpha = 1
    rho = 0.8
    c_1 = 0.6
    while(Q(x + alpha*p,mu) > (Q(x,mu) + c_1*alpha*np.dot(gradient(x,mu),p))):
        alpha = rho*alpha
    return alpha



def LineSearch(x_0,mu,tao):
    

    
    x = x_0
    while(norm(gradient(x,mu)) > tao):
        p = (-1)*gradient(x,mu)
        alpha = step_length(x,p,mu)
        x = x + alpha*p
    return x

def Quatratic_Penalty(mu_list):
    x = np.zeros(2)
    for mu in mu_list:
        tao = 1/mu
        x = LineSearch(x,mu,tao)
        print(f"{x} is the aprroximated minimizer for Q when mu is {mu}")


mu_list = [1,10,100,1000]
Quatratic_Penalty(mu_list)