import numpy as np
import matplotlib.pyplot as plt


def norm(x):
    return np.linalg.norm(x)

def f(x,c):
    x_1 = x[0]
    x_2 = x[1]
    return x_1**2 + c*x_2**2

def gradient(x,c):
    x_1 = x[0]
    x_2 = x[1]
    return np.array([2*x_1, 2*c*x_2])


def step_length(x,p,c):
    alpha = 1
    rho = 0.8
    c_1 = 0.6
    while(f(x + alpha*p,c) > (f(x,c) + c_1*alpha*np.dot(gradient(x,c),p))):
        alpha = rho*alpha
    return alpha



def hw_p2(x_0,c):
    """
    Start at x_0 and given c to decide f(x), this function will return the point with minimal solution
    """

    
    error = np.zeros(30)
    if(c > 500):
        error = np.zeros(300)
    error[0] = norm(x_0)
    yardstick = 10**(-8)
    x = x_0
    x_pre = np.copy(x)
    k = 0
    while(k < 1000 and norm(gradient(x,c)) > yardstick):
        p = (-1)*gradient(x,c)
        alpha = step_length(x,p,c)
        x_pre = np.copy(x)
        x = x + alpha*p
        k = k + 1
        error[k] = norm(x)
    print(gradient(x,c))
    print(gradient(x_pre,c))
    ratio = norm(gradient(x,c))/norm(gradient(x_pre,c))
    return f(x,c),ratio,error
    
# part e
A = np.array([1,1])
B = np.array([-1,1])
C = np.array([-1,-1])
D = np.array([1,-1])

a,converge_A,error_A = hw_p2(A,3)
b,converge_B,error_B = hw_p2(B,3)
c,converge_C,error_C = hw_p2(C,3)
d,converge_D,error_D = hw_p2(D,3)

x_axis = np.arange(30)

plt.semilogy(x_axis,error_A,label='[1,1]')
plt.semilogy(x_axis,error_B,label='[-1,1]')
plt.semilogy(x_axis,error_C,label='[-1,-1]')
plt.semilogy(x_axis,error_D,label='[1,-1]')
plt.xlabel('iteration times')
plt.ylabel('errors')
plt.legend()

# part f

A = np.array([1,1])
B = np.array([-1,1])
C = np.array([-1,-1])
D = np.array([1,-1])

a,converge_A,error_A = hw_p2(A,1000)
b,converge_B,error_B = hw_p2(B,1000)
c,converge_C,error_C = hw_p2(C,1000)
d,converge_D,error_D = hw_p2(D,1000)

x_axis = np.arange(300)

plt.semilogy(x_axis,error_A,label='[1,1]')
plt.semilogy(x_axis,error_B,label='[-1,1]')
plt.semilogy(x_axis,error_C,label='[-1,-1]')
plt.semilogy(x_axis,error_D,label='[1,-1]')
plt.xlabel('iteration times')
plt.ylabel('errors')
plt.legend()
