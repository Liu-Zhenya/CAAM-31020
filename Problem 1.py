
import scipy
import numpy as np
import matplotlib.pyplot as plt

# define Hessian matrix for Rosenbrock
def Hessian(x):
    n = int(len(x))
    H = np.zeros((n,n))

    H[0,0] = 1200*(x[0]**2)-400*x[1]+2
    H[0,1] = -400*(x[0])

    for i in range(1,n-1):

        H[i,i-1] = -400*x[i-1]
        H[i,i] = 1200*(x[i]**2)-400*x[i+1]+202
        H[i,i+1] = -400*x[i]
    
    H[n-1,n-2] = -400*x[n-2]
    H[n-1,n-1] = 200

    return H


# define Gradient for Rosenbrock

def Gradient_Rosenbrock(x):
    n = int(len(x))
    G = np.zeros(n)
    G[0] = 400*(x[0]**3)-400*x[0]*x[1]+2*x[0]-2
    for i in range(1,n-1):
        G[i] = 200*x[i] - 200*(x[i-1]**2) + 400*(x[i]**3) - 400*x[i]*x[i+1] + 2*x[i] - 2

    G[n-1] = 200*(x[n-1] - (x[n-2]**2))

    return G

# define function value of Rosenbrock

def f_Rosenbrock(x):
    f = 0
    n = int(len(x))

    for i in range(n-1):
        f = f + 100*(x[i+1] - (x[0]**2))**2 + (x[i] - 1)**2
    

    return f


# combine above parameters

def Rosenbrock(x):
    return f_Rosenbrock(x),Gradient_Rosenbrock(x),Hessian(x)

# Backtracking
def step_length_Rosenbrock(f_x,g,X,p):
    alpha = 1
    rho = 0.8
    c_1 = 0.5
    x_new = X + alpha*p
    
    f_x_new = f_Rosenbrock(x_new)
    
    special_count = 1
    
    while(f_x_new > (f_x + c_1*alpha*np.dot(g,p))):
        alpha = rho*alpha
        x_new = X + alpha*p
        f_x_new = f_Rosenbrock(x_new)
        special_count = special_count + 1

        if(special_count == 6):
            alpha = 1
            return alpha,special_count
    return alpha,special_count

def newton_ls_cg(x_in,tol,iter):
    X = x_in
    X_old = x_in
    n = 0
    f_x,g,B = Rosenbrock(X)
    number_evaluate = 0
    while(n < iter and np.linalg.norm(g) > tol):
        f_x,g,B = Rosenbrock(X)

        p = 0

        tol_k = np.minimum(0.5,np.sqrt(np.linalg.norm(g)))*np.linalg.norm(g)
        z = 0
        r = g
        d = -r

        j = 0
        while(True):
            if(j > 20):
                p = -g
                break
            if(d.T @ B @ d <= 0):
                if(j == 0):
                    p = -g
                    break
                else:
                    p = z
                    break
            alpha = np.matmul(r,r)/(d.T @ B @ d)
            z = z + alpha*d
            r_new = r + alpha*(B @ d)
            if (np.linalg.norm(r_new) < tol_k):
                p = z
                break
            beta = np.matmul(r_new,r_new)/np.matmul(r,r)
            d = -r_new + beta*d
            j = j + 1

        n = n + 1
            
        alpha_ls,counts = step_length_Rosenbrock(f_x,g,X,p)
        X_old = X
        X = X + alpha_ls*p
        
        number_evaluate = number_evaluate + 3 + counts
    
    converge = 1
    if(np.linalg.norm(X_old - X) > 0.01):
        converge = 0

    return X,number_evaluate,converge





evaluate_list= np.zeros(170)
whether_converge_list = np.zeros(170)

x_list = np.arange(30,200)

for i in x_list:
    x = 2*np.ones(i)
    X,evaluate_list[i-30],whether_converge_list[i-30] = newton_ls_cg(x,10**(-3),200)


def newton_ls_cg_b(x_in,tol,iter):
    X = x_in
    X_old = x_in
    n = 0
    f_x,g,B = Rosenbrock(X)
    g_old = g
    while(n < iter and np.linalg.norm(g) > tol):
        f_x,g,B = Rosenbrock(X)

        p = 0

        tol_k = np.minimum(0.5,np.sqrt(np.linalg.norm(g)))*np.linalg.norm(g)
        z = 0
        r = g
        d = -r

        j = 0
        while(True):
            if(j > 20):
                p = -g
                break
            if(d.T @ B @ d <= 0):
                if(j == 0):
                    p = -g
                    break
                else:
                    p = z
                    break
            alpha = np.matmul(r,r)/(d.T @ B @ d)
            z = z + alpha*d
            r_new = r + alpha*(B @ d)
            if (np.linalg.norm(r_new) < tol_k):
                p = z
                break
            beta = np.matmul(r_new,r_new)/np.matmul(r,r)
            d = -r_new + beta*d
            j = j + 1

        n = n + 1
            
        alpha_ls,counts = step_length_Rosenbrock(f_x,g,X,p)
        g_old = Gradient_Rosenbrock(X_old)
        X_old = X
        X = X + alpha_ls*p

    measure = np.linalg.norm(g)/np.linalg.norm(g_old)
    return measure

