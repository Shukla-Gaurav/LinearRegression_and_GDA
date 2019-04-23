import sys
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

#drawing a line of given slope and intercept
def line_draw(slope, intercept):
    axes = plt.gca()
    x = np.array(axes.get_xlim())
    y = slope * x + intercept
    plt.xlabel('x1 --->')
    plt.ylabel('x2 --->')
    plt.plot(x, y, 'r-',label = "Linear boundary by logistic regression")
    legend = plt.legend(loc="upper left")
    axes.add_artist(legend)

def has_converged(L_dash):
    epsilon = 0.0000000001  #10^-10
    for l in L_dash:
        if np.abs(l) > epsilon:
            return False
    return True

def logistic_reg_newtons(X,Y):
   #initial value of theta is 0
    theta = [0,0,0]
    
    #repeat until it converges
    while True:  
        converge = True
        g_theta = 1/(1+np.exp(-1*(X @ theta)))
        L_dash =-1*( X.T @ (g_theta - Y))
        
        #exit from loop if convergence happened
        if(has_converged(L_dash)):
            break
            
        w = np.array([i*(1-i) for i in g_theta])
        W = np.diag(w)
        
        #H is hessian matrix
        H = -1*(X.T @ W @ X)
        inv_H = np.linalg.inv(H)

        #update theta using newton's method
        theta = theta - inv_H @ L_dash

    #display the parameters
    print("Parameter theta = ", theta)
    
    #plot points and the seperator
    plt.scatter(X.T[1],X.T[2],c=Y)
    
    slope = -1*theta[1]/theta[2]
    intercept = -1*theta[0]/theta[2]
    line_draw(slope, intercept)
    
    class_l0 = patches.Patch(color='midnightblue', label='Class0')
    class_l1 = patches.Patch(color='yellow', label='Class1')
    plt.legend(loc='upper right', handles=[class_l0, class_l1])
    plt.show()
    
if __name__ == '__main__':
    #reading the data from files
    input_fileX = sys.argv[1]
    input_fileY = sys.argv[2]
    
    X_data = np.loadtxt(input_fileX,delimiter = ',')
    Y_data = np.loadtxt(input_fileY)
    m = len(X_data)

    #adding X0 = 1
    array_of_ones = np.ones(m)
    X = np.stack((array_of_ones,X_data.T[0],X_data.T[1]),axis=-1)
    
    logistic_reg_newtons(X,Y_data)
    

