import sys
import numpy as np
import matplotlib.pyplot as plt   

#drawing a line of given slope and intercept
def line_draw(slope, intercept):
    axes = plt.gca()
    x = np.array(axes.get_xlim())
    y = slope * x + intercept
    plt.xlabel('x --->')
    plt.ylabel('y --->')
    plt.plot(x, y, 'r-',label = "Line by Normal equation(unweighted)")
    plt.legend(loc="lower right")
    plt.show()
    
#predicting value at given point by weighted linear regression
def prediction_at_x(query_x,tou,X,Y):
    x = np.array([1,query_x])
    w = x - X 
    norm_vector = np.linalg.norm(w,axis=1)
    norm_sq= np.square(norm_vector)

    #weights associated with each point
    W = np.exp(-norm_sq/(2*tou*tou))
    W = np.diag(W)
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
    
    #prediction is y = mx+c
    prediction_y = theta[1]*query_x +theta[0]
    return prediction_y
    
def unweighted_linear_regression(X,Y):
    theta = np.linalg.inv( X.T @ X) @ X.T @ Y
    print("parameter in unweighted linear regression: ",theta)
    plt.scatter(X.T[1],Y,marker='o',color='g',label = "Data Points")
    plt.legend(loc="upper right")
    line_draw(theta[1],theta[0])
    plt.show()
    
def locally_weighted_regression(X,Y,tou):
    x_vals = np.linspace(-5,13,50)
    length = len(x_vals)
    y_vals = np.zeros(length)
    for i in range(length):
        y_vals[i] = prediction_at_x(x_vals[i],tou,X,Y)
    plt.figure(2) 
    plt.scatter(X.T[1],Y,marker='o',color='g',label = "Data Points")
    plt.legend(loc="upper right")
    plt.xlabel('x --->')
    plt.ylabel('y --->')
    plt.plot(x_vals,y_vals,color='r',label = "Weighted Linear regression at tou= "+str(tou))
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    #reading the data from files
    input_fileX = sys.argv[1]
    input_fileY = sys.argv[2]
    tau = float(sys.argv[3])
    X_data = np.loadtxt(input_fileX)
    Y_data = np.loadtxt(input_fileY)

    #adding X0 = 1
    array_of_ones = np.ones(len(X_data))
    X = np.stack((array_of_ones,X_data),axis=-1)
    unweighted_linear_regression(X,Y_data)
   
    #prediction of locally weighted regression at different points
    locally_weighted_regression(X,Y_data,tau)

