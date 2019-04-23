import sys
import numpy as np
import matplotlib.pyplot as plt   
import mpl_toolkits.mplot3d as Axes3D
import matplotlib.animation as animation

#drawing a line of given slope and intercept
def line_draw(slope, intercept):
    axes = plt.gca()
    x = np.array(axes.get_xlim())
    y = slope * x + intercept
    plt.xlabel('Acidity of wine --->')
    plt.ylabel('Density of wine --->')
    #plt.axis([-3, 5, 0.985, 1.01])
    plt.plot(x, y, 'r-',label = "Least squares linear regression line")
    plt.legend(loc="upper right")
    plt.show()
    
#used for creating lines at each phase of animation    
def update_lines(num, lines, data, threeD):
    lines.set_data(data[0:2, :num])
    if(threeD):
        lines.set_3d_properties(data[2, :num])
    lines.set_marker("o")
    lines.set_markersize(5)
    return lines

#plot the 3D mesh representing error J(0)
def plot_3D_mesh(theta,points,X,Y,m,time_gap):   
    #set 3D configuration
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')
    #ax.set_xlim(-5,5)
    #ax.set_ylim(-5,5)
    #ax.set_zlim(0,12)
    plt.xlabel('theta[0]')
    plt.ylabel('theta[1]')
    ax.set_zlabel('Cost Function J(theta)')
    
    #construct 2D set of points near the actual parameters
    x = np.linspace(theta[0]-3.0,theta[0]+3.0,100)
    y = np.linspace(theta[1]-3.0,theta[1]+3.0,100)
    theta0_grid,theta1_grid = np.meshgrid(x,y)
    theta_2D = np.stack((theta0_grid,theta1_grid),axis=-1)

    p,q,d = theta_2D.shape
    Z = np.zeros((p,q))

    #computing J(0) at each theta in theta_2D
    for i in range(p):
        for j in range(q):
            temp = X @ (theta_2D[i][j]) - Y
            Z[i][j] = (temp.T @ temp)/(2*m)
    
    #plotting 3D mesh
    surf = ax.plot_surface(theta0_grid, theta1_grid, Z,alpha = 0.8,color='y')
    
    # Lines to plot in 3D
    data = np.array([points.T[0],points.T[1],points.T[2]]) 
    lines, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1],c='r',markersize=0.5)
    
    line_anim = animation.FuncAnimation(fig, update_lines, frames=data.shape[1], fargs=(lines,data,True),interval=time_gap, repeat=True)
    plt.show()
    return line_anim,theta0_grid,theta1_grid,Z

#plotting contours
def plot_contour(points,theta0_grid,theta1_grid,Z,time_gap):
    fig = plt.figure(figsize=(7,7))
    ax = plt.gca()
    #ax.set_xlim(-5,5)
    #ax.set_ylim(-5,5)
    plt.xlabel('theta[0]')
    plt.ylabel('theta[1]')
    
    #Plotting contours of J(0)
    plt.contour(theta0_grid, theta1_grid, Z)

    data = np.array([points.T[0],points.T[1]])
    
    # Lines to plot in 2D
    lines, = plt.plot(data[0, 0:1], data[1, 0:1],c='r',markersize=0.5,label = "Values of J(theta) at different iterations")
    plt.legend(loc="upper right")
    
    line_anim = animation.FuncAnimation(fig, update_lines, frames=data.shape[1], fargs=(lines,data,False),interval=time_gap, repeat=True)
    plt.show()
    return line_anim

#normalizing the data
def standardize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X- mean)/std
    return X

#defines the condition of convergence
def convergence_criteria(X,Y,theta,m,epsilon):
    J_dash =( (X @ theta - Y) @ X ) / m
    for elem in J_dash:
        if np.abs(elem) > epsilon:
            return False
    return True

#gradient descent appraoch to get parameters
def linear_regression(x,Y,eta,time_gap):
    m = len(x)
    
    #standardize the data
    x = standardize(x)
    
    #setting the values of threshold for convergence
    epsilon = 0.00000000001 #10^-11
    
    #plotting points
    plt.scatter(x,Y,c='g',marker='o',label = "Data points")
    plt.legend(loc="upper right")
    
    #adding X0 = 1
    array_of_ones = np.ones(m)
    X = np.stack((array_of_ones,x),axis=-1)
    
    #initial value of theta
    theta = np.array([-1,2])
    points = [[]*3]
    
    #empty the points array
    points.pop()
    
    converged = False
    iterations = 0
    #repeat until it converges
    while not converged:
        iterations = iterations + 1
        error = X @ theta - Y
        J_theta = (error.T @ error)/(2*m)
        
        #constructing set of (x,y,z) coordinates where z is the error J(0)
        #and x and y are parameters theta0 and theta1
        points.append([theta[0],theta[1],J_theta])
        
        #gradient descent
        theta = theta - eta*(error @ X)/ m
        
        #check if converged
        converged = convergence_criteria(X,Y,theta,m,epsilon)
     
    points = np.array(points)
    
    #plot the line of linear regression
    print("Parameter theta = ", theta)
    print("No of iterations taken to converge = ",iterations)
    
    line_draw(theta[1],theta[0])
    anim1,theta0_grid,theta1_grid,Z = plot_3D_mesh(theta,points,X,Y,m,time_gap)
    anim2 = plot_contour(points,theta0_grid,theta1_grid,Z,time_gap)
    return anim1,anim2

if __name__ == '__main__':
    #reading the data from files
    input_fileX = sys.argv[1]
    input_fileY = sys.argv[2]
    learning_rate = float(sys.argv[3])
    time_gap = 1000.0*float(sys.argv[4])  #convering seconds to milliseconds
    
    X_data = np.loadtxt(input_fileX)
    Y_data = np.loadtxt(input_fileY)
    anim1,anim2=linear_regression(X_data,Y_data,learning_rate,time_gap)

