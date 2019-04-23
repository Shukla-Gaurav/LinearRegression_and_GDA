import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt  
import matplotlib.patches as patches

#plotting points
def plot_points(X,Y):
    plt.xlabel('growth ring diameter in fresh water --->')
    plt.ylabel('growth ring diameter in marine water --->')
    plt.scatter(X.T[0],X.T[1],c=Y)
    class_Alaska = patches.Patch(color='midnightblue', label='Alaska')
    class_Canada = patches.Patch(color='yellow', label='Canada')
    legend = plt.legend(loc='upper right', handles=[class_Alaska, class_Canada])
    plt.gca().add_artist(legend)
    
#drawing a line of given slope and intercept
def line_draw(slope, intercept,X,Y,fig_no):
    plt.figure(fig_no) 
    axes = plt.gca()
    x = np.linspace(25,200,20)
    y = slope * x + intercept  
 
    #plot points as well
    plot_points(X,Y)
       
    #plot line
    plt.plot(x, y, 'r-',label = "Linear boundary when co-variances are same")
    legend = plt.legend(loc="upper left")
    axes.add_artist(legend)

def get_mean(X,Y,m):
    #counting unique values in Y and their respective frequency
    unique, counts = np.unique(Y, return_counts=True)

    #phai,avg0,avg1,sigma are parameters of GDA
    phai = counts[1]/m
    avg0 = np.average(X, axis=0, weights = 1-Y)
    avg1 = np.average(X, axis=0, weights = Y)

    #computing (X-mean)
    positioning_mean = np.array([Y,1-Y])
    avg = np.array([avg1,avg0])
    mean = positioning_mean.T @ avg

    dev_from_mean = X-mean
    return dev_from_mean,counts,phai,avg0,avg1 

#Linear seperator if sigma0 = sigma1
def GDA_same_covariance(X,Y,dev_from_mean,phai,avg0,avg1):   
    #computing sigma
    sigma = (dev_from_mean.T @ dev_from_mean)/m

    #print all parameters
    print("phai = ",phai)
    print("mean0 = ",avg0)
    print("mean1 = ", avg1)
   
    #computing slope and intercept for linear seperator
    a = np.log(phai/(1-phai))
    b = (avg1.T @ np.linalg.inv(sigma) @ avg1) - (avg0.T @ np.linalg.inv(sigma) @ avg0)
    c = a-(b/2)
    d = np.linalg.inv(sigma) @ (avg0 - avg1)

    intercept = c/d[1]
    slope = -1*d[0]/d[1]

    return slope,intercept,sigma
    
#Quadratic seperator if sigma0 != sigma1
def GDA_different_covariance(X,Y,dev_from_mean,counts,phai,avg0,avg1,slope,intercept):
    #seperating matrices for different classes in Y
    itemindex0 = np.where(Y==0)
    D0 = dev_from_mean[itemindex0]
    itemindex1 = np.where(Y==1)
    D1 = dev_from_mean[itemindex1]

    #computing parameters sigma0 and sigma1
    sigma0 = (D0.T @ D0)/counts[0]
    sigma1 = (D1.T @ D1)/counts[1]

    #print all parameters
    print("sigma0 = ",sigma0)
    print("sigma1 = ",sigma1)

    #drawing linear boundary for third figure
    line_draw(slope, intercept,X,Y,2)

    #constructing the equation of quadratic seperator
    a = np.log(phai/(1-phai)) + np.log(np.linalg.det(sigma0)/np.linalg.det(sigma1))/2
    b = ((avg1.T @ np.linalg.inv(sigma1) @ avg1)- (avg0.T @ np.linalg.inv(sigma0) @ avg0))/2
    c = b-a
    
    #coefficients of equation
    D = np.linalg.inv(sigma0) @ avg0 - np.linalg.inv(sigma1) @ avg1
    diff_sigma = np.linalg.inv(sigma1)- np.linalg.inv(sigma0)

    A = diff_sigma[0][0]/2    
    B = (diff_sigma[0][1]+diff_sigma[1][0])/2
    C = diff_sigma[1][1]/2

    #plotting the quadratic seperator
    x = np.linspace(25, 200, 100)
    y = np.linspace(50, 600, 100)
    x, y = np.meshgrid(x, y)
    
    #drawing quadratic boundary
    plt.contour(x, y,(A*x**2 + B*x*y + C*y**2 + D[0]*x + D[1]*y + c),[0])
    
if __name__ == '__main__':
    #reading the data from files
    input_fileX = sys.argv[1]
    input_fileY = sys.argv[2]
    choice = int(sys.argv[3])
 
    X_data = np.loadtxt(input_fileX, dtype = np.float)
    Y_data = np.loadtxt(input_fileY,dtype = object)    
    m = Y_data.shape
    
    #replacing Alaska as 0 and Canada as 1
    Y_data[Y_data=='Alaska'] = 0
    Y_data[Y_data=='Canada'] = 1

    #changing datatype object to float
    Y_data = np.array(Y_data, dtype = np.float)
    dev_from_mean,counts,phai,avg0,avg1 = get_mean(X_data,Y_data,m)
    slope,intercept,sigma = GDA_same_covariance(X_data,Y_data,dev_from_mean,phai,avg0,avg1)

    #choose one of them depends on choice
    if choice == 0 :  
        print("sigma = ",sigma)     
        #plot points as well
        plot_points(X_data,Y_data)
        plt.show()
        #plot line 
        line_draw(slope, intercept,X_data,Y_data,2)
        plt.show()
        
    elif choice == 1:
        GDA_different_covariance(X_data,Y_data,dev_from_mean,counts,phai,avg0,avg1,slope,intercept)
        plt.show()
    else:
        print("Wrong choice")


