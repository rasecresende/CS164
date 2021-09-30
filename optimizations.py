#from sage.numerical.optimize import minimize
#f(x,y) = x*sin(x*y)
#this applies the conjugate gradient algo automatically
#print(minimize(f, [.1, .2], algorithm="cg"))
#print(minimize(f, [2, 6], algorithm="cg"))
#print(minimize(f, [10, 20], algorithm="cg"))

%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import random
import math

#our function
def f(input_vars):
    x = input_vars[0]
    y = input_vars[1]
    #return -np.exp(-(np.power(x*y-1.5, 2)) - (np.power(y-1.5,2)))
    return np.power((1-x),2) + 5* np.power(y- np.power(x,2),2)

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)

X, Y = np.meshgrid(x, y)
Z = f([X, Y])
#plot the func
plt.contour(X, Y, Z, levels=50, colors='black');

def get_gradient (f, input_vars, epsilon = 10**(-4)):
    result = []

    # x gradient
    input1 = [input_vars[0] - epsilon, input_vars[1]]
    input2 = [input_vars[0] + epsilon, input_vars[1]]
    result.append((f(input1) - f(input2))/ (2*epsilon))

    # y gradient
    input1 = [input_vars[0], input_vars[1] - epsilon]
    input2 = [input_vars[0], input_vars[1] + epsilon]
    result.append((f(input1) - f(input2))/ (2*epsilon))

    #normalize
    magnitude = 0
    for var in result:
        magnitude+= var**2
    magnitude = magnitude**(1/2)

    for i in range(len(result)):
        result[i] = result[i]/magnitude

    return result

def gradient_descent (f, input_vars, step_size):
    gradient = get_gradient(f, input_vars)
    for i in range(len(input_vars)):
        input_vars[i] = input_vars[i] + gradient[i]*step_size
    return input_vars

def noise (dimension, delta):
    result = []
    for i in range(dimension):
        result.append(random.random()*delta)
    return result

def gradient_descent_with_absolute_improvement_termination (f, start_place, step_size = 0.1, epsilon = 10**(-6)):
    global steps_x, steps_y
    steps_x = []
    steps_y = []
    steps_x.append(start_place[0])
    steps_y.append(start_place[1])
    step_count = 0
    improvement = 10
    last_val = f(start_place)
    max_steps = 10000
    while improvement > epsilon:
        step_size_current = step_size/(step_count+1)

        start_place = gradient_descent(f, start_place, step_size_current)
        steps_x.append(start_place[0])
        steps_y.append(start_place[1])
        improvement = abs(last_val -  f(start_place))
        #print(improvement)
        last_val = f(start_place)

        random_element = noise(len(start_place), step_size_current**2)
        for i in range(len(start_place)):
            start_place[i] += random_element[i]
        step_count += 1

        if(step_count > max_steps):
            break

    print("step_count", step_count)
    return "value is " + str(f(start_place)) + " at " + str(start_place)

def polak_ribiere (prev_gradient, current_gradient):

    a = []
    for i in range(len(prev_gradient)):
        a.append(current_gradient[i] - prev_gradient[i])


    return 1

def dot_product (a,b,):


def conjugate_descent_with_absolute_improvement_termination (f, start_place, step_size = 0.1, epsilon = 10**(-6)):
    global steps_x, steps_y
    steps_x = []
    steps_y = []
    steps_x.append(start_place[0])
    steps_y.append(start_place[1])
    step_count = 0
    improvement = 10
    last_val = f(start_place)
    max_steps = 10000
    gradient =  get_gradient(f, start_place)
    while improvement > epsilon:
        step_size_current = step_size/(step_count+1)

        for i in range(len(input_vars)):
            input_vars[i] = input_vars[i] + gradient[i]*step_size_current

        new_gradient = get_gradient(f, start_place)
        beta = polak_ribiere(gradient, new_gradient)
        for i in range(len(gradient)):
            gradient[i] = new_gradient[i] + beta*gradient[i]

        steps_x.append(start_place[0])
        steps_y.append(start_place[1])

        improvement = abs(last_val -  f(start_place))
        #print(improvement)
        last_val = f(start_place)

        #random_element = noise(len(start_place), step_size_current**2)
        #for i in range(len(start_place)):
        #    start_place[i] += random_element[i]
        step_count += 1

        if(step_count > max_steps):
            break

    print("step_count", step_count)
    return "value is " + str(f(start_place)) + " at " + str(start_place)

print("min",gradient_descent_with_absolute_improvement_termination(f, [-1,-2], 1))

plt.plot(steps_x,steps_y)

plt.show()
