import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.xkcd()

def backtracking(f, f_grad, x, d, alpha=1, p=0.8, beta=1e-3):
    y = f(*x)
    wolfe_1 = lambda alpha: f(*(x + alpha * d)) <= y + beta * alpha * f_grad(*x).dot(d)

    while (not wolfe_1(alpha)) and (alpha > 1e-6):
        alpha *= p
    return alpha



def momentum_descent(f, f_grad, x_0, beta, epsilon):
    def step(f, f_grad, x, v, beta):
        g = f_grad(*x) / np.linalg.norm(f_grad(*x))
        alpha = backtracking(f, f_grad, x, -g)
        v = beta * v - g * alpha
        return v

    Xs = [x_0]
    v = 0
    v = step(f, f_grad, Xs[-1], v, beta=beta)
    Xs.append(Xs[-1] + v)
    plt.arrow(*Xs[-2], *(Xs[-1] - Xs[-2]))

    while (Xs[-1] - Xs[-2]).max() > epsilon:
        v = step(f, f_grad, Xs[-1], v, beta=beta)
        Xs.append(Xs[-1] + v)
        plt.arrow(*Xs[-2], *(Xs[-1] - Xs[-2]))
    return Xs

def momentum_descent_original(f, f_grad, x_0, steps, beta):
    def step(f, f_grad, x, v, beta):
        g = f_grad(*x) / np.linalg.norm(f_grad(*x))
        alpha = backtracking(f, f_grad, x, -g)
        v = beta * v - g * alpha
        return v

    Xs = [x_0]
    v = 0
    v = step(f, f_grad, Xs[-1], v, beta=beta)
    Xs.append(Xs[-1] + v)
    plt.arrow(*Xs[-2], *(Xs[-1] - Xs[-2]))

    for i in range(steps):
        v = step(f, f_grad, Xs[-1], v, beta=beta)
        Xs.append(Xs[-1] + v)
        plt.arrow(*Xs[-2], *(Xs[-1] - Xs[-2]))
    return Xs

f = lambda x, y: -np.exp(-(x*y-1.5)**2 -(y-1.5)**2)
f_grad = lambda x, y: np.array([2*(x*y - 1.5)*y*np.e**(-(x*y - 1.5)**2 - (y - 1.5)**2),
                                  (2*(x*y - 1.5)*x + 2*y - 3)*np.e**(-(x*y - 1.5)**2 - (y - 1.5)**2)])
x_0 = np.array((0.5, 0.5))

Xs = np.linspace(0, 2)
Ys = np.linspace(0, 2)
X, Y = np.meshgrid(Xs, Ys)

plt.figure()
plt.contour(X, Y, f(X, Y), levels = np.linspace(-3, 0, 20))
momentum_descent(f, f_grad, x_0, 0.01, 0.001)

plt.figure()
plt.contour(X, Y, f(X, Y), levels = np.linspace(-3, 0, 20))
momentum_descent_original(f, f_grad, x_0, 3, 0.01)
