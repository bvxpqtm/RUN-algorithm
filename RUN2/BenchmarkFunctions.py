import numpy as np
def BenchmarkFunctions(F):
    D = 30
    if F == 'F1':
        fobj = F1
        lb = -100
        ub = 100
        dim = D
    elif F == 'F2':
        fobj = F2
        lb = -100
        ub = 100
        dim = D
    elif F == 'F3':
        fobj = F3
        lb = -100
        ub = 100
        dim = D
    elif F == 'F4':
        fobj = F4
        lb = -100
        ub = 100
        dim = D
    elif F == 'F5':
        fobj = F5
        lb = -100
        ub = 100
        dim = D
    elif F == 'F6':
        fobj = F6
        lb = -100
        ub = 100
        dim = D
    elif F == 'F7':
        fobj = F7
        lb = -100
        ub = 100
        dim = D
    elif F == 'F8':
        fobj = F8
        lb = -100
        ub = 100
        dim = D
    elif F == 'F9':
        fobj = F9
        lb = -100
        ub = 100
        dim = D
    elif F == 'F10':
        fobj = F10
        lb = -32.768
        ub = 32.768
        dim = D
    elif F == 'F11':
        fobj = F11
        lb = -100
        ub = 100
        dim = D
    elif F == 'F12':
        fobj = F12
        lb = -100
        ub = 100
        dim = D
    elif F == 'F13':
        fobj = F13
        lb = -600
        ub = 600
        dim = D
    elif F == 'F14':
        fobj = F14
        lb = -50
        ub = 50
        dim = D

    return lb, ub, dim, fobj


def F1(x):
    D = x.shape[0]
    z = x[0]**2 + 10**6 * np.sum(x[1:D]**2)
    return z

def F2(x):
    D = len(x)
    f = np.array([abs(x[i])**(i+1) for i in range(D)])
    z = np.sum(f)
    return z

def F3(x):
    z = np.sum(x**2) + (np.sum(0.5*x))**2 + (np.sum(0.5*x))**4
    return z

def F4(x):
    D = len(x)
    ff = np.array([100*(x[i]**2 - x[i+1])**2 + (x[i] - 1)**2 for i in range(D-1)])
    z = np.sum(ff)
    return z

def F5(x):
    D = len(x)
    z = 10**6 * x[0]**2 + np.sum(x[1:D]**2)
    return z

def F6(x):
    D = len(x)
    f = np.array([((10**6)**((i-1)/(D-1))) * x[i]**2 for i in range(D)])
    z = np.sum(f)
    return z

def F7(x):
    D = len(x)
    f = np.zeros(D)
    for i in range(D):
        if i == D-1:
            f[i] = 0.5 + (np.sin(np.sqrt(x[i]**2 + x[0]**2))**2 - 0.5) / (1 + 0.001 * (x[i]**2 + x[0]**2)**2)
        else:
            f[i] = 0.5 + (np.sin(np.sqrt(x[i]**2 + x[i+1]**2))**2 - 0.5) / (1 + 0.001 * (x[i]**2 + x[i+1]**2)**2)
    z = np.sum(f)
    return z


def F8(x):
    D = len(x)
    w = np.zeros(D - 1)
    f = np.zeros(D - 1)

    for i in range(D - 1):
        w[i] = 1 + (x[i] - 1) / 4
        f[i] = (w[i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[i] + 1) ** 2)

    w[-1] = 1 + (x[-1] - 1) / 4
    z = np.sin(np.pi * w[0]) ** 2 + np.sum(f) + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)

    return z


def F9(x):
    D = len(x)
    f = np.zeros(D)

    for i in range(D):
        y = x[i] + 4.209687462275036e+002
        if abs(y) < 500:
            f[i] = y * np.sin(abs(y) ** 0.5)
        elif y > 500:
            f[i] = (500 - y % 500) * np.sin(np.sqrt(abs(500 - y % 500))) - (y - 500) ** 2 / (10000 * D)
        elif y < -500:
            f[i] = (abs(y) % 500 - 500) * np.sin(np.sqrt(abs(abs(y) % 500 - 500))) - (y + 500) ** 2 / (10000 * D)

            z = 418.9829 * D - np.sum(f)

    return z


def F10(x):
    D = len(x)
    z = -20 * np.exp(-0.2 * ((1 / D) * np.sum(x ** 2)) ** 0.5) - np.exp(
        1 / D * np.sum(np.cos(2 * np.pi * x))) + 20 + np.exp(1)

    return z


def F11(x):
    D = len(x)
    x = x + 0.5
    a = 0.5
    b = 3
    kmax = 20
    c1 = a ** (np.arange(0, kmax + 1))
    c2 = 2 * np.pi * b ** (np.arange(0, kmax + 1))

    f = 0
    c = -w(0.5, c1, c2)

    for i in range(D):
        f = f + w(x[i], c1, c2)

    z = f + c * D

    return z


def w(x, c1, c2):
    y = np.zeros(len(x))

    for k in range(len(x)):
        y[k] = np.sum(c1 * np.cos(c2 * x[k]))

    return y


def F12(x):
    D = len(x)
    z = (np.abs(np.sum(x ** 2) - D)) ** (1 / 4) + (0.5 * np.sum(x ** 2) + np.sum(x)) / D + 0.5

    return z


def F13(x):
    dim = len(x)
    z = np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1)))) + 1

    return z


def F14(x):
    dim = len(x)
    z = (np.pi / dim) * (10 * ((np.sin(np.pi * (1 + (x[0] + 1) / 4))) ** 2) + np.sum((((x[0:dim - 2] + 1) / 4) ** 2) *
        (1 + 10 * ((np.sin(np.pi * (1 + (x[1:dim-1] + 1) / 4)))) ** 2) + ((x[dim - 1] + 1) / 4) ** 2) +np.sum(Ufun(x, 10, 100, 4)))

    return z


def Ufun(x, a, k, m):
    o = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)

    return o
