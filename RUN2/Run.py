import numpy as np
from initialization import *
from RungeKutta import *

import numpy as np


def RUN(n, MaxIt, lb, ub, dim, fobj):
    Cost = np.zeros(n)
    X = initialization(n, dim, ub, lb)
    Xnew2 = np.zeros(dim)
    Convergence_curve = np.zeros(MaxIt)

    for i in range(n):
        Cost[i] = fobj(X[i])

    Best_Cost, ind = mymin(Cost)
    Best_X = X[ind]
    Convergence_curve[0] = Best_Cost

    it = 0
    while it < MaxIt-1:
        it += 1
        f = 20 * np.exp(-12 * ((it+1) / MaxIt))
        Xavg = np.mean(X, axis=0)
        SF = 2 * (0.5 - np.random.rand(n)) * f

        for i in range(n):
            _, ind_l = mymin(Cost)
            lBest = X[ind_l]
            A, B, C = RndX(n, i)
            _, ind1 = mymin(Cost[[A, B, C]])
            gama = np.random.rand(dim) * (X[i] - np.random.rand(dim) * (ub - lb)) * np.exp(-4 * (it+1) / MaxIt)
            Stp = np.random.rand(dim) * ((Best_X - np.random.rand(dim) * Xavg) + gama)
            DelX = 2 * np.random.rand(dim) * np.abs(Stp)

            if Cost[i] < Cost[ind1]:
                Xb = X[i]
                Xw = X[ind1]
            else:
                Xb = X[ind1]
                Xw = X[i]

            SM = RungeKutta(Xb, Xw, DelX)
            L = np.random.rand(dim) < 0.5
            Xc = L * X[i] + (1 - L) * X[A]
            Xm = L * Best_X + (1 - L) * lBest
            vec = np.array([1, -1])
            flag = np.asarray(np.floor(2 * np.random.rand(dim)),dtype=int)
            r = vec[flag]
            g = 2 * np.random.rand()
            mu = 0.5 + 0.1 * np.random.randn(dim)

            if np.random.rand() < 0.5:
                Xnew = (Xc + r * SF[i] * g * Xc) + SF[i] * SM + mu * (Xm - Xc)
            else:
                Xnew = (Xm + r * SF[i] * g * Xm) + SF[i] * SM + mu * (X[A] - X[B])

            FU = Xnew > ub
            FL = Xnew < lb
            Xnew = (Xnew * ~(FU + FL)) + ub * FU + lb * FL
            CostNew = fobj(Xnew)

            if CostNew < Cost[i]:
                X[i] = Xnew
                Cost[i] = CostNew

            if np.random.rand() < 0.5:
                EXP = np.exp(-5 * np.random.rand() * it / MaxIt)
                r = np.floor(Unifrnd(-1, 2, 1, 1))
                u = 2 * np.random.rand(dim)
                w = Unifrnd(0, 2, 1, dim) * EXP
                [A, B, C] = RndX(n, i)
                Xavg = (X[A] + X[B] + X[C]) / 3
                beta = np.random.rand(dim)
                Xnew1 = beta * Best_X + (1 - beta) * Xavg

                for j in range(dim):
                    if w[j] < 1:
                        Xnew2[j] = Xnew1[j] + r * w[j] * np.abs((Xnew1[j] - Xavg[j]) + np.random.randn())
                    else:
                        Xnew2[j] = (Xnew1[j] - Xavg[j]) + r * w[j] * np.abs(
                            (u[j] * Xnew1[j] - Xavg[j]) + np.random.randn())

                FU = Xnew2 > ub
                FL = Xnew2 < lb
                Xnew2 = Xnew2 * ~(FU + FL) + ub * FU + lb * FL
                CostNew = fobj(Xnew2)

                if CostNew < Cost[i]:
                    X[i] = Xnew2
                    Cost[i] = CostNew

                else:
                    if np.random.rand() < w[np.random.randint(dim)]:
                        SM = RungeKutta(X[i], Xnew2, DelX)
                        Xnew = (Xnew2 - np.random.rand() * Xnew2) + SF[i] * (
                                    SM + 2 * np.random.rand(dim) * Best_X - Xnew2)
                        FU = Xnew > ub
                        FL = Xnew < lb
                        Xnew = Xnew * ~(FU + FL) + ub * FU + lb * FL
                        CostNew = fobj(Xnew)

                        if CostNew < Cost[i]:
                            X[i] = Xnew
                            Cost[i] = CostNew

            if Cost[i] < Best_Cost:
                Best_X = X[i]
                Best_Cost = Cost[i]

        Convergence_curve[it] = Best_Cost
        print(f"it: {it+1}, Best Cost = {Convergence_curve[it]}")
    return [Best_Cost,Best_X,Convergence_curve]


def Unifrnd(a, b, c, dim):
    a2 = a / 2
    b2 = b / 2
    mu = a2 + b2
    sig = b2 - a2
    if c == 1:
        z = mu + sig * (2 * np.random.rand(dim) - 1)
    else:
        z = mu + sig * (2 * np.random.rand(c, dim) - 1)
    return z


def RndX(nP, i):
    Qi = np.random.permutation(nP)
    Qi = Qi[Qi != i]
    A, B, C = Qi[0], Qi[1], Qi[2]
    return A, B, C
def mymin(arr):
    index = 0
    min = arr[0]
    for i, a in enumerate(arr):
        if a < min:
            min = a
            index = i
    return [min, index]