import numpy as np
import pandas as pd
import QuantLib as ql
from matplotlib.pyplot import plot as plt

sigma = 0.1
a = 0.1
timestep = 360
length = 30 # in years
forward_rate = 0.05
day_count = ql.Thirty360()
todays_date = ql.Date(15, 1, 2015)
class Model:

    def __init__(self):

        pass

    def GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta):
        #time step for differentiation
        dt = 0.0001
        f0T = lambda T: -(np.log(P0T(T+dt))-np.log(P0T(T-dt)))/(2*dt)

        #initial interest rate is the forward rate at time t0
        r0 = f0T(T)

        #long term mean
        theta = lambda t: 1/lambd * (f0T(t+dt) - f0T(t-dt))/(2*dt) + f0T(t) + \
            eta*eta/(2*lambd*lambd*(1-np.exp(-2*lambd*t)))

        Z = np.random.normal(0, 1, [NoOfPaths, NoOfSteps])
        W = np.zeros([NoOfPaths, NoOfSteps+1])
        R = np.zeros([NoOfPaths, NoOfSteps+1])
        R[:,0] = r0
        time = np.zeros(([NoOfSteps+1]))

        dt = T/float(NoOfSteps)

        for i in range(0, NoOfSteps):
            if NoOfPaths>1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i]))/np.sd(Z[:,i])
            W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
            R[:,i+1] = R[:,i] + (theta(time[i]) - lambd * R[:,i])*dt + eta*(W[:,i+1] - W[:,i])
            time[i+1] = time[i] + dt

        paths = {"time": time, "R": R}

        return paths

    def mainCalculation(self):
        NoOfPaths = 1
        NoOfSteps = 5000
        T = 50 # years
        lambd = 0.5
        eta = 0.01

        # ZCB curve obtained from market
        P0T = lambda T: np.exp(-0.5*T)

        plt.figure(1)
        legend = []
        lambdVec = [-0.01, 0.2, 5.0]
        for l in lambdVec:
            np.random.seed(2)
            Paths = self.GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta)
            legend.append('lambda={0}'.format(l))
            timeGrid = Paths["time"]
            R = Paths["R"]
            plt.plot(timeGrid, np.transpose(R))
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("R(t)")
        plt.legend(legend)
        







    def forward_curve(self):
        ql.Settings.instance().evaluationDate = todays_date

        spot_curve = ql.FlatForward(todays_date, ql.QuoteHandle(ql.SimpleQuote(forward_rate)), day_count)
        spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)

        hw_process = ql.HullWhiteProcess(spot_curve_handle, a, sigma)


    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
    seq = ql.GaussianPathGenerator(hw_process, length, timestep, rng, False)

    def generate_paths(num_paths, timestep):
        arr = np.zeros((num_paths, timestep+1))
        for i in range(num_paths):
            sample_path = seq.next()
            path = sample_path.value()
            time = [path.time(j) for j in range(len(path))]
            value = [path[j] for j in range(len(path))]
            arr[i, :] = np.array(value)
        return np.array(time), arr

