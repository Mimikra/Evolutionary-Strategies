import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from time import perf_counter 


class Specimen:
    __slots__ = ("param", "sigma", "mse")
    tau = None
    data = None
    def __init__(self, param: np.ndarray, sigma: np.ndarray) -> None:
        self.param: np.ndarray[float] = param
        self.sigma: np.ndarray[float] = sigma
        self.mse: float = self.evaluate()

    def mutate(self):
        r = [np.random.normal(0, s) for s in self.sigma]
        param = self.param + r
        r1 = np.exp(Specimen.tau[0]*np.random.normal())
        r2 = np.exp(Specimen.tau[1]*np.array([np.random.normal() for i in range(len(self.sigma))]))
        sigma = self.sigma*r2*r1
        return Specimen(param, sigma)

    def evaluate(self) -> float:
        a, b, c = self.param
        data = Specimen.data
        o = a*(np.square(data['X'])-b*np.cos(c*np.pi*data['X']))
        mse = (np.sum(np.square(data['Y']-o)))/len(data['Y'])
        return mse
    
    def __str__(self) -> str:
        a,b,c = self.param
        s1,s2,s3 = self.sigma
        return f"a:{a:.4f} b:{b:.4f} c:{c:.4f} \u03C3a:{s1:.5f} \u03C3b:{s2:.5f} \u03C3c:{s3:.5f}"
    
    def plotdata(self) -> None:
        a, b, c = self.param
        data = Specimen.data
        o = a*(np.square(data['X'])-b*np.cos(c*np.pi*data['X']))
        plt.plot(data['X'], data['Y'], 'k', label = "Original", linewidth = 1.5, alpha = 0.9)
        plt.plot(data['X'], o, 'c', label = "Approximation" ,linewidth = 2, alpha = 0.45)
        plt.legend()
        plt.show()


class EvoStrategy:
    def __init__(self, maxiter: int, mselim: float, popsize: int, child_coef: int, datafilepath: str, approach: str = "mi+lam", chromlen: int = 6) -> None:
        self.maxiter: int = maxiter
        self.mselim: float = mselim
        self.popsize: int = popsize
        self.child_coef: int = child_coef
        self.approach: str = approach
        self.chromlen: int = chromlen
        self.data: pd.DataFrame = self.read_data(datafilepath)
        self.population: list[Specimen] = []
        self.children: list[Specimen] = []
        self.convergence = False
        self.differ = 0

    def read_data(self, path: str) -> pd.DataFrame:
        file = pd.read_csv(path, delim_whitespace=True,
                           header=None, names=['X', 'Y'])
        return file

    def pop_init(self):
        for i in range(self.popsize):
            param = np.random.uniform(-10, 10, 3)
            sigma = np.random.uniform(0, 10, 3)
            self.population.append(Specimen(param, sigma))

    def child_init(self):
        children = []
        for parent in self.population:
            for i in range(self.child_coef):
                children.append(parent.mutate())
        self.children = children

    def check_convergence(self):
        parent_eval = []
        for pop in self.population:
            parent_eval.append(pop.mse)
        best_parent_mse = sorted(parent_eval)
        best_parent_mse = best_parent_mse[0]
        child_eval = []
        for pop in self.children:
            child_eval.append(pop.mse)
        best_child_mse = sorted(child_eval)
        best_child_mse = best_child_mse[0]
        if abs(best_child_mse - best_parent_mse) < self.mselim:
            self.convergence = True
        self.differ = abs(best_child_mse - best_parent_mse)
    
    def evaluate_population(self):
        if self.approach == "mi+lam":
            population: list[Specimen] = self.population
            population.extend(self.children)
        elif self.approach == "mi,lam":
            population: list[Specimen] = self.children
        pop_eval = []
        for pop in population:
            pop_eval.append(pop.mse)
        population = [pop for pop, _ in sorted(zip(population, pop_eval), key = lambda x: x[1])]
        self.population = population[:self.popsize]

    def mainloop(self):
        start = perf_counter()
        Specimen.tau = [1/(sqrt(2*self.chromlen)), (1/sqrt(2*(sqrt(self.chromlen))))]
        Specimen.data = self.data
        self.pop_init()
        self.children.extend(self.population)
        self.evaluate_population()
        best_individual = self.population[0]
        for i in range(self.maxiter):
            self.child_init()
            self.check_convergence()
            if self.convergence:
                break
            self.evaluate_population()
            best_individual = self.population[0]
            print(f"Iteration {i+1 :>2}, MSE: {best_individual.mse:.4f}", best_individual)
            print(f"Difference between best parent and offspring: {self.differ:.7f}\n")
        print(f"Ended on iteration {i+1} MSE:{best_individual.mse:.6f}", best_individual)
        print(f"Difference between best parent and offspring: {self.differ}")
        print(f"time: {perf_counter()-start}") 
        best_individual.plotdata()
        return best_individual

if __name__ == "__main__":
    Es = EvoStrategy(150, 10**-5, 500, 5, "model5.txt", 'mi+lam')
    Es.mainloop()