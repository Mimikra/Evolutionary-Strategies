import numpy as np


class Specimen:
    __slots__ = ("param", "sigma")
    def __init__(self, param: np.ndarray, sigma: np.ndarray) -> None:
        self.param: np.ndarray[float] = param
        self.sigma: np.ndarray[float] = sigma

    def mutate(self):
        r = [np.random.normal(0, s) for s in self.sigma]
        param = self.param + r
        return Specimen(param, self.sigma)
    
    def evaluate(self, i: np.ndarray) -> float:
        a, b, c = self.param
        o = a*(np.square(i)-b*np.cos(c*np.pi*i))
        sse = (np.sum(np.square(i-o)))/len(i)
        return sse
        

class EvoStrategy:
    def __init__(self, maxiter: int, mselim: float, popsize: int, child_coef: int) -> None:
        self.maxiter = maxiter
        self.mselim = mselim
        self.popsize = popsize
        self.chid_coef = child_coef
        self.population = []

    def pop_init(self):
        for i in range(self.popsize):
            param = np.random.uniform(-10, 10, 3)
            sigma = np.random.uniform(0, 10, 3)
            self.population.append(Specimen(param, sigma))     
    