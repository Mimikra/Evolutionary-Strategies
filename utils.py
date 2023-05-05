import numpy as np
import pandas as pd


class Specimen:
    __slots__ = ("param", "sigma")

    def __init__(self, param: np.ndarray, sigma: np.ndarray) -> None:
        self.param: np.ndarray[float] = param
        self.sigma: np.ndarray[float] = sigma

    def mutate(self):
        r = [np.random.normal(0, s) for s in self.sigma]
        param = self.param + r
        return Specimen(param, self.sigma)

    def evaluate(self, data: np.ndarray) -> float:
        a, b, c = self.param
        o = a*(np.square(data['X'])-b*np.cos(c*np.pi*data['X']))
        sse = (np.sum(np.square(data['Y']-o)))/len(data['Y'])
        return sse


class EvoStrategy:
    def __init__(self, maxiter: int, mselim: float, popsize: int, child_coef: int, datafilepath: str, approach: str = "mi+lam") -> None:
        self.maxiter: int = maxiter
        self.mselim: float = mselim
        self.popsize: int = popsize
        self.chid_coef: int = child_coef
        self.approach: str = approach
        self.data: pd.DataFrame = self.read_data(datafilepath)
        self.population: list[Specimen] = []
        self.children: list[Specimen] = []

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

    def evaluate_population(self):
        if self.approach == "mi+lam":
            population: list[Specimen] = self.children.extend(self.population)
        elif self.approach == "mi,lam":
            population: list[Specimen] = self.children
        pop_eval = []
        for pop in population:
            pop_eval.append(pop.evaluate(self.data))
        population = [pop for pop, _ in sorted(zip(population, pop_eval), reverse = True, key = lambda x: x[1])]
        self.population = population[:self.popsize]
