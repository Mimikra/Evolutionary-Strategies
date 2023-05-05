import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Specimen:
    __slots__ = ("param", "sigma")

    def __init__(self, param: np.ndarray, sigma: np.ndarray) -> None:
        self.param: np.ndarray[float] = param
        self.sigma: np.ndarray[float] = sigma

    def mutate(self):
        r = [np.random.normal(0, s) for s in self.sigma]
        param = self.param + r
        return Specimen(param, self.sigma)

    def evaluate(self, data: pd.DataFrame) -> float:
        a, b, c = self.param
        o = a*(np.square(data['X'])-b*np.cos(c*np.pi*data['X']))
        mse = (np.sum(np.square(data['Y']-o)))/len(data['Y'])
        return mse
    
    def __str__(self) -> str:
        a,b,c = self.param
        return f"a:{a} b:{b} c:{c}"
    
    def plotdata(self, data: pd.DataFrame) -> None:
        a, b, c = self.param
        o = a*(np.square(data['X'])-b*np.cos(c*np.pi*data['X']))
        plt.plot(data['X'], o, 'b', label = "Approximation")
        plt.plot(data['X'], data['Y'], '--r', label = "Original")
        plt.legend()
        plt.show()


class EvoStrategy:
    def __init__(self, maxiter: int, mselim: float, popsize: int, child_coef: int, datafilepath: str, approach: str = "mi+lam") -> None:
        self.maxiter: int = maxiter
        self.mselim: float = mselim
        self.popsize: int = popsize
        self.child_coef: int = child_coef
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
        self.children = children

    def evaluate_population(self):
        if self.approach == "mi+lam":
            population: list[Specimen] = self.population
            population.extend(self.children)
        elif self.approach == "mi,lam":
            population: list[Specimen] = self.children
        pop_eval = []
        for pop in population:
            pop_eval.append(pop.evaluate(self.data))
        population = [pop for pop, _ in sorted(zip(population, pop_eval), key = lambda x: x[1])]
        self.population = population[:self.popsize]

    def mainloop(self):
        self.pop_init()
        self.children.extend(self.population)
        self.evaluate_population()
        best_individual = self.population[0]
        i = 0
        while i < self.maxiter:
            if best_individual.evaluate(self.data) < self.mselim:
                break
            self.child_init()
            self.evaluate_population()
            best_individual = self.population[0]
            print(f"Iteration {i+1}")
            i+=1
        print(best_individual ,f"MSE:{best_individual.evaluate(self.data)}")
        best_individual.plotdata(self.data)
        return best_individual

if __name__ == "__main__":
    Es = EvoStrategy(50, 0.3, 500, 5, "model5.txt", 'mi,lam')
    Es.mainloop()