import numpy as np
from deap import tools, base
from deap.algorithms import eaMuPlusLambda
from deap import creator
import functions
import draw_log


creator.create("BaseFitness", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.BaseFitness)


class SimpleGAExperiment:
    def __init__(self, function, dimension, pop_size, iterations,
                 mutation_probability, crossover_probability):
        self.pop_size = pop_size
        self.iterations = iterations
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability

        self.function = function
        self.dimension = dimension

        self.engine = base.Toolbox()
        self.engine.register("map", map)
        self.engine.register("individual", tools.initIterate,
                             creator.Individual, self.factory)
        self.engine.register("population", tools.initRepeat, list,
                             self.engine.individual, self.pop_size)
        self.engine.register("mate", tools.cxOnePoint)
        self.engine.register("mutate", tools.mutGaussian, mu=0, sigma=0.5,
                             indpb=0.2)
        self.engine.register("select", tools.selTournament, tournsize=4)
        self.engine.register("evaluate", self.function)

        return

    def factory(self):
        return np.random.random(self.dimension) * 10 - 5

    def mutation(self, individual):
        for i in range(len(individual)):
            if np.random.random() < len(individual) * 0.15:
                individual[i] += np.random.normal(0.0, 0.2)
                individual[i] = np.clip(individual[i], -5, 5)

        return individual,

    def run(self):
        pop = self.engine.population()
        hof = tools.HallOfFame(3, np.array_equal)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = eaMuPlusLambda(pop, self.engine,
                                  mu=self.pop_size,
                                  lambda_=int(self.pop_size * 0.8),
                                  cxpb=self.crossover_probability,
                                  mutpb=self.mutation_probability,
                                  ngen=self.iterations,
                                  stats=stats, halloffame=hof, verbose=True)

        print("Best = {}".format(hof[0]))
        print("Best fit = {}".format(hof[0].fitness.values[0]))

        return log


def main():
    dimension = 100
    pop_size = 100
    iterations = 1000

    # Changed for the base experiment
    mutation_probability = 0.6
    crossover_probability = 0.3

    scenario = SimpleGAExperiment(functions.rastrigin, dimension, pop_size,
                                  iterations,  mutation_probability,
                                  crossover_probability)

    log = scenario.run()

    draw_log.draw_log(log, "Original algorithm")

    return


if __name__ == "__main__":
    main()
