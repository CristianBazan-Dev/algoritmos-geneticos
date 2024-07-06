# Importación de las librerías para los primeros 3 aspectos
import random 
from deap import base 
from deap import creator 
from deap import tools
from deap import algorithms
import numpy as np
import math

# Creación de los objetos para definir el problema y el tipo de individuo 
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Generación de genes 
toolbox.register("attr_uniform", random.uniform, -100, 100)


# Generación de individuos y población 
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_uniform, 2)

toolbox.register("population", tools.initRepeat, list, toolbox.individual, 30)

individuo = toolbox.individual() 

poblacion = toolbox.population()

# print(individuo.fitness.values)

# Registro de función objetivo 
def funcion_objetivo(x): 
    # Para cada variable independiente 
    for i in range(len(x)):
        # Se evalua si se encuentra dentro de los limites 
        if x[i] > 100 or x[i] < -100: 
            # En caso de no estar dentro de los límites, se descarta la solución 
            return -1, 
    # En caso contrario, se calcula su valor 
    res = math.sqrt(x[0]**2 + x[1]**2)
    return res, 

toolbox.register("evaluate", funcion_objetivo)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma = 5, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


# Generación del main 
def main():
    random.seed(42)
    CXPB, MUTPB, NGEN = 0.5, 0.3, 20
    
    pop = toolbox.population()
    hof = tools.HallfOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitnessvalues)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = CXPB, mutpb= MUTPB, ngen=NGEN,stats=stats, halloffame=hof, verbose=True)
    
    return hof, logbook

if __name__ == '__main__':
    best, log = main() 
    print("Mejor fitness: %f" %best[0].fitness.values)
    print("Mejor individuo: %f" %best[0])
    plot_evolucion(log)
    
    


    