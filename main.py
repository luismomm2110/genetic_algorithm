import numpy as np
import matplotlib.pyplot as plt
from KnapsackItem import KnapsackItem
import ga


def main():

    # entradas da equação
    saco_de_dormir = KnapsackItem('saco de dormir', weight=15, points=15)
    corda = KnapsackItem('corda', 3, 10)
    canivete = KnapsackItem('canivete', 2, 10)
    tocha = KnapsackItem('tocha', 5, 5)
    garrafa = KnapsackItem('garrafa', 9, 8)
    comida = KnapsackItem('comida', 20, 17)

    itens = [saco_de_dormir, corda, canivete, tocha, garrafa, comida]
    # número de pesos a otimizar
    num_weights = 6

    sol_per_pop = 8

    # população tem sol_per_pop cromossomos com num_weights gens
    pop_size = (sol_per_pop, num_weights)

    # Algoritmo genético
    num_generations = 100
    num_parents_mating = 4

    # geracao que foi atingida o numero maximo (43)
    TRIALS = 100
    reached_max_value = []

    for _ in range(TRIALS):
        # População inicial
        new_population = np.random.randint(2, size=pop_size)

        for generation in range(num_generations):
            print(f"Geração: {generation}")

            # medir o ‘fitness’ de cada cromossomo na população
            fitness = ga.cal_pop_fitness(itens, new_population)

            if any(value == 43 for value in fitness):
                reached_max_value.append(generation)
                break

            # Selecionar os melhores pais na população para o cruzamento
            parents = ga.select_mating_pool(
                new_population, fitness, num_parents_mating)

            # formar a próxima geração usando crossover
            offspring_crossover = ga.crossover(parents, offspring_size=(
                pop_size[0] - parents.shape[0], num_weights
            ))

            # adicionar variações aos filhos usando mutação
            offspring_mutation = ga.mutation(offspring_crossover)

            # criar a nova população baseada nos pais e filhos
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation

            fitness = ga.cal_pop_fitness(itens, new_population)
            best_match_idx = (np.where(fitness == np.max(fitness)))

    print('media', np.mean(reached_max_value))
    print('mediana', np.median(reached_max_value))
    print('desvio padrao', np.std(reached_max_value))

    plt.hist(reached_max_value)
    plt.show()


if __name__ == '__main__':
    main()
