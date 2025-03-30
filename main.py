import numpy as np
import matplotlib.pyplot as plt

output_file = open("Evolution.txt", "w")

class GeneticAlgorithm:
    def __init__(self, population_size, left, right, a, b, c,
                 precision, crossover_prob, mutation_prob, steps):
        
        output_file.write("Initialize the Genetic Algorithm with parameters.\n")
        output_file.write(f"Population Size: {population_size}\n")
        output_file.write(f"Left Bound: {left}\n")
        output_file.write(f"Right Bound: {right}\n")
        output_file.write(f"Coefficients: a={a}, b={b}, c={c}\n")
        output_file.write(f"Precision: {precision}\n")
        output_file.write(f"Crossover Probability: {crossover_prob}\n")
        output_file.write(f"Mutation Probability: {mutation_prob}\n")
        output_file.write(f"Steps: {steps}\n")
        
        self.population_size = population_size
        self.left = left
        self.right = right
        self.a = a
        self.b = b
        self.c = c
        self.precision = precision
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.steps = steps
        
        self.max_fitness_history = [] 
        self.mean_fitness_history = [] 
        
        self.initialize_population()
        
        
    def initialize_population(self):
        output_file.write("Initialize the population with random values.\n")
        self.population =  np.random.uniform(self.left, self.right, self.population_size)
        output_file.write("Population initialization completed.\n")


    def compute_data(self):
        fitness_values = []
    
        output_file.write("Individuals details: \n")
        for individual in self.population:
            binary_repr = self.to_binary(individual)
            fitness_value = self.fitness_function(individual)
            fitness_values.append(fitness_value)
            output_file.write(f"Individual: {individual:.6f}, Binary: {binary_repr}, Fitness: {fitness_value:.6f}\n")
        
        self.selection_probabilites, self.cumulative_probabilities = self.compute_selection_probabilites(fitness_values)
        
        output_file.write("Selection probabilities computed.\n")
        for i, prob in enumerate(self.selection_probabilites):
            output_file.write(f"Individual {i}: Selection Probability: {prob:.6f}\n")
            
        output_file.write("Cumulative probabilities computed.\n")
        for i, cum_prob in enumerate(self.cumulative_probabilities):
            output_file.write(f"Individual {i}: Cumulative Probability: {cum_prob:.6f}\n")
            
        self.best_individual = self.population[np.argmax(fitness_values)]
        output_file.write(f"Best Individual: {self.best_individual:.6f}\n")
        
        output_file.write("Starting the evolution process...\n")
        
        
    def evolution(self):
        self.compute_fitness_statistics()
        all_generations = []  
        
        for i in range(self.steps):
            output_file.write(f"Step {i+1}:\n")
            
            self.compute_data()
            self.crossover()
            self.mutation()
            self.compute_fitness_statistics()
            
            all_generations.append(self.population.copy())
            
            output_file.write(f"Population after step {i + 1}:\n")
            for individual in self.population:
                output_file.write(f"{individual:.6f}\n")
        
        return all_generations  
        
    
    def to_binary(self, value):
        range_size = self.right - self.left
        num_bits = int(np.ceil(np.log2(range_size * (10 ** self.precision))))
        scaled_value = int((value - self.left) * (2 ** num_bits) / range_size)
        return f"{scaled_value:0{num_bits}b}"
    
    
    def to_decimal(self, binary_str):
        num_bits = len(binary_str)
        range_size = self.right - self.left
        scaled_value = int(binary_str, 2)
        decimal_value = (scaled_value * range_size / (2 ** num_bits)) + self.left
        return decimal_value


    def fitness_function(self, x):
        # fitness function is 2nd degree polynomial
        return self.a * x**2 + self.b * x + self.c
    
    
    def compute_selection_probabilites(self, fitness_values):
        total_fitness = np.sum(fitness_values)
        probabilities = fitness_values / total_fitness
        cumulative_probabilities = np.cumsum(probabilities)
        return probabilities, cumulative_probabilities
    
    
    def select_interval(self):
        random_value = np.random.uniform(0, 1)
        low, high = 0, len(self.cumulative_probabilities) - 1

        while low < high:
            mid = (low + high) // 2
            if random_value <= self.cumulative_probabilities[mid]:
                high = mid
            else:
                low = mid + 1

        lower_bound = self.cumulative_probabilities[low - 1] if low > 0 else 0
        upper_bound = self.cumulative_probabilities[low]

        output_file.write(f"Random value: {random_value:.6f}, Interval: [{lower_bound:.6f}, {upper_bound:.6f}]\n")
        return low
        

    def crossover(self):
        output_file.write("Performing crossover...\n")
        new_population = []

        for _ in range(self.population_size // 2):  
            parent1_index = self.select_interval()
            parent2_index = self.select_interval()
            
            while parent2_index == parent1_index:
                parent2_index = self.select_interval()
            
            parent1 = self.to_binary(self.population[parent1_index])
            parent2 = self.to_binary(self.population[parent2_index])
            
            output_file.write(f"Selected Parents: {parent1} with index {parent1_index} and {parent2} with index {parent2_index}\n")

            if np.random.uniform(0, 1) < self.crossover_prob:
                crossover_point = np.random.randint(1, len(parent1) - 1)
                output_file.write(f"Crossover Point: {crossover_point}\n")

                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]

                child1_decimal = self.to_decimal(child1)
                child2_decimal = self.to_decimal(child2)

                output_file.write("Children created:\n")
                output_file.write(f"Child 1: {child1} -> {child1_decimal:.6f}\n")
                output_file.write(f"Child 2: {child2} -> {child2_decimal:.6f}\n")

                new_population.extend([child1_decimal, child2_decimal])
            else:
                output_file.write("No crossover performed, parents copied to new population.\n")
                new_population.extend([self.population[parent1_index], self.population[parent2_index]])

        self.population = np.array(new_population)
        output_file.write("Crossover completed.\n")
        
    
    def mutation(self):
        output_file.write("Performing mutation...\n")
        for i in range(len(self.population)):
            if np.random.uniform(0, 1) < self.mutation_prob:
                binary_repr = self.to_binary(self.population[i])
                mutation_point = np.random.randint(0, len(binary_repr))
                
                mutated_binary = list(binary_repr)
                mutated_binary[mutation_point] = '1' if mutated_binary[mutation_point] == '0' else '0'
                mutated_binary_str = ''.join(mutated_binary)
                
                self.population[i] = self.to_decimal(mutated_binary_str)
                output_file.write(f"Mutated Individual {i}: {binary_repr} -> {mutated_binary_str}\n")
        
        output_file.write("Mutation completed.\n")
        
    
    def compute_fitness_statistics(self):
        fitness_values = [self.fitness_function(individual) for individual in self.population]
        
        max_fitness = np.max(fitness_values)
        mean_fitness = np.mean(fitness_values)
    
        output_file.write(f"Max Fitness: {max_fitness:.6f}\n")
        output_file.write(f"Mean Fitness: {mean_fitness:.6f}\n")
        
        self.max_fitness_history.append(max_fitness)
        self.mean_fitness_history.append(mean_fitness)
        
        return max_fitness, mean_fitness
    
    
    def plot_fitness_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.max_fitness_history, label="Max Fitness", color="blue")
        plt.plot(self.mean_fitness_history, label="Mean Fitness", color="orange")
        plt.title("Fitness Evolution Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid()
        plt.savefig("fitness_evolution.png")  
        plt.show()
       
        
    def plot_fitness_function_with_all_individuals(self, all_generations):
        x_values = np.linspace(self.left, self.right, 500)
        y_values = [self.fitness_function(x) for x in x_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, label="Fitness Function", color="green")
        
        num_generations = len(all_generations)
        for generation, population in enumerate(all_generations):
            fitness_values = [self.fitness_function(individual) for individual in population]
            point_size = 10 + (generation / num_generations) * 40  
            plt.scatter(population, fitness_values, label=f"Generation {generation + 1}", alpha=0.5, s=point_size)
        
        plt.title("2nd Degree Fitness Function with Individuals from All Generations")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        #plt.legend()
        plt.grid()
        plt.savefig("fitness_function_all_generations.png") 
        plt.show()
            

gen = GeneticAlgorithm(20, -1, 2, -1, 1, 2, 6, 0.25, 0.01, 50)
all_generations = gen.evolution() 
gen.plot_fitness_function_with_all_individuals(all_generations) 
gen.plot_fitness_history()
output_file.write("Evolution process completed.\n")
output_file.close()