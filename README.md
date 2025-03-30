# Genetic Algorithm for Optimizing a 2nd-Degree Polynomial
This project implements a Genetic Algorithm (GA) to optimize a 2nd-degree polynomial fitness function. The algorithm evolves a population of individuals over multiple generations to find the maximum value of the fitness function.

## Features
Fitness Function: A customizable 2nd-degree polynomial (f(x) = ax^2 + bx + c).
### Genetic Operators:
Selection: Roulette wheel selection.
Crossover: Single-point crossover.
Mutation: Random bit-flip mutation.
## Visualization:
Fitness evolution over generations.
Fitness function with individuals from all generations plotted.
Project Structure
## How It Works
### Initialization:
A population of individuals is initialized with random values within a specified range.
### Fitness Evaluation:
Each individual is evaluated using the fitness function (f(x) = ax^2 + bx + c).
### Selection:
Individuals are selected for reproduction based on their fitness using roulette wheel selection.
### Crossover:
Pairs of parents are combined using single-point crossover to produce offspring.
### Mutation:
Random mutations are applied to offspring to introduce diversity.
### Evolution:
The process repeats for a specified number of generations, with the population evolving toward the optimal solution.
## Installation
Clone the repository:
```
git clone https://github.com/your-username/GeneticAlgorithm.git
cd GeneticAlgorithm
```
Install the required Python libraries:
```
pip install numpy matplotlib
```
Usage
Run the main.py file to execute the Genetic Algorithm:
```
python main.py
```
