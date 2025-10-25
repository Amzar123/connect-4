import random

from bots.evaluation import Evaluation

class GeneticAlgorithm(Evaluation):
    def __init__(self, piece):
        super().__init__(piece)
        self.bot_piece = piece
        self.best_genome = None

    def create_population(self, size, valid_locations):
        return [random.choice(valid_locations) for _ in range(size)]

    def select_best(self, population, fitness_scores, elite_count):
        combined = list(zip(population, fitness_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in combined[:elite_count]]

    def crossover(self, parent1, parent2):
        return random.choice([parent1, parent2])

    def mutate(self, genome, mutation_rate, valid_locations):
        if random.random() < mutation_rate:
            return random.choice(valid_locations)
        return genome

    def genetic_algorithm(self, board, population_size=50, elite_size=10, mutation_rate=0.2, generations=50):
        valid_locations = board.get_valid_locations()
        if not valid_locations:
            return None
        
        population = self.create_population(population_size, valid_locations)
        
        for _ in range(generations):
            fitness_scores = []
            for move in population:
                temp_board = board.copy_board()
                temp_board.drop_piece(move, self.bot_piece)
                fitness_scores.append(super().score_position(temp_board))
            
            best_moves = self.select_best(population, fitness_scores, elite_size)
            
            next_generation = best_moves[:]
            while len(next_generation) < population_size:
                parent1 = random.choice(best_moves)
                parent2 = random.choice(best_moves)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, mutation_rate, valid_locations)
                next_generation.append(child)
            
            population = next_generation
            
            best_move = self.select_best(population, fitness_scores, 1)[0]
            self.best_genome = best_move
            
        return self.best_genome

    def get_move(self, board):
        return self.genetic_algorithm(board)