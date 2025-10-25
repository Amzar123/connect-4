import math
import random

from bots.evaluation import Evaluation

class SimulatedAnnealing(Evaluation):
    def __init__(self, piece):
        super().__init__(piece)
        self.bot_piece = piece
        self.depth = 5
        self.initial_temp = 100.0
        self.cooling_rate = 0.99
        self.n_iterations = 1000

    def objective_function(self, board, col):
        temp_board = board.copy_board()
        if not temp_board.is_valid_location(col):
            return math.inf

        if temp_board.winning_move(self.bot_piece):
            return -100000
        
        return -super().score_position(temp_board)

    def get_neighbor(self, current_col, valid_locations):
        if not valid_locations:
            return None
        if len(valid_locations) <= 1:
            return valid_locations[0]
        
        if random.random() < 0.7:
            neighbors = []
            if current_col - 1 in valid_locations:
                neighbors.append(current_col - 1)
            if current_col + 1 in valid_locations:
                neighbors.append(current_col + 1)
            if neighbors:
                return random.choice(neighbors)
        
        other_cols = [c for c in valid_locations if c != current_col]
        return random.choice(other_cols) if other_cols else current_col

    def simulated_annealing(self, board):
        valid_locations = board.get_valid_locations()
        
        if not valid_locations:
            return None
        if len(valid_locations) == 1:
            return valid_locations[0]
        
        current_col = random.choice(valid_locations)
        current_eval = self.objective_function(board, current_col)
        
        best_col = current_col
        best_eval = current_eval
        
        temperature = self.initial_temp
        
        for _ in range(self.n_iterations):
            temperature *= self.cooling_rate
            if temperature < 1e-4: 
                break
            
            candidate_col = self.get_neighbor(current_col, valid_locations)
            
            current_eval = self.objective_function(board, current_col)
            
            candidate_eval = self.objective_function(board, candidate_col)
            
            delta = candidate_eval - current_eval
            
            if delta < 0:
                current_col = candidate_col
            else:
                try:
                    acceptance_prob = math.exp(-delta / temperature)
                    if random.random() < acceptance_prob:
                        current_col = candidate_col
                except OverflowError:
                    continue 
            
            if self.objective_function(board, current_col) < best_eval:
                 best_eval = self.objective_function(board, current_col)
                 best_col = current_col
        
        return best_col

    def get_move(self, board):
        return self.simulated_annealing(board)

