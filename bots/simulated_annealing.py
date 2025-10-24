import random
import math

from bots.evaluation import Evaluation

class SimulatedAnnealing(Evaluation):
    def __init__(self, piece, depth=5):
        super().__init__(piece)
        self.depth = depth
        self.initial_temp = 100
        self.cooling_rate = 0.95
        self.n_iterations = 1000
    
    # Objective function: Evaluate board position for a given column move
    def objective_function(self, board, col):
        """
        Evaluates how good a move is in column 'col'.
        Returns negative score (lower is better for SA minimization).
        """
        valid_locations = board.get_valid_locations()
        if col not in valid_locations:
            return math.inf
        
        # Simulate the move
        row = board.get_next_open_row(col)
        temp_board = board.copy_board()
        board.drop_piece(col, self.bot_piece)
        
        # Check for immediate win
        if board.winning_move(self.bot_piece):
            return -100000
        
        # Evaluate board position
        score = 0
        opp_piece = 1 if self.bot_piece == 2 else 2
        
		# Get the board array
        b = temp_board.board if hasattr(temp_board, 'board') else temp_board
        
        # Evaluate all possible windows
        # Horizontal
        for r in range(6):
            for c in range(4):
                window = [b[r][c+i] for i in range(4)]
                score += self._score_window(window, self.bot_piece, opp_piece)
        
        # Vertical
        for c in range(7):
            for r in range(3):
                window = [temp_board[r+i][c] for i in range(4)]
                score += self._score_window(window, self.bot_piece, opp_piece)
        
        # Positive diagonal
        for r in range(3):
            for c in range(4):
                window = [temp_board[r+i][c+i] for i in range(4)]
                score += self._score_window(window, self.bot_piece, opp_piece)
        
        # Negative diagonal
        for r in range(3):
            for c in range(4):
                window = [temp_board[r+3-i][c+i] for i in range(4)]
                score += self._score_window(window, self.bot_piece, opp_piece)
        
        # Center preference
        center_count = sum([1 for r in range(6) if temp_board[r][3] == self.bot_piece])
        score += center_count * 3
        
        return -score  # Negative because SA minimizes
    
    def _score_window(self, window, piece, opp_piece):
        """Helper to score a 4-cell window"""
        score = 0
        piece_count = window.count(piece)
        empty_count = window.count(0)
        opp_count = window.count(opp_piece)
        
        if piece_count == 4:
            score += 1000
        elif piece_count == 3 and empty_count == 1:
            score += 100
        elif piece_count == 2 and empty_count == 2:
            score += 10
        
        if opp_count == 3 and empty_count == 1:
            score -= 80
        
        return score
    
    # Neighbor function: select a different valid column
    def get_neighbor(self, current_col, valid_locations):
        """
        Generate a neighboring solution by selecting a different valid column.
        Prefers adjacent columns for local search.
        """
        if len(valid_locations) <= 1:
            return current_col
        
        # 70% chance to pick adjacent column, 30% random
        if random.random() < 0.7:
            neighbors = []
            if current_col - 1 in valid_locations:
                neighbors.append(current_col - 1)
            if current_col + 1 in valid_locations:
                neighbors.append(current_col + 1)
            if neighbors:
                return random.choice(neighbors)
        
        # Pick random different column
        other_cols = [c for c in valid_locations if c != current_col]
        return random.choice(other_cols) if other_cols else current_col
    
    # Simulated Annealing function
    def simulated_annealing(self, board):
        """
        Use simulated annealing to find the best column move.
        """
        valid_locations = board.get_valid_locations()
        
        if not valid_locations:
            return None
        if len(valid_locations) == 1:
            return valid_locations[0]
        
        # Initial solution: random valid column
        current_col = random.choice(valid_locations)
        current_eval = self.objective_function(board, current_col)
        
        best_col = current_col
        best_eval = current_eval
        
        temperature = self.initial_temp
        
        # SA iterations
        for i in range(self.n_iterations):
            # Cool down temperature
            temperature *= self.cooling_rate
            
            # Generate neighbor
            candidate_col = self.get_neighbor(current_col, valid_locations)
            candidate_eval = self.objective_function(board, candidate_col)
            
            # Calculate change in objective
            delta = candidate_eval - current_eval
            
            # Accept or reject
            if delta < 0:
                # Better solution, always accept
                current_col = candidate_col
                current_eval = candidate_eval
            else:
                # Worse solution, accept with probability
                acceptance_prob = math.exp(-delta / temperature)
                if random.random() < acceptance_prob:
                    current_col = candidate_col
                    current_eval = candidate_eval
            
            # Track best solution
            if current_eval < best_eval:
                best_col = current_col
                best_eval = current_eval
        
        return best_col
    
    def get_move(self, board):
        col = self.simulated_annealing(board)
        return col