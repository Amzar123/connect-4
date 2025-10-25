import random
import math
from bots.evaluation import Evaluation

class ExpectiMaxBot(Evaluation):

	
	def __init__(self, piece, depth=5):
		super().__init__(piece)  # Initialize parent class with piece assignment
		self.depth = depth       # Set the search depth limit
		
		
		self.column_weights = [1, 2, 3, 4, 3, 2, 1]

	def evaluate_window_with_position(self, board, window, start_col):
		base_score = super().evaluate_window(board, window)
		
		if base_score == 0:
			return 0
		
		window_cols = [start_col + i for i in range(4)]
		valid_cols = [col for col in window_cols if 0 <= col < len(self.column_weights)]
		
		if not valid_cols:
			return base_score
			
		avg_weight = sum(self.column_weights[col] for col in valid_cols) / len(valid_cols)
		
		positional_score = base_score * avg_weight
		
		return positional_score


	def score_position(self, board):
		score = 0

		## Score center column
		center_array = [int(i) for i in list(board.get_board()[:, board.COLUMN_COUNT//2])]
		center_count = center_array.count(self.bot_piece)
		score += center_count * 3

		## Score Horizontal with position weights
		for r in range(board.ROW_COUNT):
			row_array = [int(i) for i in list(board.get_board()[r,:])]
			for c in range(board.COLUMN_COUNT-3):
				window = row_array[c:c+board.WINDOW_LENGTH]
				score += self.evaluate_window_with_position(board, window, c)

		## Score Vertical with position weights
		for c in range(board.COLUMN_COUNT):
			col_array = [int(i) for i in list(board.get_board()[:,c])]
			for r in range(board.ROW_COUNT-3):
				window = col_array[r:r+board.WINDOW_LENGTH]
				# For vertical windows, use single column weight
				base_score = super().evaluate_window(board, window)
				score += base_score * self.column_weights[c]

		## Score positive sloped diagonal with position weights
		for r in range(board.ROW_COUNT-3):
			for c in range(board.COLUMN_COUNT-3):
				window = [board.get_board()[r+i][c+i] for i in range(board.WINDOW_LENGTH)]
				score += self.evaluate_window_with_position(board, window, c)

		## Score negative sloped diagonal with position weights
		for r in range(board.ROW_COUNT-3):
			for c in range(board.COLUMN_COUNT-3):
				window = [board.get_board()[r+3-i][c+i] for i in range(board.WINDOW_LENGTH)]
				score += self.evaluate_window_with_position(board, window, c)

		return score
		
	

	def expectimax(self, board, depth, alpha, beta, maximizingPlayer):

		# Get valid columns
		valid_locations = board.get_valid_locations()
		
		# Check state
		is_terminal = super().is_terminal_node(board)

		# BASE CASE: 
		if depth == 0 or is_terminal:
			if is_terminal:
				if board.winning_move(self.bot_piece):
					return (None, 100000000000000)    # Bot wins
				elif board.winning_move(self.opp_piece):
					return (None, -10000000000000)    # Bot loses
				else: 
					return (None, 0)
			else: 
				return (None, self.score_position(board))

		#Maximize, player turn
		if maximizingPlayer:
			value = -math.inf                          
			column = random.choice(valid_locations) #Random start
			
			for col in valid_locations:
				b_copy = board.copy_board()
				b_copy.drop_piece(col, self.bot_piece)
				
				# Evaluate move
				new_score = self.expectimax(b_copy, depth-1, alpha, beta, False)[1]

				# Update score if better
				if new_score > value:
					value = new_score
					column = col

				# Alpha-beta pruning
				alpha = max(alpha, value)
				if alpha >= beta:
					break
			return column, value
		else: #Minimize, opponent turn
			value = 0                                 
			column = random.choice(valid_locations)
			
			
			for col in valid_locations:
				b_copy = board.copy_board()
				b_copy.drop_piece(col, self.opp_piece)
				
				#evaluate opponent move
				new_score = self.expectimax(b_copy, depth-1, alpha, beta, True)[1]

				
				if new_score <= value:
					value = new_score
					column = col

				# Pruning
				beta = math.floor(value/len(valid_locations))
				if alpha >= beta:
					break 
			return column, value

	def get_move(self, board):
		col, expectimax_score = self.expectimax(board, self.depth, -math.inf, 0, True)
		return col
