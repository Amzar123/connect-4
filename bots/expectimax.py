import random
import math
from bots.evaluation import Evaluation

class ExpectiMaxBot(Evaluation):
	"""
	ExpectiMax Bot implementation for Connect-4 game.
	
	ExpectiMax is a variant of the Minimax algorithm that handles uncertainty
	in opponent moves by calculating expected values instead of assuming
	optimal opponent play.
	"""
	
	def __init__(self, piece, depth=5, center_weight=3, four_weight=100, three_weight=5, two_weight=2, opp_three_penalty=4):
		"""
		Initialize the ExpectiMax bot with customizable evaluation parameters.
		
		Args:
			piece: The piece type for this bot (1 or 2)
			depth: Maximum search depth for the algorithm (default: 5)
			center_weight: Weight for center column preference (default: 3)
			four_weight: Score for 4-in-a-row (default: 100)
			three_weight: Score for 3-in-a-row + 1 empty (default: 5)
			two_weight: Score for 2-in-a-row + 2 empty (default: 2)
			opp_three_penalty: Penalty for opponent 3-in-a-row + 1 empty (default: 4)
		"""
		super().__init__(piece)  # Initialize parent class with piece assignment
		self.depth = depth       # Set the search depth limit
		
		# ExpectiMax-specific evaluation parameters for performance tuning
		self.center_weight = center_weight
		self.four_weight = four_weight
		self.three_weight = three_weight
		self.two_weight = two_weight
		self.opp_three_penalty = opp_three_penalty

	def evaluate_window(self, board, window):
		"""
		evaluate_window that uses parent's logic with ExpectiMax-specific adjustments.

		Calls the parent's evaluate_window first, then applies ExpectiMax-specific
		modifications based on the customizable parameters.
		
		Args:
			board: The game board
			window: List of 4 pieces to evaluate
			
		Returns:
			int: Score for this window with ExpectiMax-specific adjustments
		"""
		# Start with the base evaluation from parent class
		base_score = super().evaluate_window(board, window)
		
		# Apply ExpectiMax-specific weight adjustments
		adjusted_score = 0
		
		# Re-calculate with custom weights instead of default ones
		if window.count(self.bot_piece) == 4:
			# Replace default weight (100) with custom weight
			adjusted_score += self.four_weight - 100
		elif window.count(self.bot_piece) == 3 and window.count(board.EMPTY) == 1:
			# Replace default weight (5) with custom weight  
			adjusted_score += self.three_weight - 5
		elif window.count(self.bot_piece) == 2 and window.count(board.EMPTY) == 2:
			# Replace default weight (2) with custom weight
			adjusted_score += self.two_weight - 2

		# Adjust opponent threat penalty
		if window.count(self.opp_piece) == 3 and window.count(board.EMPTY) == 1:
			# Replace default penalty (-4) with custom penalty
			adjusted_score += (-self.opp_three_penalty) - (-4)

		return base_score + adjusted_score

	def score_position(self, board):
		"""
		score_position that builds upon parent's evaluation with ExpectiMax customizations.
		
		Call parents score_position get the base evaluation, then applies ExpectiMax-specific
		adjustments for center column weighting and other strategic considerations.
		
		Args:
			board: The game board to evaluate
			
		Returns:
			int: Total score with ExpectiMax-specific enhancements applied
		"""
		# Get base score from parent class using super()
		base_score = super().score_position(board)
		
		# Apply ExpectiMax-specific center column weight adjustment
		center_array = [int(i) for i in list(board.get_board()[:, board.COLUMN_COUNT//2])]
		center_count = center_array.count(self.bot_piece)
		
		# Adjust center weight: remove default weight (3) and add custom weight
		center_adjustment = center_count * (self.center_weight - 3)
		
		return base_score + center_adjustment

	def expectimax(self, board, depth, alpha, beta, maximizingPlayer):
		"""
		Core ExpectiMax algorithm implementation.
		
		Args:
			board: Current game board state
			depth: Remaining search depth
			alpha: Alpha value for alpha-beta pruning (best value for maximizing player)
			beta: Beta value for alpha-beta pruning (best value for minimizing player)
			maximizingPlayer: True if it's the bot's turn, False if opponent's turn
			
		Returns:
			tuple: (best_column, best_value) where best_column is the optimal move
		"""
		
		# Get all valid columns where a piece can be dropped
		valid_locations = board.get_valid_locations()
		
		# Check if this is a terminal state (game over or winning position)
		is_terminal = super().is_terminal_node(board)

		# BASE CASE: End condition - either reached depth limit or game is over
		if depth == 0 or is_terminal:
			if is_terminal:
				# Game has ended, return large positive/negative values for wins/losses
				if board.winning_move(self.bot_piece):
					return (None, 100000000000000)    # Bot wins - very high score
				elif board.winning_move(self.opp_piece):
					return (None, -10000000000000)    # Bot loses - very low score
				else: # Game is over, no more valid moves (draw)
					return (None, 0)                  # Draw - neutral score
			else: # Depth limit reached - evaluate current position
				return (None, self.score_position(board))  # Use enhanced ExpectiMax evaluation with super()

		# MAXIMIZING NODE: Bot's turn - choose the move that maximizes score
		if maximizingPlayer:
			value = -math.inf                          # Start with worst possible value
			column = random.choice(valid_locations)    # Default random move
			
			# Try each valid column to find the best move
			for col in valid_locations:
				# Create a copy of the board to simulate the move
				b_copy = board.copy_board()
				b_copy.drop_piece(col, self.bot_piece)  # Simulate bot's move
				
				# Recursively evaluate this move (opponent's turn next)
				new_score = self.expectimax(b_copy, depth-1, alpha, beta, False)[1]

				# Update best move if this score is better
				if new_score > value:
					value = new_score
					column = col

				# Alpha-beta pruning: update alpha and check for cutoff
				alpha = max(alpha, value)
				if alpha >= beta:
					break  # Beta cutoff - no need to explore further
			return column, value
		else: # EXPECTATION NODE: Opponent's turn - calculate expected value
			"""
			Unlike minimax which assumes optimal opponent play, expectimax
			calculates the expected value assuming the opponent chooses
			moves with some probability distribution.
			"""
			value = 0                                  # Start with neutral value
			column = random.choice(valid_locations)    # Default random move
			
			# Evaluate each possible opponent move
			for col in valid_locations:
				# Create a copy of the board to simulate opponent's move
				b_copy = board.copy_board()
				b_copy.drop_piece(col, self.opp_piece)  # Simulate opponent's move
				
				# Recursively evaluate this move (bot's turn next)
				new_score = self.expectimax(b_copy, depth-1, alpha, beta, True)[1]

				# Note: This implementation seems to have a logic issue
				# In true expectimax, we should sum all values and divide by count
				# Current implementation uses <= comparison which is unusual
				if new_score <= value:
					value = new_score
					column = col

				# Calculate expected value (average) for beta pruning
				beta = math.floor(value/len(valid_locations))
				if alpha >= beta:
					break  # Alpha cutoff - pruning condition
			return column, value

	def get_move(self, board):
		"""
		Public interface method to get the bot's next move.
		
		This method initiates the expectimax search algorithm and returns
		the best column to drop a piece in.
		
		Args:
			board: Current game board state
			
		Returns:
			int: Column number (0-6) where the bot wants to drop its piece
		"""
		# Start expectimax search from root with initial alpha-beta values
		# alpha starts at -infinity, beta at 0 (neutral)
		col, expectimax_score = self.expectimax(board, self.depth, -math.inf, 0, True)
		return col
