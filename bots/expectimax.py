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
	
	def __init__(self, piece, depth=5):
		"""
		Initialize the ExpectiMax bot.
		
		Args:
			piece: The piece type for this bot (1 or 2)
			depth: Maximum search depth for the algorithm (default: 5)
		"""
		super().__init__(piece)  # Initialize parent class with piece assignment
		self.depth = depth       # Set the search depth limit

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
				return (None, super().score_position(board))  # Use heuristic evaluation

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
