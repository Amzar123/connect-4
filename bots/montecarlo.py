import numpy as np
import sys
import copy
import time
import random

class MonteCarloBot():
    """
    Monte Carlo Tree Search (MCTS).
    
    MCTS is a probabilistic search algorithm that builds a search tree incrementally
    and uses random simulations to evaluate positions. It balances exploration
    of new moves with exploitation of promising moves using the UCT formula.
    
    The algorithm consists of four main phases:
    1. Selection: Navigate down the tree using UCT values
    2. Expansion: Add a new child node to the tree
    3. Simulation (Rollout): Play random moves to terminal state
    4. Backpropagation: Update statistics back up the tree
    """
    
    def __init__(self, piece, max_iterations = 20000 , timeout = 2):
        self.piece = piece                    # Bot's piece type
        self.max_iterations = max_iterations  # Limit on MCTS iterations
        self.timeout = timeout               # Time limit for move selection
        self.currentNode = None              # Current position in the search tree

    def montecarlo_tree_search(self, board, max_iterations, currentNode, timeout = 100):
        """
        Core Monte Carlo Tree Search algorithm implementation.
        
        This method performs the four phases of MCTS iteratively:
        1. Selection: Traverse tree using UCT to find promising leaf
        2. Expansion: Add new child node to expand the tree
        3. Simulation: Random rollout from new position to game end
        4. Backpropagation: Update win/visit statistics up the tree
        
        Args:
            board: Current game board state
            max_iterations: Maximum number of MCTS iterations to perform
            currentNode: Existing tree node to continue from (or None for new tree)
            timeout: Maximum time in seconds to spend on search
            
        Returns:
            tuple: (root_node, best_move) where best_move is the column to play
        """
        # Initialize root node - represents current board position
        rootnode = Node(piece=board.PREV_PLAYER, board=board)

        # Continue from existing tree if available (tree reuse optimization)
        if currentNode is not None:
            rootnode = currentNode

        # Track time to respect timeout limit
        start = time.perf_counter()
        
        # Main MCTS loop - perform iterations until limit or timeout
        for i in range(max_iterations):
            # PHASE 1: SELECTION
            # Start from root and traverse down tree using UCT values
            # Continue until we reach a leaf node (unexpanded or terminal)
            node = rootnode
            state = board.copy_board()  # Working copy of board state

            # Navigate down tree using UCT selection policy
            # Stop when we find a node with unexplored moves or no children
            while node.available_moves == [] and node.children != []:
                node = node.selection()  # Choose child with highest UCT value
                state.drop_piece(node.move, state.CURR_PLAYER)  # Update board state

            # PHASE 2: EXPANSION
            # If current node has unexplored moves, expand tree by adding new child
            if node.available_moves != []:
                # Randomly select an unexplored move to expand
                col = random.choice(node.available_moves)
                state.drop_piece(col, state.CURR_PLAYER)  # Apply the move
                node = node.expand(col, state)            # Create new child node

            # PHASE 3: SIMULATION (ROLLOUT)
            # Play random moves from current position until game ends
            # This gives us a sample outcome for this position
            while state.get_valid_locations():  # Continue while moves available
                col = random.choice(state.get_valid_locations())  # Random move
                state.drop_piece(col, state.CURR_PLAYER)
                # Check for immediate win after each move
                if state.winning_move(state.PREV_PLAYER):
                    break  # Game over - someone won

            # PHASE 4: BACKPROPAGATION
            # Propagate simulation result back up the tree
            # Update win/visit statistics for all nodes on path to root
            while node is not None:
                # Update this node with the game result
                node.update(state.search_result(node.piece))
                node = node.parent  # Move up to parent node

            # Check timeout - stop if we've exceeded time limit
            duration = time.perf_counter() - start
            if duration > timeout:
                break  # Time's up - use current tree state

        # MOVE SELECTION: Choose best move based on simulation results
        # Select child with highest win rate (exploitation)
        win_ratio = lambda x: x.wins/x.visits
        sorted_children = sorted(rootnode.children, key = win_ratio)[::-1]

        # Optional: Print statistics for debugging
        #for node in sorted_children:
        #    print('Move: %s Win Rate: %.2f%%' % (node.move + 1, 100 * node.wins / node.visits))
        #print('Simulations performed: %s\n' % i)

        # Return updated tree and best move
        return rootnode, sorted_children[0].move

    def get_child_node(self, node, board, move, piece):
        """
        Helper method to find or create a child node for a specific move.
        
        This supports tree reuse between moves by finding existing nodes
        that correspond to the current game state.
        
        Args:
            node: Parent node to search in
            board: Current board state
            move: The move that was played
            piece: The piece type for the new node
            
        Returns:
            Node: Existing child node if found, or new node if not found
        """
        # Search existing children for matching move
        for child in node.children:
            if child.move == move:
                return child
        # Move not found in tree - create new node
        return Node(piece = piece, board = board)

    def get_move(self, board):
        """
        Public interface method to get the bot's next move.
        
        This method manages the search tree between moves and initiates
        the MCTS algorithm to find the best move.
        
        Args:
            board: Current game board state
            
        Returns:
            int: Column number (0-6) where the bot wants to drop its piece
        """
        # Initialize tree on first move
        if self.currentNode is None:
            self.currentNode = Node(piece=self.piece, board=board)
        
        # Update tree to reflect opponent's last move (tree reuse)
        if board.PREV_MOVE is not None:
            self.currentNode = self.get_child_node(self.currentNode, board, board.PREV_MOVE, board.CURR_PLAYER)

        # Run MCTS to find best move
        self.currentNode, col = self.montecarlo_tree_search(board, self.max_iterations, self.currentNode, self.timeout)
        
        # Update tree to reflect our chosen move
        self.currentNode = self.get_child_node(self.currentNode, board, col, board.PREV_PLAYER)
        
        return col

class Node:
    """
    Node class representing a game state in the Monte Carlo search tree.
    
    Each node stores:
    - Game board state
    - Statistical information (wins, visits)
    - Tree structure (parent, children)
    - Available moves for expansion
    
    The node implements the key MCTS operations:
    - Selection using UCT (Upper Confidence Bound for Trees)
    - Expansion by adding new child nodes
    - Update by incorporating simulation results
    """
    
    def __init__(self, piece, board, parent=None, move=None):
        """
        Initialize a new tree node.
        
        Args:
            piece: The piece type that just moved to reach this position
            board: The game board state at this node
            parent: Parent node in the tree (None for root)
            move: The move that led to this position (None for root)
        """
        self.board = board.copy_board()           # Store board state
        self.parent = parent                      # Link to parent node
        self.move = move                          # Move that created this node
        self.available_moves = board.get_valid_locations()  # Unexplored moves
        self.children = []                        # Child nodes (explored moves)
        self.wins = 0                            # Number of wins in simulations
        self.visits = 0                          # Total number of simulations
        self.piece = piece                       # Piece type for this position

    def selection(self):
        """
        Select child node with highest UCT (Upper Confidence Bound for Trees) value.
        
        UCT balances exploitation (high win rate) with exploration (low visit count).
        Formula: UCT = win_rate + C * sqrt(ln(parent_visits) / child_visits)
        where C = sqrt(2) is the exploration constant.
        
        Returns:
            Node: Child node with highest UCT value
        """
        # UCT formula: exploitation + exploration
        # Higher win rate = better exploitation
        # Lower visit count = more exploration potential
        uct_val = lambda x: x.wins / x.visits + np.sqrt(2 * np.log(self.visits) / x.visits)
        return sorted(self.children, key = uct_val)[-1]  # Return node with max UCT

    def expand(self, move, board):
        """
        Expand the tree by adding a new child node for the given move.
        
        This method:
        1. Creates a new child node for the specified move
        2. Removes the move from available_moves (now explored)
        3. Adds the child to the children list
        
        Args:
            move: Column number for the move to expand
            board: Board state after the move is applied
            
        Returns:
            Node: The newly created child node
        """
        # Create new child node representing the state after this move
        child = Node(piece = board.PREV_PLAYER, board = board, parent = self, move = move)
        
        # Mark this move as explored by removing from available moves
        self.available_moves.remove(move)
        
        # Add child to tree structure
        self.children.append(child)
        return child

    def update(self, result):
        """
        Update node statistics with a simulation result.
        
        This method is called during backpropagation to update the
        win count and visit count for this node.
        
        Args:
            result: Simulation result for this node's piece
                   (1.0 = win, 0.5 = draw, 0.0 = loss)
        """
        self.wins += result     # Add win value (can be fractional for draws)
        self.visits += 1        # Increment total visit count
