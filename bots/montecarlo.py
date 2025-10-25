import numpy as np
import sys
import copy
import time
import random

class MonteCarloBot():
    
    def __init__(self, piece, max_iterations = 20000 , timeout = 2):
        self.piece = piece                    
        self.max_iterations = max_iterations 
        self.timeout = timeout              
        self.currentNode = None           

    def montecarlo_tree_search(self, board, max_iterations, currentNode, timeout = 100):
        
        # Initialize root
        rootnode = Node(piece=board.PREV_PLAYER, board=board)

        # CContinue from tree if available
        if currentNode is not None:
            rootnode = currentNode


        start = time.perf_counter()
        
        for i in range(max_iterations):
            #Selection
            node = rootnode
            state = board.copy_board()

            while node.available_moves == [] and node.children != []:
                node = node.selection() 
                state.drop_piece(node.move, state.CURR_PLAYER)

            #Expansion
            if node.available_moves != []:
                col = random.choice(node.available_moves)
                state.drop_piece(col, state.CURR_PLAYER)
                node = node.expand(col, state)

            #Rollout
            while state.get_valid_locations():
                col = random.choice(state.get_valid_locations())
                state.drop_piece(col, state.CURR_PLAYER)
                #Check winner
                if state.winning_move(state.PREV_PLAYER):
                    break

            #Backpropagation
            while node is not None:
                # Update result
                node.update(state.search_result(node.piece))
                node = node.parent #Move to parent

            duration = time.perf_counter() - start
            if duration > timeout:
                break

        win_ratio = lambda x: x.wins/x.visits
        sorted_children = sorted(rootnode.children, key = win_ratio)[::-1]

        return rootnode, sorted_children[0].move

    def get_child_node(self, node, board, move, piece):
        for child in node.children:
            if child.move == move:
                return child
        
        return Node(piece = piece, board = board)

    def get_move(self, board):
        # Initialize first move
        if self.currentNode is None:
            self.currentNode = Node(piece=self.piece, board=board)
        
        # Update tree 
        if board.PREV_MOVE is not None:
            self.currentNode = self.get_child_node(self.currentNode, board, board.PREV_MOVE, board.CURR_PLAYER)

        self.currentNode, col = self.montecarlo_tree_search(board, self.max_iterations, self.currentNode, self.timeout)
        
        self.currentNode = self.get_child_node(self.currentNode, board, col, board.PREV_PLAYER)
        
        return col

class Node:
    
    def __init__(self, piece, board, parent=None, move=None):
       
        self.board = board.copy_board()         
        self.parent = parent                    
        self.move = move                         
        self.available_moves = board.get_valid_locations() 
        self.children = []                       
        self.wins = 0                         
        self.visits = 0                       
        self.piece = piece                     

    def selection(self):
        uct_val = lambda x: x.wins / x.visits + np.sqrt(2 * np.log(self.visits) / x.visits)
        return sorted(self.children, key = uct_val)[-1]  # Return node with max UCT

    def expand(self, move, board):
    
        child = Node(piece = board.PREV_PLAYER, board = board, parent = self, move = move)
        
        self.available_moves.remove(move)
        
        self.children.append(child)
        return child

    def update(self, result):
        
        self.wins += result     
        self.visits += 1   
