#Packages
import numpy as np
import random
from copy import deepcopy
import csv
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from math import log


class Human_Player(object):
	def __init__(self):
		self.state_index = []
	
	def place(self, state, free_space, player):
		print('List of valid moves: ')
		print(free_space)
		temp = input('Pick a move (counting from 0): ')
		return free_space[int(temp)]


	def valid_move(self, state, game_type, free_space, pieces):
		valid_moves = []
		if game_type == 3:
			for piece in pieces:
				for space in adj_dict_3[str(piece)]:
					if space in free_space:
						valid_moves.append((piece,space))

		if game_type == 6:
			for piece in pieces:
				for space in adj_dict_6[str(piece)]:
					if space in free_space:
						valid_moves.append((piece,space))

		if game_type == 9:
			for piece in pieces:
				for space in adj_dict_9[str(piece)]:
					if space in free_space:
						valid_moves.append((piece,space))

		if game_type == 12:
			for piece in pieces:
				for space in adj_dict_12[str(piece)]:
					if space in free_space:
						valid_moves.append((piece,space))

		return valid_moves

	def move(self, state, game_type, free_space, pieces, player, enable_flying):
		valid_moves = valid_move(self, state, game_type, free_space, pieces)
		print('List of valid moves ((piece, space to move to)): ')
		print(valid_moves)
		temp = input('Pick a move (counting from 0): ')
		if enable_flying:
			print('We can fly! Free Spaces: ')
			print(free_space)
			temp2 = input('Pick a space (counting from 0): ')
			return (valid_moves[int(temp)][0],free_space[int(temp2)])
		else:
			return valid_moves[int(temp)]
	
	def remove_piece(self, state, piece_list, game_type, player, enable_flying):
		print('List of opposition pieces: ')
		print(piece_list)
		temp = input('Pick a piece to remove (counting from 0): ')
		
		return piece_list[int(temp)]
