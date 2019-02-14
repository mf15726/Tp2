#Packages
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import csv
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from math import log
import networkx as nx
import operator


adj_dict_3 = [[1,3,4],[0, 2, 4],[1, 4, 5],[0, 4, 6],[0, 1, 2, 3, 5, 6, 7, 8],[2, 4, 8],[7, 3, 4],[6, 8, 4],[7, 4, 5]]

adj_dict_6 = [[1, 6],[0, 2, 4],[1, 9],[4, 7],[1, 3, 5],[4, 8],[7, 0, 13],[10, 3, 6],[12, 5, 9],[2, 8, 15],[7, 11],[12, 14, 10],[8, 11],
	     [14, 6],[13, 15, 11],[9, 14]]

adj_dict_9 = [[1, 9],[0, 2, 4],[1, 14],[4, 10],[1, 3, 5, 7],[4, 13],[7, 11],[4, 6, 8],[12, 7],[0, 21, 10],[11, 18, 3, 9],[6, 15, 10],
	      [8, 17, 13, 14],[14, 20, 5],[2, 23, 13],[11, 15],[15, 17, 19],[12, 16],[10, 19],[18, 20, 10, 22],[19, 13],[9, 22],
	      [21, 23, 19],[22, 14]]

adj_dict_12 = [[1, 9, 3],[0, 2, 4],[1, 14, 5],[4, 10, 0, 6],[1, 3, 5, 7],[4, 13, 2, 8],[7, 11, 3],[4, 6, 8],[12, 7, 5],[0, 21, 10],
	       [11, 18, 3],[6, 15],[8, 17],[14, 20, 5],[2, 23, 13],[11, 15, 18],[15, 17, 19],[12, 16, 20],[10, 18, 15, 21],
	       [18, 20, 10, 22],[19, 13, 17, 20],[9, 22, 18],[21, 23, 19],[22, 14, 20]]

decision_type_to = [1,0,0]
decision_type_from = [0,1,0]
decision_type_remove = [0,0,1]

sym3 = [2,5,8,1,4,7,0,3,6]
sym6 = [2,9,15,5,8,12,1,4,11,14,3,7,10,0,6,13]
sym9 = [2,14,23,5,13,20,8,12,17,1,4,7,16,19,22,6,11,15,3,10,18,0,9,21]

class Learned_Player(object):
	
	def __init__(self, epsilon, alpha, gamma, limit):

		self.sess = tf.Session()
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.limit = limit
		self.state_index = []
		
		self.to_index = [(None, None, None)] * self.limit
		self.from_index = [(None, None, None)] * (self.limit - 6)
		self.remove_index = [(None, None, None)] * 19
		
		self.to_qval_index = [None] * self.limit
		self.from_qval_index = [None] * (self.limit - 6)
		self.remove_qval_index = [None] * 19

		self.n_classes = 24
		self.n_input = 79
		self.n_nodes_1 = self.n_classes * 2
		self.n_nodes_2 = self.n_classes * 2
		self.n_nodes_3 = self.n_classes * 2
		self.n_nodes_4 = self.n_classes * 2

		self.input = tf.placeholder(tf.float32, [24])
		self.x_p1 = tf.cast(tf.equal(self.input, 1), tf.float32)
		self.x_p2 = tf.cast(tf.equal(self.input, 2), tf.float32)
		self.x_empty = tf.cast(tf.equal(self.input, 0), tf.float32)
		
		#game_type = 1 at 0 if game_type = 3, 1 if 6, 2 if 9, 3 if 12
		self.game_type = tf.placeholder(tf.float32, [4])
#		self.game_type_list = [self.game_3,self.game_6,self.game_9,self.game_12]
#		self.x_game_type = tf.reshape(self.game_type, shape=[1,4])
		
		#decision_type = 1 at 0 if place, 1 if choose piece to move, 2 if move piece to, 3 if remove piece
		self.decision_type = tf.placeholder(tf.float32, shape=[3])
		
		self.ttemp = [self.x_empty,self.x_p1,self.x_p2]
#		self.tempp = [self.game_type,self.decision_type]
		self.tempp = tf.concat([self.game_type, self.decision_type], 0)
		self.tttemp = tf.reshape(self.ttemp, shape=[72])
		self.temppp = tf.reshape(self.tempp, shape=[7])
		self.x_bin = tf.concat([self.tttemp, self.temppp], 0)
		self.x = tf.reshape(self.x_bin, shape=[1,self.n_input])
		self.reward = tf.placeholder(tf.float32,[self.n_classes])
		self.y = tf.reshape(self.reward, [1, self.n_classes])
		self.Q_val = self.neural_network()
#		self.Q_val_from = self.neural_network_from()
		self.Q_val_stored = tf.placeholder(tf.float32, shape=[self.n_classes])
		#cost
		#        self.cost = tf.reduce_mean(tf.square(self.y - self.Q_val))
		#        self.cost = tf.square(self.Q_val - self.y)
		self.cost = tf.square(self.y - self.Q_val)
#		self.cost = tf.square(self.y - self.Q_val_stored)
#		self.cost_from = tf.square(self.y - self.Q_val_from)
		#optimiser

# 		 self.optimiser = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
		#        self.optimiser = tf.train.AdamOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
		self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost)
#		self.optimiser_from = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost_from)
		#        self.optimizer = tf.train.AdograadOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)

	def neural_network(self):

		l1 = tf.layers.dense(
			inputs=self.x,
			units=self.n_input,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

		l2 = tf.layers.dense(
			inputs=l1,
			units=self.n_nodes_1,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

#		l3 = tf.layers.dense(
	#		inputs=l2,
	#		units=self.n_nodes_2,
	#		bias_initializer=tf.constant_initializer(0, 1),
	#		activation=tf.nn.leaky_relu,
	#		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
	#		activity_regularizer=tf.nn.softmax
#		)

#		l4 = tf.layers.dense(
	#		inputs=l3,
	#		units=self.n_nodes_3,
	#		bias_initializer=tf.constant_initializer(0, 1),
	#		activation=tf.nn.leaky_relu,
	#		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
	#		activity_regularizer=tf.nn.softmax
#		)

#		l5 = tf.layers.dense(
	#		inputs=l4,
	#		units=self.n_nodes_4,
	#		bias_initializer=tf.constant_initializer(0, 1),
	#		activation=tf.nn.leaky_relu,
	#		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
	#		activity_regularizer=tf.nn.softmax
#		)

		l_out = tf.layers.dense(
			inputs=l2,
			units=self.n_classes,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

		l_norm = tf.contrib.layers.softmax(
			logits=l_out
		)
		
		return l_norm
		
		
	def piece_adj(self, state, game_type, space, pieces, player):
		piece_to_move = []
		
		
		if game_type == 3:
			for item in adj_dict_3[space]:
				if state[item] == player:
					piece_to_move.append(item)
					
		if game_type == 6:
			for item in adj_dict_6[space]:
				if state[item] == player:
					piece_to_move.append(item)
		
		if game_type == 9:
			for item in adj_dict_9[space]:
				if state[item] == player:
					piece_to_move.append(item)
					
		if game_type == 12:
			for item in adj_dict_12[space]:
				if state[item] == player:
					piece_to_move.append(item)
					
		if not piece_to_move:
			return False, None
		else:
			return True, piece_to_move
		
	def valid_move(self, state, game_type, free_space, pieces):
		valid_moves = []
		if game_type == 3:
			for piece in pieces:
				for space in adj_dict_3[piece]:
					if space in free_space:
						valid_moves.append((piece,space))

		if game_type == 6:
			for piece in pieces:
				for space in adj_dict_6[piece]:
					if space in free_space:
						valid_moves.append((piece,space))

		if game_type == 9:
			for piece in pieces:
				for space in adj_dict_9[piece]:
					if space in free_space:
						valid_moves.append((piece,space))

		if game_type == 12:
			for piece in pieces:
				for space in adj_dict_12[piece]:
					if space in free_space:
						valid_moves.append((piece,space))

		return valid_moves
		
	def random_place(self, state, free_space):
		temp = random.randint(0, len(free_space) - 1)
		return free_space[temp]
	
	def padding(self,state,game_type):
		temp = deepcopy(state)
		if game_type > 6:
			return temp
		if game_type == 3:
			temp.extend([0]*15)
		else:
			temp.extend([0]*8)
		return temp
	
	def convert_board(self, state, player):
		if player == 1:
			return state
		else:
			new_state = deepcopy(state)
			for item in new_state:
				item = (item % 2) + 1
			return new_state
		
	def place(self, state, free_space, game_type, player, move_no):
		rand = random.randint(1,100)
		move = None
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		input_state = self.convert_board(state,player)
		input_state = self.padding(input_state,game_type)
		predictions_to = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_to})
		
		if rand <= 100*self.epsilon:
			move = self.random_place(state,free_space)
			self.to_qval_index.append(predictions_to[0][0])
			self.to_index.append((deepcopy(input_state),move,player))
			return move
		else:
			opt_val = -float('Inf')
			for index, item in enumerate(free_space):
				if item is None:
					continue
				val = predictions_to[0][0][index]
#			for index, val in enumerate(predictions_to[0][0]):
				if val > opt_val:
					opt_val = val
					move = item
				if item == len(state):
					break
			self.to_qval_index[move_no] = predictions_to[0][0]
			self.to_index[move_no] = ((deepcopy(input_state),move,player))
			return move
	
	def random_move(self, valid_moves):
		temp = random.randint(0, len(valid_moves) - 1)
		return valid_moves[temp]
	
	
	def move(self, state, game_type, free_space, pieces, player, enable_flying, move_no):
		valid_moves = self.valid_move(state, game_type, free_space, pieces)
		if len(valid_moves) == 0 and not enable_flying:
			return (25, 25)
		move = None
		piece = None
		rand = random.randint(1,100)
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		input_state = self.convert_board(state,player)
		input_state = self.padding(input_state,game_type)
		predictions_to = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_to})
		if rand <= 100*self.epsilon:
			valid_moves = self.valid_move(state, game_type, free_space, pieces)
			random_move = self.random_move(valid_moves)
			predictions_from = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_from})
			self.to_index[move_no] = (deepcopy(input_state),random_move[0], player)
			self.from_index[int(move_no - (game_type * 2))] = (deepcopy(input_state),random_move[1],player)
			self.to_qval_index[move_no] = predictions_to[0][0]
			self.from_qval_index[int(move_no - (game_type * 2))] = predictions_from[0][0]
			return random_move
		else:
			opt_val = -float('Inf')
			if enable_flying:
				adj_piece_list = deepcopy(pieces)
				for index, item in enumerate(free_space):
					if item is None:
						continue
					val = predictions_to[0][0][index]
#				for index, val in enumerate(predictions_to[0][0]):
#					print('Index, Val ' +str(index) + ' ' + str(val))
					if val > opt_val:
						opt_val = val
						move = item
						continue
					if item == len(state):
						break
			else:
				for index, item in enumerate(free_space):
					if item is None:
						continue
					val = predictions_to[0][0][index]
#				for index, val in enumerate(predictions_to[0][0]):
#					print('Index, Val ' +str(index) + ' ' + str(val))
					if val > opt_val:
						adj_piece, _ = self.piece_adj(state, game_type, item, pieces, player)
						if adj_piece:
							adj_piece_list = deepcopy(_)
							opt_val = val
							move = item
							continue
					if item == len(state):
						break
						
			if move is None:
				return (25,25)
			
			predictions_from = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_from})
			
			opt_val = -float('Inf')
#			print('Adj Pieces ' +str(adj_piece_list))
			for item in adj_piece_list:
				val = predictions_from[0][0][item]
#			for index, val in enumerate(predictions_from[0][0]):
				if val > opt_val:
					opt_val = val
					piece = item
				if item == len(state):
					break
			if piece is None:
				print('THAT IS THE PROBLEm')
				return(25,25)
					
			predicted_move = (piece, move)
		self.to_index[move_no] = (deepcopy(input_state),piece,player)
		self.from_index[int(move_no - (game_type * 2))] = (deepcopy(input_state),move,player)
		self.to_qval_index[move_no] = predictions_to[0][0]
		self.from_qval_index[int(move_no - (game_type * 2))] = predictions_from[0][0]
		return predicted_move
	
	def remove_piece(self, state, piece_list, game_type, player, pieces_removed):
		rand = random.randint(1,100)
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		input_state = self.convert_board(state,player)
		input_state = self.padding(input_state,game_type)
		predictions_remove = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_remove})
		if rand <= 100*self.epsilon:
			temp = random.randint(0, len(piece_list) - 1)
			self.remove_index[pieces_removed] = (deepcopy(input_state),piece_list[temp],player)
			self.remove_qval_index[pieces_removed] = predictions_remove[0][0]
			return piece_list[temp]
		else:
			opt_val = -float('Inf')
			for index, item in enumerate(piece_list):
#				if item == None:
#					continue
				val = predictions_remove[0][0][item]
#			for index, val in enumerate(predictions_remove[0][0]):
				if val > opt_val:
					opt_val = val
					piece = item
					if item == len(state):
						break
			self.remove_index[pieces_removed] = (deepcopy(input_state),piece,player)
			self.remove_qval_index[pieces_removed] = predictions_remove[0][0]
		return piece
	
	def reward_function(self,game_type, winner, player, qval_index):
		if winner == player:
			reward = [1] * self.n_classes
		elif winner != 0:
			reward =  [-1] * self.n_classes
		else:
			reward = [0] * self.n_classes
		return list(map(operator.add, qval_index,reward))
	
	def learn(self, game_type, winner):
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		counter = 0
		for item in self.to_index:
			reward_to = self.reward_function(game_type,winner,item[2],self.to_qval_index[counter])
			self.sess.run([self.optimiser], feed_dict={self.reward: reward_to, self.input: item[0], self.game_type: game_type_input,
								   self.decision_type: decision_type_to})
			counter += 1 
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.place_qval_index})
		counter = 0
		for item in self.from_index:
			reward_from = self.reward_function(game_type,winner,item[2],self.from_qval_index[counter]) 
			self.sess.run([self.optimiser], feed_dict={self.reward: reward_from, self.input: item[0], self.game_type: game_type_input,
								   self.decision_type: decision_type_from})
			counter += 1
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.choose_qval_index})
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.move_qval_index})
		counter = 0
		for item in self.remove_index:
			reward_remove = self.reward_function(game_type,winner,item[2],self.remove_qval_index[counter])
			self.sess.run([self.optimiser], feed_dict={self.reward: reward_remove, self.input: item[0], self.game_type: game_type_input,
								   self.decision_type: decision_type_remove})
			counter += 1
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.place_remove_index})
			
			
		self.to_index = []
		self.from_index = []
		self.remove_index = []
		
		self.to_qval_index = []
		self.from_qval_index = []
		self.remove_qval_index = []
		
		return 0
