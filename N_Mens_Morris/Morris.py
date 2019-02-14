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
#Classes
from learned_player import Learned_Player
from random_player import Random_Player
from human_player import Human_Player
#from multi_task_player import Multi_Task_Player

#game_type = 3, 6, 9, 12 (Men's Morris)

mill_dict_3 = [[[1, 2], [3, 6], [4, 8]],
	       [[0, 2], [4, 7]],
	       [[5, 8], [4, 6], [0,1]],
	       [[0, 6], [4, 5]],
	       [[0, 8], [1, 7], [2, 6], [3, 5]],
	       [[2, 8], [4, 3]],
	       [[7, 8], [2, 4], [0, 3]],
	       [[6, 8], [1, 4]],
	       [[7, 6], [2, 5], [0, 4]]]


mill_dict_6 = [[[1, 2], [13, 6]],
	       [[0, 2]],
	       [[0, 1], [9, 15]],
	       [[10, 7], [4, 5]],
	       [[3, 5]],
	       [[12, 8], [4, 3]],
	       [[0, 13]],
	       [[10, 3]],
	       [[5, 12]],
	       [[2, 15]],
	       [[3, 7], [11, 12]],
	       [[10, 12]],
	       [[10, 11], [5, 8]],
	       [[0, 6], [14, 15]],
	       [[13, 15]],
	       [[2, 9], [13, 14]]]


mill_dict_9 = [[[1, 2], [9, 21]],[[0, 2], [7, 4]],[[0,  1], [14, 23]],[[18, 10], [4, 5]],[[3, 5], [1, 7]],[[13, 20], [4, 3]],
	       [[11, 15], [7, 8]],[[6, 8], [1, 4]],[[12, 17], [6, 7]],[[10, 11], [0, 21]],[[9, 11], [3, 18]],[[6, 15], [9, 11]],
	       [[8, 17], [13, 14]],[[12, 14], [5, 20]],[[2, 23], [12, 13]],[[6, 11], [16, 17]],[[15, 17], [19, 22]],
	       [[12, 8], [15, 16]],[[10, 3], [19, 20]],[[18, 20], [10, 3]],[[19, 18], [5, 13]],[[9, 0], [22, 23]],[[21, 23], [16, 19]],
	       [[21, 22], [2, 14]]]

mill_dict_12 = [[[1, 2], [9, 21], [3, 6]],[[0, 2], [7, 4]],[[0, 1], [14, 23], [5, 8]],[[18, 10], [4, 5], [0, 6]],[[3, 5], [1, 7]],
		[[13, 20], [4, 3], [2, 8]],[[11, 15], [7, 8], 9],[[6, 8], [1, 4]],[[12, 17], [6, 7], 20],[[10, 11], [0, 21]],
		[[9, 11], [3, 18]],[[6, 15], [9, 11]],[[8, 17], [13, 14]],[[12, 14], [5, 20]],[[2, 23], [12, 13]],
		[[6, 11], [16, 17], [18, 21]],[[15, 17], [19, 22]],[[12, 8], [15, 16], [20, 23]],[[10, 3], [19, 20], [15, 21]],
		[[18, 20], [10, 3]],[[19, 18], [5, 13], [17, 23]], [[9, 0], [22, 23],[15, 18]],[[21, 23], [16, 19]],
		[[21, 22],[2, 14], [17, 20]]]


def printboard(game_type,state):
#    gameboard = nx.graph()
	if game_type == 3:
		print(str(state[0])+'-'+str(state[1])+'-'+str(state[2]))
		print('|\|/|')
		print(str(state[3])+'-'+str(state[4])+'-'+str(state[5]))
		print('|/|\|')
		print(str(state[6])+'-'+str(state[7])+'-'+str(state[8]))

	if game_type == 6:
		print(str(state[0])+'---'+str(state[1])+'---'+str(state[2]))
		print('|   |   |')
		print('| '+str(state[3])+'-'+str(state[4])+'-'+str(state[5])+' |')
		print('| |   | |')
		print(str(state[6])+'-'+str(state[7])+'   '+str(state[8])+'-'+str(state[9]))
		print('| |   | |')
		print('| '+str(state[10])+'-'+str(state[11])+'-'+str(state[12])+' |')
		print('|   |   |')
		print(str(state[13])+'---'+str(state[14])+'---'+str(state[15]))

	if game_type == 9:
		print(str(state[0])+'-----'+str(state[1])+'-----'+str(state[2]))
		print('|     |     |')
		print('| '+str(state[3])+'---'+str(state[4])+'---'+str(state[5])+' |')
		print('| |   |   | |')
		print('| | '+str(state[6])+'-'+str(state[7])+'-'+str(state[8])+' | |')
		print('| | |   | | |')
		print(str(state[9])+'-'+str(state[10])+'-'+str(state[11])+'   '+str(state[12])+'-'+str(state[13])+'-'+str(state[14]))
		print('| | |   | | |')
		print('| | '+str(state[15])+'-'+str(state[16])+'-'+str(state[17])+' | |')
		print('| |   |   | |')
		print('| '+str(state[18])+'---'+str(state[19])+'---'+str(state[20])+' |')
		print('|     |     |')
		print(str(state[21])+'-----'+str(state[22])+'-----'+str(state[23]))

	if game_type == 12:
		print(str(state[0])+'-----'+str(state[1])+'-----'+str(state[2]))
		print('|\    |    /|')
		print('| '+str(state[3])+'---'+str(state[4])+'---'+str(state[5])+' |')
		print('| |\  |  /| |')
		print('| | '+str(state[6])+'-'+str(state[7])+'-'+str(state[8])+' | |')
		print('| | |   | | |')
		print(str(state[9])+'-'+str(state[10])+'-'+str(state[11])+'   '+str(state[12])+'-'+str(state[13])+'-'+str(state[14]))
		print('| | |   | | |')
		print('| | '+str(state[15])+'-'+str(state[16])+'-'+str(state[17])+' | |')
		print('| |/  |  \| |')
		print('| '+str(state[18])+'---'+str(state[19])+'---'+str(state[20])+' |')
		print('|/    |    \|')
		print(str(state[21])+'-----'+str(state[22])+'-----'+str(state[23]))



def end_game(state):
	count1 = state.count(1)
	count2 = state.count(2)
#	print(('Count1 = ') + str(count1))
#	print(('Count2 = ') + str(count2))
	if count1 <= 2:
		return 2
	if count2 <= 2:
		return 1
	else:
		return 0

def det_mill(state, move, game_type):
	if game_type == 3:
		for item in mill_dict_3[move]:
			if state[move] == state[item[0]] == state[item[1]]:
				return True
			else:
				return False

	if game_type == 6:
		print(move)
		for item in mill_dict_6[move]:
			if state[move] == state[item[0]] == state[item[1]]:
				return True
			else:
				return False


	if game_type == 9:
		for item in mill_dict_9[move]:
			if state[move] == state[item[0]] == state[item[1]]:
				return True
			else:
				return False


	if game_type == 12:
		for item in mill_dict_12[move]:
			if state[move] == state[item[0]] == state[item[1]]:
				return True
			else:
				return False

def free_space_finder(state):
	free_space = []
	for i in range(len(state)):
		if state[i] == 0:
			free_space.append(i)

	return free_space

#def free_space_finder(state):
#	free_space = [0] * 9
#	for i in range(len(state)):
		

def flying_check(state, player):
	if game_type == 3:
		return False
	count = state.count(player)
	if count == 3:
		return True
	else:
		return False

def game_play(player1,player2,game_type,print_board,flying,limit):
	winner = 0
	move_no = 0
	player1_piece_list = [None] * game_type
	player2_piece_list = [None] * game_type
	p1_pieces_removed = 0
	p2_pieces_removed = 0
	if game_type == 3:
		state = [0,0,0,0,0,0,0,0,0]
#		free_space = [1,1,1,1,1,1,1,1,1]
	elif game_type == 6:
		state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#		free_space = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
	else:
		state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#		free_space = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
	if print_board:
		printboard(game_type,state)
	free_space = free_space_finder(state)
	while winner == 0:
		player = (move_no % 2) + 1
		if move_no < game_type * 2:
			if player == 1:
				move = player1.place(state,free_space,game_type,player,move_no)
				player1_piece_list[int(move_no/2)] = move
			else:
				move = player2.place(state,free_space,game_type,player,move_no)
				print(int((move_no - 1)/2))
				player2_piece_list[int((move_no - 1)/2)] = move
			state[move] = player
			_ = free_space.index(move)
#			print('Placed by Player ' + str(player) + ' ' +  str(move))
#			print('Free Space = ' +str(free_space))
			free_space.remove(move)
			if print_board:
				printboard(game_type,state)
			if det_mill(state, move, game_type):
				if game_type == 3:
					return player
#				print('Mill Created')
				if player == 1:
					removed_piece = player1.remove_piece(state,player2_piece_list,game_type,player,p2_pieces_removed)
#					print('P2 Plist = ' + str(player2_piece_list))
#					print('Removed piece = ' + str(removed_piece))
					state[removed_piece] = 0
					p2_pieces_removed += 1
					player2_piece_list.remove(removed_piece)
				else:
					removed_piece = player2.remove_piece(state,player1_piece_list,game_type,player,p1_pieces_removed)
#					print('P1 Plist = ' + str(player1_piece_list))
#					print('Removed piece = ' + str(removed_piece))
					state[removed_piece] = 0
					p1_pieces_removed += 1
					player1_piece_list.remove(removed_piece)
				free_space.append(removed_piece)
				if print_board:
					printboard(game_type,state)
		else:
			if move_no == game_type * 2:
				winner = end_game(state)
				if winner != 0:
					return winner
				if flying:
					p1_fly = flying_check(state,1)
					p2_fly = flying_check(state,2)
			if player == 1:
				prev_pos, move = player1.move(state,game_type,free_space,player1_piece_list,player,p1_fly,move_no)
				if move == 25:
					return 2
				player1_piece_list.append(move)
				player1_piece_list.remove(prev_pos)
#				print('Player1 moves' + str(move))
#				print('From ' + str(prev_pos))
			else:
				prev_pos, move = player2.move(state,game_type,free_space,player2_piece_list,player,p2_fly,move_no)
				if move == 25:
					return 1
				player2_piece_list.append(move)
				player2_piece_list.remove(prev_pos)
#				print('Player2 moves' + str(move))
#				print('From ' + str(prev_pos))
			state[move] = player
			state[prev_pos] = 0
			free_space.remove(move)
			free_space.append(prev_pos)
			if print_board:
				printboard(game_type,state)
			if det_mill(state, move, game_type):
#				print('Mill Created')
				if player == 1:
					removed_piece = player1.remove_piece(state,player2_piece_list,game_type,player,p2_pieces_removed)
#					print('P2 Plist = ' + str(player2_piece_list))
#					print('Removed piece = ' + str(removed_piece))
					state[removed_piece] = 0
					p2_pieces_removed += 1
					player2_piece_list.remove(removed_piece)
					if flying:
						p1_fly = flying_check(state,1)
				else:
					removed_piece = player2.remove_piece(state,player1_piece_list,game_type,player,p1_pieces_removed)
#					print('P1 Plist = ' + str(player1_piece_list))
#					print('Removed piece = ' + str(removed_piece))
					state[removed_piece] = 0
					p1_pieces_removed += 1
					player1_piece_list.remove(removed_piece)
					if flying:
						p2_fly = flying_check(state,2)
				free_space.append(removed_piece)
				if print_board:
					printboard(game_type,state)
				winner = end_game(state)
		move_no += 1
		if move_no == limit:
			return 0




	return winner

winner_list = []
enable_flying = True
game_type = 6
see_board = True
total_move_no = 1000

human_player = Human_Player()
random_player = Random_Player()
learned_player = Learned_Player(epsilon=0.01, alpha=0.3, gamma=0.9, limit=total_move_no)
learned_player.sess.run(tf.global_variables_initializer())
#gameboard = define_board(6)
#nx.draw(gameboard)
#plt.show()
for i in range(100):
	if i%2 == 0:
		print('Game Number = ' +str(i+1))
	winner = game_play(random_player,random_player, game_type, see_board, enable_flying, total_move_no)
#	winner = game_play(random_player,random_player, 12, False)
	print('Winner of game ' + str(i+1) + ' is Player ' + str(winner))
	winner_list.append(winner)
#	learned_player.learn(game_type, winner)
print('P1 wins = ' + str(winner_list.count(1)))
print('P2 wins = ' + str(winner_list.count(2)))
