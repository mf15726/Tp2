#Maybe delete this eventually?
def detect_mill(state, move, game_type):
    if game_type == 3:
        if move == [0,0] or move == [1,1] or move == [2,2]:
            if state[0][0] == state[1][1] == state[2,2]:
                return True
        if move == [0,2] or move == [1,1] or move == [2,0]:
            if state[0][2] == state[1][1] == state[2][0]:
                return True
        if state[0][move[1]] == state[1][move[1]] == state[2][move[1]]:
            return True
        if state[move[0]][0] == state[move[0]][1] == state[move[0]][2]:
            return True
        else:
            return False


    if game_type == 6:
        if move[0] != 2:
            if state[move[0]][0] == state[move[0]][1] == state[move[0]][2]:
                return True
            if move[1] != 1:
                if move[0] % 2 == 0:
                    if state[0][move[1]] == state[2][(move[1] * 4) % 5] == state[4][move[1]]:
                        return True
                else:
                    if state[1][move[1]] == state[2][(move[1] * 4) % 7] == state[3][move[1]]:
                        return True
        else:
            if move[1] % 3 == 0:
                if state[0][move[1]] == state[2][(move[1] * 4) % 5] == state[4][move[1]]:
                    return True
                if state[1][move[1]] == state[2][(move[1] * 4) % 7] == state[3][move[1]]:
                    return True
        return False

    if game_type == 9:
        if move[0] != 3:
            if state[move[0]][0] == state[move[0]][1] == state[move[0]][2]:
                return True
            if move[1] == 1:
                if move[0] < 3:
                    if state[0][move[1]] == state[1][move[1]] == state[2][move[1]]:
                        return True
                else:
                    if state[4][move[1]] == state[5][move[1]] == state[6][move[1]]:
                        return True
        if move[0] == 3:
            if move[1] < 3:
                if state[move[0]][0] == state[move[0]][1] == state[move[0]][2]:
                    return True
            else:
                if state[move[0]][3] == state[move[0]][4] == state[move[0]][5]:
                    return True
        if move == [0,0] or move == [3,0] or move == [6,0]:
            if state[0][0] == state[3,0] == state[6,0]:
                return True
            continue
        if move == [1,0] or move == [3,1] or move == [5,0]:
            if state[1][0] == state[3][1] == state[5][0]:
                return True
            continue
        if move == [2,0] or move == [3,2] or move == [4,0]:
            if state[2][0] == state[3][2] == state[4][0]:
                return True
            continue
        if move == [2,2] or move == [3,3] or move ==[4,2]:
            if state[2][2] == state[3][3] == state[4][2]:
                return True
            continue
        if move == [1,2] or move == [3,4] or move == [5,2]:
            if state[1][2] == state[3][4] == state[5][2]:
                return True
            continue
        if move == [0,2] or move == [3,5] or move == [6,2]:
            if state[0][2] == state[3][5] == state[6][2]:
                return True

        return False

    if game_type == 12:
        if move[0] != 3:
            if state[move[0]][0] == state[move[0]][1] == state[move[0]][2]:
                return True
            if move[1] != 1:
                if move[0] < 3:
                    if state[0][move[1]] == state[1][move[1]] == state[2][move[1]]:
                        return True
                else:
                    if state[4][move[1]] == state[5][move[1]] == state[6][move[1]]:
                        return True
            else:
                if move[0] < 3:
                    if state[0][move[1]] == state[1][move[1]] == state[2][move[1]]:
                        return True
                else:
                    if state[4][move[1]] == state[5][move[1]] == state[6][move[1]]:
                        return True
        if move[0] == 3:
            if move[1] < 3:
                if state[move[0]][0] == state[move[0]][1] == state[move[0]][2]:
                    return True
            else:
                if state[move[0]][3] == state[move[0]][4] == state[move[0]][5]:
                    return True

        if move == [0,0] or move == [3,0] or move == [6,0]:
            if state[0][0] == state[3,0] == state[6,0]:
                return True
            continue
        if move == [1,0] or move == [3,1] or move == [5,0]:
            if state[1][0] == state[3][1] == state[5][0]:
                return True
            continue
        if move == [2,0] or move == [3,2] or move == [4,0]:
            if state[2][0] == state[3][2] == state[4][0]:
                return True
            continue
        if move == [2,2] or move == [3,3] or move ==[4,2]:
            if state[2][2] == state[3][3] == state[4][2]:
                return True
            continue
        if move == [1,2] or move == [3,4] or move == [5,2]:
            if state[1][2] == state[3][4] == state[5][2]:
                return True
            continue
        if move == [0,2] or move == [3,5] or move == [6,2]:
            if state[0][2] == state[3][5] == state[6][2]:
                return True

        return False
