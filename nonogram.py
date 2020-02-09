from math import factorial as fct
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class nonogram():
    def __init__(self, csv_file: str ='./nonogram.csv', data: list =[], row_possibilities_limit=100000):
        if len(csv_file) > 0:
            arr = pd.read_csv(csv_file, header=None, dtype='object').values
            vectors = np.array([arr[1:,0], arr[0,1:]])
        else:
            vectors = data
        self.dim = [len(v) for v in vectors]
        rows = []
        for i in range(len(vectors)):
            for j in range(len(vectors[i])):
                vec = [int(v) for v in vectors[i][j].split(' ')]
                rows.append(row(i, j, self.dim[::-1][i], vec))
        self.rows = rows
        self.row_possibilities_limit = row_possibilities_limit
        self.gameboard = np.zeros(self.dim)
    def solve(self, show=True, verbose=False):
        counter = 1
        while 0 in self.gameboard:
            # rank possibilities
            # send rows in order of increasing possibility size. 
            possibilities = [min(len(r.poss_mat),r.total_poss) 
                             if len(r.poss_mat) > 0 else r.total_poss 
                             for r in self.rows]
            order = [[possibilities[i],i] for i in range(len(possibilities))]
            order.sort(key=lambda l:l[0])
            if verbose:
                print('Iteration: {}\tSum of row possibilities: {}'.format(counter, sum(possibilities)))
            for index in order:
                r = self.rows[index[1]]
                # if poss_mat already generated: 
                    # check_gameboard
                    # update_gameboard
                # else: 
                    # generate poss_mat
                    # check if poss_mat -> check_gameboard -> len<100,000 [or some limit]:
                    # else del poss_mat then continue
                if len(r.poss_mat) > 0:
                    r.check_gameboard(self.gameboard)
                    self.gameboard = r.update_gameboard(self.gameboard)
                else:
                    r.poss_mat = r.generate_poss_mat()
                    r.check_gameboard(self.gameboard)
                    if r.poss_mat.shape[0] == 0:
                        print('Error: puzzle values seem to be wrong.')
                        return
                    elif r.poss_mat.shape[0] > self.row_possibilities_limit:
                        del r.poss_mat
                        r.poss_mat = []
                    else:
                        self.gameboard = r.update_gameboard(self.gameboard)
            counter += 1
            if verbose:
                self.show_board()
        if show:
            self.show_board()
    def show_board(self, no_labels=True):
        ax = sns.heatmap(self.gameboard * -1, cbar=False)
        if no_labels:
            ax.tick_params(
                left=False, 
                labelleft=False,
                bottom=False, 
                labelbottom=False)
        plt.show()

class row():
    def __init__(self, axis, index, length, vector):
        self.axis = axis
        self.index = index
        self.length = length
        self.vector = vector
        self.total_poss = self.compute_possibilities()
        self.poss_mat = []
    def compute_possibilities(self):
        len_row = self.length
        len_vec, sum_vec = len(self.vector), sum(self.vector)
        # translate current problem to "how many ways to distribute spaces(blank boxes)
        # to the slots between the blocks of filled squares?" 
        # [compulsory space of at least 1 between the blocks]
        # =>
        # classic combinatorics problem: how many ways to put b X balls into c X containers?
        balls = len_row - sum_vec - len_vec + 1
        containers = len_vec + 1
        elements = balls + containers - 1
        partitions = containers - 1
        return int(fct(elements) / (fct(balls)*fct(partitions)))
    def generate_poss_mat(self):
        l, vec = self.length, self.vector
        spaces = l - sum(vec) - len(vec) + 1
        slots = len(vec)
        def generate_possibilities(spaces, slots):
            mat = np.array([np.zeros(slots)]).astype(int)
            for i in range(slots):
                mat = np.repeat(mat, spaces + 1, axis=0)
                mat[:,i] = [i% (spaces + 1) for i in range(len(mat))]
                mat = mat[np.sum(mat, axis=1) <= spaces]
            return mat
        spaces_mat = np.array(generate_possibilities(spaces, slots))
        # spaces_mat = possible ways for total num of spaces(blanks) 
        # to distribute to the slots between the blocks of filled squares. 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # n x blocks will give n x slots 
        # last slot between last block and border not needed to specify 
        # because remaining spaces left will all go there. 
        m = len(spaces_mat)
        possibility_matrix = np.zeros((m, l)) - 1
        for j in range(spaces_mat.shape[1]):
            start_pos = np.sum(spaces_mat[:,:j+1], axis=1) + np.sum(vec[:j]) + j
            for k in range(vec[j]):
                possibility_matrix[range(m), start_pos.astype(int) + k] = 1
        return possibility_matrix.astype(int)
    def check_gameboard(self, gameboard):
        # Checks possible values against gameboard row. 
        # Multiply gameboard row to each row of poss_mat
        # Rows with -1 are wrong
        gameboard_row = gameboard[self.index,:] if self.axis == 0 else gameboard[:,self.index]
        evaluation = self.poss_mat * gameboard_row
        self.poss_mat = self.poss_mat[np.amin(evaluation,axis=1) != -1]
    def update_gameboard(self, gameboard):
        poss_vec = np.sum(self.poss_mat, axis=0)/self.poss_mat.shape[0]
        for k in range(len(poss_vec)):
            if np.abs(poss_vec[k]) == 1:
                coordinates = [k, k]
                coordinates[self.axis] = self.index
                i, j = coordinates
                gameboard[i,j] = poss_vec[k]
        return gameboard