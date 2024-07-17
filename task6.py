import math
import numpy as np


class Simplex:

    def simplex_method(self, dual: bool):
        stb = self.simplex_table
        ix_vec = self.ix_vec
        rw = stb.shape[0]
        cl = stb.shape[1]

        optimal = False
        while not optimal:
            if not dual:
                costs = [stb[0][c] for c in range(1, cl)]
            else:
                costs = [stb[c][0] for c in range(1, rw)]

            if all(c >= 0 for c in costs):
                optimal = True
                break
            else:
                mx = 1e7
                row = 1
                col = 1
                if not dual:
                    for i in range(1, cl):
                        if stb[0][i] < 0:
                            col = i
                            break
                else:
                    for i in range(1, rw):
                        if stb[i][0] < 0:
                            row = i
                            break
                if not dual:
                    for i in range(1, rw):
                        if stb[i][col] > 0 and stb[i][0] / stb[i][col] < mx:
                            mx = stb[i][0] / stb[i][col]
                            row = i
                else:
                    for i in range(1, cl):
                        if stb[row][i] < 0 and stb[0][i] / abs(stb[row][i]) < mx:
                            mx = stb[0][i] / abs(stb[row][i])
                            col = i

                pivot = stb[row][col]
                for i in range(cl):
                    stb[row][i] /= pivot
                for i in range(rw):
                    if stb[i][col] != 0 and i != row:
                        cf = stb[i][col]
                        for j in range(cl):
                            stb[i][j] -= cf * stb[row][j]
                ix_vec[row - 1] = col

        if optimal:
            x = [0] * (cl - 1)
            for i, b in enumerate(ix_vec):
                x[b - 1] = stb[i + 1][0]
                self.is_optimal = True
            self.solution_vec = x
            self.F = -stb[0][0]

    def apply_gomory_cut(self):
        stb = self.simplex_table
        ix_vec = self.ix_vec

        row = stb.shape[0]
        col = stb.shape[1]

        chosen_rw = 0
        for i in range(1, row):
            if math.modf(stb[i][0])[0] != 0:
                chosen_rw = i
                break

        stb = np.vstack((stb, np.array([0.] * col)))
        stb = np.hstack((stb, np.zeros((row + 1, 1))))
        stb[row][col] = 1.

        row = stb.shape[0]
        col = stb.shape[1]
        ix_vec.append(col - 1)
        for i in range(col - 1):
            if stb[chosen_rw][i] < 0:
                stb[row - 1][i] = -(stb[chosen_rw][i] - int(stb[chosen_rw][i] - 1))
            else:
                stb[row - 1][i] = -(stb[chosen_rw][i] - int(stb[chosen_rw][i]))

        self.simplex_table = stb
        self.ix_vec = ix_vec

    def run_simplex(self):
        self.simplex_method(False)
        values = [self.simplex_table[i][0] for i in range(1, self.simplex_table.shape[0])]
        is_integer_solv = all(elem == int(elem) for elem in values)
        if self.is_integer_task:
            self.is_optimal = False
            while not is_integer_solv:
                self.apply_gomory_cut()
                self.simplex_method(True)
                values = [self.simplex_table[i][0] for i in range(1, self.simplex_table.shape[0])]
                is_integer_solv = all(item == int(item) for item in values)
        print("Optimal solution is =", self.solution_vec, "\nOptimal value is F =", self.F)

    def __init__(self, is_int, simplex_table, ix_vec):
        self.F = 0
        self.is_optimal = False

        self.simplex_table = simplex_table
        self.ix_vec = ix_vec
        self.is_integer_task = is_int
        self.solution_vec = []


if __name__ == '__main__':
    t0 = np.array([[0., -4., -5., -6.],
                   [35., 1., 2., 3.],
                   [45., 2., 3., 2.],
                   [40., 3., 1., 1.]])

    t1 = np.array([[0., -4., -5., -6.],
                   [35., 1., 2., 3.],
                   [45., 4., 3., 2.],
                   [40., 4., 1., 1.]])
    ix = [1, 2, 3]

    Simplex(is_int=False, simplex_table=t0, ix_vec=ix).run_simplex()
    Simplex(is_int=True, simplex_table=t1, ix_vec=ix).run_simplex()
