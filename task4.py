import numpy as np
import numpy.linalg
from scipy.linalg import lstsq
import scipy.linalg
from scipy.linalg import null_space


class Rosenbrock():
    def count_gradient(self, x, y):
        f1 = (400 * (x ** 3) - 400 * x * y + 2 * x - 2)
        f2 = (-200 * (x ** 2) + 200 * y)
        return np.array([[f1], [f2]])

    def get_coordinates(self, x, y, lx):
        return np.array([[x], [y], [lx]])

    def count_gradient_vector(self, vec):
        return self.count_gradient(vec[0, 0], vec[1, 0])


if __name__ == '__main__':
    matrix = []
    vector = []

    print("Enter rows in format 'a1 a2 ... an | b'. For stop writing enter the empty line:")
    while True:
        line = input()
        if line.strip() == "":
            break
        left_part, right_part = line.split("|")
        matrix.append([int(x) for x in left_part.split()])
        vector.append(int(right_part.strip()))

    matrix = np.array(matrix)
    vector = np.array(vector)

    sol = lstsq(matrix, vector)
    x = np.array(sol[0]).reshape(len(sol[0]), 1)
    kernel = null_space(matrix)
    rank = np.linalg.matrix_rank(kernel)

    t_vec = np.random.randint(0, 10, size=rank)
    print("The start vector is ", t_vec)
    points = [t_vec]
    EPS = 0.0000001
    target = Rosenbrock()

    for a in points:
        step = 1e-4
        count = 0
        while True:
            right_part = kernel @ t_vec
            mx = x + right_part.reshape(len(right_part), 1)
            F = np.transpose(target.count_gradient_vector(mx)) @ kernel
            nextIter = t_vec - (step * F)

            print("Iteration: ", count, " | Result: ", nextIter)

            if (numpy.linalg.norm(nextIter - t_vec)) < EPS:
                print("FINISH Iteration: ", count, " | Result: ", nextIter)
                break
            t_vec = nextIter
            count += 1
