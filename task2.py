import numpy as np
import numpy.linalg


class Rosenbrock():

    def count_gradient(self, x, y, z):
        f1 = 4 * x * (x ** 2 + y) + 800 * x * (200 * x ** 2 - 200 * y + z) + 8 * (1 + 600 * x ** 2 - 200 * y - z) * (
                    -1 + 200 * x ** 3 - x * (-1 + 200 * y + z))
        f2 = 1600 * x - 320000 * x ** 4 + 80002 * y - 400 * z + 2 * x ** 2 * (-40799 + 160000 * y + 800 * z)
        f3 = 2 * (4 * x - 800 * x ** 4 - 200 * y + z + 4 * x ** 2 * (49 + 200 * y + z))
        return np.array([[f1], [f2], [f3]])

    def get_coordinates(self, x, y, lx):
        return np.array([[x], [y], [lx]])


if __name__ == '__main__':
    points = [[3, 2, 0], [1, 2, 1], [0.5, 0.4, 0.9], [-2, -0.4, 2], [0, 0, 0]]
    EPS = 0.0000001
    # точки на вход, погрешность
    target = Rosenbrock()

    for a in points:
        x = a[0]
        y = a[1]
        lx = a[2]
        step = 1e-7
        count = 0
        while True:
            F = target.count_gradient(x, y, lx)
            X = target.get_coordinates(x, y, lx)
            nextIter = X - (step * F)

            x, y, lx = nextIter.flatten()
            FX1 = target.count_gradient(x, y, lx)
            right_mx = FX1 - F

            if (numpy.linalg.norm(FX1)) < EPS:
                print("Point: ", a[0], " ", a[1], " ", a[2], " | Iteration: ", count, " | Result: ", nextIter)
                break

            left_mx = np.transpose(nextIter - X)
            mx = left_mx @ right_mx
            step = (abs(mx[0]) / (numpy.linalg.norm(FX1 - F) ** 2))
            count += 1
