import numpy as np

class Rosenbrock():
    def count_jacobian(self, x, y, lx):
        j11 = 1200 * (x ** 2) - 400 * y + 2 * lx + 2
        j12 = -400 * x
        j13 = 2 * x
        j21 = -400 * x
        j22 = 200
        j23 = 1
        j31 = 2 * x
        j32 = 1
        j33 = 0
        return np.array([[j11, j12, j13], [j21, j22, j23], [j31, j32, j33]])

    def count_gradient(self, x, y, lx):
        f1 = (400 * (x ** 3) - 400 * x * y + 2 * x * lx + 2 * x - 2)
        f2 = (-200 * (x ** 2) + 200 * y + lx)
        f3 = (y + (x ** 2))
        return np.array([[f1], [f2], [f3]])

    def get_coordinates(self, x, y, lx):
        return np.array([[x], [y], [lx]])


if __name__ == '__main__':
    points = [[3, 2, 0], [1, 2, 1], [0.5, 0.4, 0.9], [-2, -0.4, 2], [0, 0, 0]]
    EPS = 0.00000001

    target = Rosenbrock()
    for a in points:
        x = a[0]
        y = a[1]
        lx = a[2]

        count = 0
        while True:
            J = target.count_jacobian(x, y, lx)
            F = target.count_gradient(x, y, lx)
            X = target.get_coordinates(x, y, lx)
            J_inv = np.linalg.inv(J)
            nextIter = X - (J_inv @ F)
            if (abs(nextIter[0, 0] - x) < EPS and abs(nextIter[1, 0] - y) < EPS and abs(nextIter[2, 0] - lx) < EPS):
                print("Point: ", a[0], " ", a[1], " ", a[2], " | Iteration: ", count, " | Result: ", nextIter)
                break
            x = nextIter[0, 0]
            y = nextIter[1, 0]
            lx = nextIter[2, 0]
            count += 1
