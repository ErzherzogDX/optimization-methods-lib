import numpy as np
import numpy.linalg

from collections import deque


class Rosenbrock():
    def count_gradient(self, v):
        grad = []
        n = len(v)
        ax = v.flatten()
        for i in range(1, n // 2 + 1):
            x = ax[i * 2 - 2]
            y = ax[i * 2 - 1]
            grad.append([400 * (x ** 3) - 400 * x * y + 2 * x - 2])
            grad.append([200 * y - 200 * (x ** 2)])
        return np.array(grad)

    def get_fun_value(self, v):
        res = 0
        n = len(v)
        for i in range(1, n // 2 + 1):
            x = v[i * 2 - 2]
            y = v[i * 2 - 1]
            res += 100 * ((x ** 2 - y) ** 2) + ((x - 1) ** 2)
        return res


def LFBGS(M, points, target):
    EPS = 0.00001
    for a in points:
        mx = np.array(a).reshape(-1, 1)
        s_deq = deque()
        y_deq = deque()

        prev_x = mx.copy()
        q = target.count_gradient(prev_x)

        current_x = mx + (EPS * q)
        s_deq.append(current_x - prev_x)
        y_deq.append(target.count_gradient(current_x) - target.count_gradient(prev_x))
        count = 0

        while True:
            count += 1
            if np.linalg.norm(current_x - prev_x) < EPS:
                break

            q = target.count_gradient(current_x)
            ro = []
            gm = []

            for i in range(len(s_deq) - 1, max(0, len(s_deq) - M) - 1, -1):
                denom_pk = np.transpose(y_deq[i]) @ s_deq[i]
                ro.append(1 / denom_pk[0, 0])

                sq = np.transpose(s_deq[i]) @ q
                gm.append(ro[-1] * sq[0, 0])
                q = q - gm[-1] * y_deq[i]

            sy = np.transpose(s_deq[-1]) @ y_deq[-1]
            yy = np.transpose(y_deq[-1]) @ y_deq[-1]
            r = q * (sy[0, 0] / yy[0, 0])

            it = len(ro) - 1
            for i in range(max(0, len(s_deq) - M), len(s_deq)):
                val_yr = np.transpose(y_deq[i]) @ r
                bx = ro[it] * val_yr[0, 0]
                r = r + s_deq[i] * (gm[it] - bx)
                it -= 1

            p = r * (-1)

            left_b = 0
            right_b = 1e9
            c1 = 0.1
            c2 = 0.9

            alpha_coeff = 1

            while (right_b - left_b > EPS):
                shifted_x = current_x + p * alpha_coeff
                grad = target.count_gradient(current_x)
                d = np.transpose(p) @ grad

                grad2 = target.count_gradient(shifted_x)
                d2 = np.transpose(p) @ grad

                if target.get_fun_value(shifted_x) > target.get_fun_value(current_x) + c1 * alpha_coeff * d[0, 0]:
                    right_b = alpha_coeff
                    alpha_coeff = (left_b + right_b) / 2
                elif d2[0, 0] < c2 * d[0, 0]:
                    left_b = alpha_coeff
                    if right_b == 1e9:
                        alpha_coeff = 2 * left_b
                    else:
                        right_b = (left_b + right_b) / 2
                else:
                    break

            prev_x = current_x
            current_x = prev_x + p * alpha_coeff
            s_deq.append(current_x - prev_x)
            y_deq.append(target.count_gradient(current_x) - target.count_gradient(prev_x))

            if len(s_deq) > M:
                s_deq.popleft()
                y_deq.popleft()

        print("Result L-BFGS for", a, "is \n", current_x)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    points = [[1, 2, 3, 4],
              [3, 2, 0, 4, 10, 5, -2, -6, -12, 4],
              [0.1, 0.3, 33, -92.2, 44, 102, 88, -3.3, 0, 12, 0, 13, 13, -47, 5.88, 7.3]]
    EPS = 0.00001
    M = 4
    target = Rosenbrock()
    LFBGS(M, points, target)
