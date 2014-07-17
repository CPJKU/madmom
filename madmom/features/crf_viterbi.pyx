import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def viterbi(np.ndarray[np.float32_t, ndim=1] pi,
            np.ndarray[np.float32_t, ndim=1] transition,
            np.ndarray[np.float32_t, ndim=1] norm_factor,
            np.ndarray[np.float32_t, ndim=1] activations,
            int tau):

    cdef int num_st = activations.shape[0]
    cdef int num_x = num_st / tau

    cdef list vs = []
    cdef list bps = []
    cdef np.ndarray[np.float32_t, ndim=1] v_c = np.zeros(num_st, dtype=np.float32)
    cdef np.ndarray[np.int_t, ndim=1] bp_c = np.ones_like(v_c, dtype=int)
    cdef np.ndarray[np.float32_t, ndim=1] v_p
    cdef list path = []

    cdef int k, i, j
    cdef double cur, new, sum_k, log_sum = 0.0

    bp_c[:] = -1

    v_c = pi * activations
    v_c /= v_c.sum()
    v_p = v_c.copy()
    vs.append(v_p)

    for k in range(num_x - 1):
        for i in range(num_st):
            v_c[i] = 0.0

        for i in range(num_st):
            for j in range(transition.shape[0]):
                if (i + j) >= num_st:
                    break

                cur = v_c[i + j]
                new = v_p[i] * transition[j] * activations[i + j] * norm_factor[i]

                if new > cur:
                    v_c[i + j] = new
                    bp_c[i + j] = i

        sum_k = 0.0
        for i in range(num_st):
            sum_k += v_c[i]

        for i in range(num_st):
            v_c[i] /= sum_k

        log_sum += log(sum_k)

        v_p = v_c.copy()
        vs.append(v_p)
        bps.append(bp_c.copy())

    next_state = vs[num_x - 1].argmax()
    path.append(next_state)
    for i in range(num_x - 2, -1, -1):
        next_state = bps[i][next_state]
        path.append(next_state)

    return np.array(path[::-1]), log(vs[num_x - 1].max()) + log_sum
