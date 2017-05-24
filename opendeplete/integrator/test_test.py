import concurrent.futures
from itertools import repeat
import numpy as np

n = 10000

x = np.random.rand(n, n)

def sum_chunk(i):
    return np.sum(x[:, i])

def test():

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(sum_chunk, range(n))

    result_new = list(result)

if __name__ == '__main__':
    test()