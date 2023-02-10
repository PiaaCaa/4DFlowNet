import numpy as np


if __name__ == "__main__": 
    A = np.zeros((3, 4, 5 , 6))

    print(np.transpose(A, axes=(0, 1, 2, 3)).shape)

    print(np.transpose(A, axes=(1, 0, 2, 3)).shape)
