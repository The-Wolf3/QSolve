import scipy as sp
import numpy as np


class Qmat:

    def __init__(self, dims):
        self.dims = dims
        self.space = len(dims)
        self.N = 1
        self.mult = np.ones(self.space, dtype=int)
        for i in range(self.space):
            self.N *= dims[i]+1
            if i < self.space-1:
                self.mult[:i+1] *= dims[i+1]+1

        self.Q = sp.sparse.csc_array((self.N, self.N))

    def check(self, pos):
        if len(pos) == self.space:
            for i in range(self.space):
                if pos[i] > self.dims[i] or pos[i] < 0:
                    return False
            return True
        return False

    def index(self, pos):
        if self.check(pos):
            ans = 0
            for i in range(self.space):
                ans += pos[i]*self.mult[i]
            return ans
        return -1

    def get(self, pos, pos2):
        i1 = self.index(pos)
        i2 = self.index(pos2)
        if i1 != -1 and i2 != -1:
            return self.Q[i1, i2]
        raise IndexError()
        
    def safeSet(self, pos, pos2, value):
        i1 = self.index(pos)
        i2 = self.index(pos2)
        if i1 != -1 and i2 != -1:
            self.Q[i1, i2] = value

    def safeAdd(self, pos, pos2, value):
        i1 = self.index(pos)
        i2 = self.index(pos2)
        if i1 != -1 and i2 != -1:
            self.Q[i1, i2] += value

    def set(self, pos, pos2, value):
        i1 = self.index(pos)
        i2 = self.index(pos2)
        if i1 != -1 and i2 != -1:
            self.Q[i1, i2] = value
        else:
            raise IndexError("Invalid index encoutered in set")
        
    def getQ(self):
        return self.Q
    
    def setOnes(self, pos):
        i = self.index(pos)
        if i != -1:
            self.Q[i] = 1

    def getX(self, b):
        x = sp.sparse.linalg.spsolve(self.Q, b)
        return x


        
