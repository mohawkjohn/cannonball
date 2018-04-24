import numpy as np

class Log(object):
    """Read and write binary data files"""
    SIZEOF_DOUBLE = 8
    
    def __init__(self, name, num_fields = 7, mode = 'r'):
        self.num_fields = num_fields
        if mode == 'w':
            self.f = open(name, "wb")
        else:
            import os
            self.length = os.stat(name).st_size / (num_fields * self.SIZEOF_DOUBLE)
            self.f = open(name, "rb")
        
    def write(self, fields):
        return self.f.write(bytearray(fields))
        
    def read(self):
        data = np.empty((self.num_fields, self.length))
        for ii in range(0, self.length):
            line = self.f.read(self.num_fields * self.SIZEOF_DOUBLE)
            data[:,ii] = np.frombuffer(line)

        return data

    def close(self):
        return self.f.close()
