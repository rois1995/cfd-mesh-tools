import os.path
import numpy as np

class BinaryRestartReader:


    def __init__(self, fileName, nDim=2):
        if not os.path.isfile(fileName):
            print("ERROR! File named ", fileName, " does not exist!")
            exit(1)
        else:
            print("Creating Binary Restart struct from file", fileName)

        self.fileName = fileName
        self.nDim = nDim

    def read(self, what2Extract="ALL"):
        A = np.fromfile(self.fileName, dtype=np.int32, count=5)
        restartFields = []
        for i in range(A[1]):
            strbits = np.fromfile(self.fileName, dtype=np.int8, count=33, offset=20+33*i)
            a_string = ''.join([chr(item) for item in strbits])
            # print(i, " ", a_string)
            restartFields.append(a_string.rstrip('\x00'))

        if what2Extract == "FIELD_NAMES":
            return restartFields

        restartData = np.fromfile(self.fileName, dtype=np.float64, count=A[1]*A[2], offset=20+33*A[1])
        restartData = restartData.reshape((A[1], A[2]), order='F')

        if what2Extract == "COORDS":
            Coords = restartData[0:self.nDim, :]

            Coords = np.transpose(Coords)

            if self.nDim == 2:
                Coords = np.hstack((Coords, np.zeros((len(Coords[:, 1]), 1))))

            return Coords

        return restartData, restartFields
