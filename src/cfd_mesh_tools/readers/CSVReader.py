import os.path
import numpy as np
try:
    import polars as pl
    print("The Polars package is installed")
    usePolar= True
except ImportError as error:
    print("The Polars package is not installed")
    usePolar= False

# usePolar = False

class CSVReader:


    def __init__(self, fileName, separator = ',', type=int):
        if not os.path.isfile(fileName):
            print("ERROR! File named ", fileName, " does not exist!")
            exit(1)
        else:
            print("Creating CSV struct from file", fileName)

        self.fileName = fileName
        self.separator = separator
        self.type = type

    def read(self):
        if usePolar:
            buffer = pl.read_csv(self.fileName, has_header=True, separator=self.separator, infer_schema_length=10000)
            data = pl.DataFrame.to_numpy(buffer, writable=True)
        else:
            data = np.loadtxt(self.fileName, delimiter=self.separator, skiprows=1, dtype=self.type)

        return data

    def readFieldsByIndex(self, ListOfFieldsIndices):

        if usePolar:
            buffer = pl.read_csv(self.fileName, has_header=True, separator=self.separator, infer_schema_length=10000)
            data = pl.DataFrame.to_numpy(buffer, writable=True)[:, ListOfFieldsIndices]
        else:
            data = np.loadtxt(self.fileName, delimiter=self.separator, skiprows=1, dtype=self.type)[:,ListOfFieldsIndices]


        return data

    def readFieldsByName(self, ListOfFieldsNames):

        dataDict = {}
        with open(self.fileName) as f:
            first_line = f.readline()
            first_line = np.array(first_line.split(','))
            first_line[-1] = first_line[-1].split('\n')[0]

        indices2Read = []
        fieldsPresent = []
        for field in ListOfFieldsNames:
            index = np.where(first_line == field)[0]
            if len(index) == 1:
                indices2Read.append(index[0])
                fieldsPresent.append(field)
                
        indices2Read = np.array(indices2Read, dtype=np.int64)
        fieldsPresent = np.array(fieldsPresent)
        sort_index = np.argsort(indices2Read)
        indices2Read = indices2Read[sort_index]
        fieldsPresent = fieldsPresent[sort_index]

        data = self.readFieldsByIndex(indices2Read)

        for iField, field in enumerate(fieldsPresent):
            dataDict[field] = data[:, iField]

        return dataDict
    
    def readFieldsNames(self):

        with open(self.fileName) as f:
            first_line = f.readline()
            first_line = np.array(first_line.split(','))
            first_line[-1] = first_line[-1].split('\n')[0]

        return first_line
