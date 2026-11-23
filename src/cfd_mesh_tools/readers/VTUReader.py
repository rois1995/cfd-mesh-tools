import os
import time
import numpy as np
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    vtk = None
    vtk_to_numpy = None


class VTUReader:
    """
    Reader for VTU flow solutions, providing node coordinates, mesh connectivity
    and point-data fields straight from the solution file (no ParaView, and no
    separate mesh file needed).

    Implements the subset of the CGNSReader/SU2Reader interface that MeshReader
    and CreatePointCloud.py rely on:
      - extractAllCoords()
      - extractMeshStructure()          -> (Elements dict, cell_types)
      - extractUnpackedMeshStructure()  -> per-type arrays + (PointIDs, CellTypes)
      - extractPointDataFromFieldNames()

    Cells use VTK type codes (5=TRI, 9=QUAD, 10=TET, 12=HEXA, 13=PRISM, 14=PYRA,
    3=LINE), which match the SU2 numbering used by ElementsUtilities.
    """

    # VTK cell type -> (section name, nodes per element). Linear elements only.
    _TYPE_INFO = {
        5:  ("TRI", 3),
        9:  ("QUAD", 4),
        10: ("TET", 4),
        14: ("PYRA", 5),
        13: ("PRISM", 6),
        12: ("HEXA", 8),
        3:  ("LINE", 2),
    }
    # Order in which sections are concatenated into the global cell numbering.
    # Both extractMeshStructure (ranges) and extractUnpackedMeshStructure
    # (PointIDs) iterate this same order so cell IDs stay consistent.
    _TYPE_ORDER = [5, 9, 10, 14, 13, 12, 3]

    def __init__(self, fileName, Logger=None, nDim=2):
        if vtk is None:
            raise ImportError("VTUReader requires vtk; install cfd-mesh-tools[vtu] or cfd-mesh-tools[readers]")
        if not os.path.isfile(fileName):
            print("ERROR! Flow file named ", fileName, " does not exist!")
            exit(1)
        else:
            print("Creating VTU flow struct from file", fileName)

        self.fileName = fileName
        self.nDim = nDim
        self.Logger = Logger

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(fileName)
        reader.Update()
        self.grid = reader.GetOutput()
        self.pointData = self.grid.GetPointData()

        self.Elements = {}
        self.cell_types = None
        self._sections = None   # cached [(vtkType, connectivity2d), ...]

    # ------------------------------------------------------------------ #
    # Coordinates and fields
    # ------------------------------------------------------------------ #
    def extractAllCoords(self, zone2Read=None):
        print("Reading coordinates from ", self.fileName, "...")
        start = time.time()

        pts = vtk_to_numpy(self.grid.GetPoints().GetData()).astype(np.float64)
        coords = np.zeros((pts.shape[0], 3), dtype=np.float64)
        coords[:, :pts.shape[1]] = pts

        print("Done! Elapsed time = ", time.time() - start, "s")
        return coords

    def extractPointDataFromFieldNames(self, zone2Read, ListOfFieldsNames):
        print("Reading fields from ", self.fileName, "...")
        start = time.time()

        available = [self.pointData.GetArrayName(i)
                     for i in range(self.pointData.GetNumberOfArrays())]

        fieldDict = {}
        fieldsPresent = []
        for field in ListOfFieldsNames:
            arr_vtk = self.pointData.GetArray(field)
            if arr_vtk is not None:
                arr = vtk_to_numpy(arr_vtk).astype(np.float64)
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                fieldDict[field] = arr
                fieldsPresent.append(field)

        print("Fields 2 read =", fieldsPresent)
        missing = [f for f in ListOfFieldsNames if f not in fieldsPresent]
        if missing:
            print("WARNING: the following fields were not found in", self.fileName, ":", missing)
            print("Available point-data fields are:", available)

        print("Done! Elapsed time = ", time.time() - start, "s")
        return fieldDict

    # ------------------------------------------------------------------ #
    # Connectivity
    # ------------------------------------------------------------------ #
    def _buildSections(self):
        """
        Group the VTU cells by VTK type, returning an ordered list of
        (vtkType, connectivity) where connectivity is a (nCells, nNodes) int32
        array of 0-based node ids. Cached on first call.
        """
        if self._sections is not None:
            return self._sections

        ug = self.grid
        types = vtk_to_numpy(ug.GetCellTypesArray()).astype(np.int64)
        cells = ug.GetCells()
        conn = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64)
        offsets = vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int64)
        starts = offsets[:-1]

        sections = []
        for t in self._TYPE_ORDER:
            mask = types == t
            count = int(mask.sum())
            if count == 0:
                continue
            n = self._TYPE_INFO[t][1]
            idx = starts[mask][:, None] + np.arange(n)[None, :]
            block = conn[idx].astype(np.int32)
            sections.append((t, block))

        # Warn (once) about any cell type we don't know how to handle.
        known = set(self._TYPE_ORDER)
        unknown = sorted(set(np.unique(types).tolist()) - known)
        if unknown:
            print("WARNING: VTU contains unsupported cell types", unknown,
                  "- they are ignored.")

        self._sections = sections
        return sections

    def extractMeshStructure(self, zone2Read=None):
        sections = self._buildSections()

        self.Elements = {}
        nCellsTot = sum(block.shape[0] for _, block in sections)
        AllCell_types = np.zeros((nCellsTot,), dtype=int)

        cursor = 0
        for t, block in sections:
            name = self._TYPE_INFO[t][0]
            count = block.shape[0]
            self.Elements[name] = {
                'type': t,
                'connectivity': block,
                'range': (cursor, cursor + count),
            }
            AllCell_types[cursor:cursor + count] = t
            cursor += count

        self.cell_types = AllCell_types
        return self.Elements, self.cell_types

    def extractUnpackedMeshStructure(self, zone2Read=None):
        sections = self._buildSections()

        perType = {5: [], 9: [], 10: [], 12: [], 13: [], 14: []}
        nCellsTot = sum(block.shape[0] for _, block in sections)
        maxNodes = max((block.shape[1] for _, block in sections), default=0)

        PointIDs = np.zeros((nCellsTot, maxNodes), dtype=np.int32)
        CellTypes = np.zeros((nCellsTot,), dtype=int)

        cursor = 0
        for t, block in sections:
            count, n = block.shape
            PointIDs[cursor:cursor + count, :n] = block
            CellTypes[cursor:cursor + count] = t
            if t in perType:
                perType[t] = block
            cursor += count

        Trias = perType[5]
        Quads = perType[9]
        Tetras = perType[10]
        Hexas = perType[12]
        Wedges = perType[13]
        Pyras = perType[14]

        return Trias, Quads, Hexas, Tetras, Wedges, Pyras, PointIDs, CellTypes
