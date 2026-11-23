import os.path
import numpy as np
import time
from .ElementsUtilities import ElementsUtilities

class SU2Reader:

    def __init__(self, fileName, Logger, nDim=2):
        if not os.path.isfile(fileName):
            print("ERROR! Mesh file named ", fileName, " does not exist!")
            exit(1)
        else:
            print("Creating SU2 mesh struct from file", fileName)

        self.fileName = fileName
        self.nDim = nDim
        self.Logger = Logger
        self.ElementsUtilities = ElementsUtilities(Logger)
        self.Elements = {}
        self.cell_types = None

    def extractMarkersIndices(self, Markers2Extract):

        import itertools

        extractallMarkers = Markers2Extract is None
        if extractallMarkers:
            print("Extracting all Markers...")
        else:
            print("Extracting indices", Markers2Extract, "...")
        start = time.time()

        markersDict = {}
        allParts = []

        with open(self.fileName, 'rb') as fid2Read:
            while True:
                line = fid2Read.readline()
                if not line:
                    break
                if line.startswith(b'NMARK'):
                    nMarks = int(line.decode('ascii').split('=')[1])
                    for _ in range(nMarks):
                        MarkerName = fid2Read.readline().decode('ascii').split('=')[1].strip()
                        nElements = int(fid2Read.readline().decode('ascii').split('=')[1])
                        block = list(itertools.islice(fid2Read, nElements))   # consume the marker's elements
                        if not (extractallMarkers or MarkerName in Markers2Extract):
                            continue
                        # Unique node ids of this marker. The node columns are all
                        # columns after the leading element-type code (boundary
                        # elements carry no trailing id). One C-level parse for a
                        # single-type marker; ragged (mixed tri+quad surface) falls
                        # back to a per-row scan.
                        try:
                            arr = np.loadtxt(block, dtype=np.int64, ndmin=2)
                            ids = np.unique(arr[:, 1:])
                        except ValueError:
                            s = set()
                            for row in block:
                                s.update(int(t) for t in row.split()[1:])
                            ids = np.array(sorted(s), dtype=np.int64)
                        markersDict[MarkerName] = ids.astype(np.int32)
                        allParts.append(ids)
                    break

        AllPoints = (np.unique(np.concatenate(allParts)).astype(np.int32)
                     if allParts else np.array([], dtype=np.int32))

        end = time.time()
        print("Done! Elapsed time ", end-start, "s")

        return markersDict, AllPoints


    def extractCoordsFromIDs(self, IDs2Extract):

        import itertools

        IDs2Extract = np.asarray(IDs2Extract)
        ExtractedCoords = np.zeros((len(IDs2Extract), 3), dtype=np.float64)

        print("Extracting Mesh Coordinates at indices...")
        start = time.time()

        # Pull the whole NPOIN block, then gather the requested ids by an index
        # map -- O(N + M) instead of the old O(N * M) (a per-point parse with an
        # np.where over IDs2Extract inside the loop). Point row = nDim coords + a
        # trailing point-id column.
        with open(self.fileName, 'rb') as fid2Read:
            while True:
                line = fid2Read.readline()
                if not line:
                    raise RuntimeError(f"No NPOIN section found in {self.fileName}")
                if line.startswith(b'N') and line.decode('ascii').split('=')[0].strip() == "NPOIN":
                    NPoints = int(line.decode('ascii').split('=')[1].split()[0])
                    rows = list(itertools.islice(fid2Read, NPoints))
                    break

        data = np.loadtxt(rows, ndmin=2)
        coords_all = data[:, :self.nDim]
        ids_all = data[:, self.nDim].astype(np.int64)        # last column = point id

        # id -> row map (ids are usually 0..N-1, but build a map to be safe)
        maxid = int(ids_all.max()) if len(ids_all) else -1
        id2row = np.full(maxid + 2, -1, dtype=np.int64)
        id2row[ids_all] = np.arange(len(ids_all))
        req = IDs2Extract.astype(np.int64)
        rows_wanted = np.full(len(req), -1, dtype=np.int64)
        inrange = (req >= 0) & (req <= maxid)
        rows_wanted[inrange] = id2row[req[inrange]]
        found = rows_wanted >= 0                              # missing ids stay (0,0,0)
        ExtractedCoords[found, :self.nDim] = coords_all[rows_wanted[found], :self.nDim]

        end = time.time()
        print("Done! Elapsed time ", end-start, "s")

        if self.nDim == 2:
            ExtractedCoords[:, 2] = 0.0

        return ExtractedCoords

    def extractUnpackedMeshStructure(self, zone2Read=None):
        # Ensure the element sections (with their 'range') have been read.
        if not self.Elements:
            self._read_su2_mesh()

        # SU2 element type codes are VTK codes already (5=TRI, 9=QUAD, 10=TET,
        # 12=HEXA, 13=PRISM, 14=PYRA), so they are used directly as CellTypes.
        perType = {5: [], 9: [], 10: [], 12: [], 13: [], 14: []}
        sections = []   # (start_idx, end_idx, vtkType, connectivity)
        nCells = 0
        for name, data in self.Elements.items():
            vtk = data['type']
            conn = data['connectivity']
            start_idx, end_idx = data['range']
            sections.append((start_idx, end_idx, vtk, conn))
            nCells = max(nCells, end_idx)
            if vtk in perType:
                perType[vtk] = conn

        maxNodes = max((conn.shape[1] for _, _, _, conn in sections), default=0)
        PointIDs = np.zeros((nCells, maxNodes), dtype=np.int32)
        CellTypes = np.zeros((nCells,), dtype=int)
        for start_idx, end_idx, vtk, conn in sections:
            npe = conn.shape[1]
            PointIDs[start_idx:end_idx, :npe] = conn
            CellTypes[start_idx:end_idx] = vtk

        Trias = perType[5]
        Quads = perType[9]
        Tetras = perType[10]
        Hexas = perType[12]
        Wedges = perType[13]
        Pyras = perType[14]

        return Trias, Quads, Hexas, Tetras, Wedges, Pyras, PointIDs, CellTypes
    
    def extractAllCoords(self, zone2Read=None):

        ExtractedCoords = []

        print("Extracting Mesh Coordinates...")
        start = time.time()

        fid2Read = open(self.fileName, 'rb')

        while True:
            # Get next line from file
            line = fid2Read.readline()
            if not line:
                break  # EOF

            if line.startswith(b'N'):
                line_str = line.decode('ascii').strip()
                line1 = line_str.split('=')
                if line1[0] == "NPOIN" :
                    NPoints = int(line1[1].split()[0])

                    # SU2 point row = nDim coordinate columns + a trailing point-id
                    # (3 columns in 2D: x y id; 4 in 3D: x y z id). Parse the whole
                    # block in one C-level call and keep the first nDim coordinate
                    # columns (np.loadtxt splits on arbitrary whitespace, so mixed
                    # space/tab rows are handled and the id column is dropped).
                    coords = np.loadtxt(fid2Read, max_rows=NPoints,
                                        usecols=range(self.nDim), ndmin=2)
                    ExtractedCoords = np.zeros((NPoints, 3), dtype=np.float64)
                    ExtractedCoords[:, :self.nDim] = coords
                    break

        fid2Read.close()
        end = time.time()
        print("Done! Elapsed time ", end-start, "s")

        return ExtractedCoords

    def _read_su2_mesh(self):

        # Cache guard: the SU2 mesh is a full ASCII parse, so skip it if the
        # element sections have already been read (e.g. extractUnpackedMeshStructure
        # followed by extractMeshStructure via buildNodesConnections).
        if self.Elements and self.cell_types is not None:
            return

        import itertools

        self.Logger.info(f"Extracting Mesh Elements...")
        startAll = time.time()

        # Locate the NELEM header, then pull the whole element block at once.
        with open(self.fileName, 'rb') as fid2Read:
            while True:
                line = fid2Read.readline()
                if not line:
                    raise RuntimeError(f"No NELEM section found in {self.fileName}")
                if line.startswith(b'N') and line.decode('ascii').strip().split('=')[0] == "NELEM":
                    NElements = int(line.decode('ascii').split('=')[1].split()[0])
                    rawLines = list(itertools.islice(fid2Read, NElements))   # fast, C-level
                    break

        if len(rawLines) != NElements:
            raise RuntimeError(f"Expected {NElements} elements but read {len(rawLines)} in {self.fileName}")

        NPE = self.ElementsUtilities.NODES_PER_ELEMENT['SU2']
        NAME = self.ElementsUtilities.ELEMENT_NAMES_FROM_TYPES['SU2']

        # Vectorised parse. Fast path: a single element type (or types of equal
        # width) -> the whole block is rectangular and is parsed in ONE C-level
        # np.loadtxt. Fallback: a genuinely mixed-width block (e.g. TRI+QUAD) is
        # ragged, so loadtxt raises; group rows by their leading type code and
        # parse each type's (uniform) rows. Replaces the old per-element double
        # Python loop. Element row = <type> <node0..> [trailing id]; we keep the
        # first nDim-appropriate `npe` node columns, dropping any trailing id.
        try:
            full = np.loadtxt(rawLines, dtype=np.int64, ndmin=2)
            types = full[:, 0]
            uniform = True
        except ValueError:
            types = np.fromiter((int(r.split(None, 1)[0]) for r in rawLines),
                                dtype=np.int64, count=NElements)
            uniform = False

        self.Elements = {}
        cursorRange = 0
        for elem_type in np.unique(types):                  # sorted -> contiguous, type-grouped ranges
            elem_type = int(elem_type)
            npe = NPE[elem_type]
            if uniform:
                conn = full[types == elem_type, 1:1 + npe]
            else:
                idx = np.where(types == elem_type)[0]
                conn = np.loadtxt([rawLines[i] for i in idx], dtype=np.int64,
                                  usecols=range(1, npe + 1), ndmin=2)
            # Internally, prism shape functions and face tables use CGNS
            # PENTA_6 layer orientation. Pointwise's paired SU2 export swaps
            # the two triangular layers, so normalize only at this boundary.
            if elem_type == 13:
                conn = conn[:, [3, 4, 5, 0, 1, 2]]
            conn = np.ascontiguousarray(conn).astype(np.int32)
            ne = conn.shape[0]
            # Contiguous per-type cell-id range (same numbering as
            # SetCellsConnectedToNode / extractUnpackedMeshStructure).
            self.Elements[NAME[str(elem_type)]] = {
                'type': elem_type,
                'connectivity': conn,
                'range': (cursorRange, cursorRange + ne),
            }
            cursorRange += ne

        self.cell_types = types.reshape(-1, 1).astype(int)
        self.Logger.info(f"Done reading Mesh Elements! Elapsed time {str(time.time()-startAll)} s")

    def extractMarkerElements(self, Markers2Extract=None):
        """
        Like extractMarkersIndices, but keeps the boundary-element CONNECTIVITY
        (not just the unique node set). Returns ({MarkerName: connectivity
        (nElems, nPtsPerElem) int array}, [names in file order]). In 2D the
        boundary elements are lines (2 nodes each), which is what the MMG/Medit
        export needs to tag and preserve the boundary markers.
        """
        import itertools

        extractAll = Markers2Extract is None
        markersDict = {}
        names = []
        with open(self.fileName, 'rb') as fid2Read:
            while True:
                line = fid2Read.readline()
                if not line:
                    break
                if line.startswith(b'NMARK'):
                    nMarks = int(line.decode('ascii').split('=')[1])
                    for _ in range(nMarks):
                        MarkerName = fid2Read.readline().decode('ascii').split('=')[1].strip()
                        nElements = int(fid2Read.readline().decode('ascii').split('=')[1])
                        block = list(itertools.islice(fid2Read, nElements))   # consume the elements
                        if not (extractAll or MarkerName in Markers2Extract):
                            continue
                        names.append(MarkerName)
                        # Connectivity is every column after the leading type code.
                        # A marker may hold MIXED face types -- a 3D mixed grid's
                        # boundary is routinely tri+quad -- and those rows have
                        # different column counts (4 vs 5), which np.loadtxt
                        # rejects outright ("number of columns changed"). So the
                        # single-type case keeps the fast one-call parse, and the
                        # mixed case is split by type and keyed the way the CGNS
                        # reader already keys it ('tri_NAME' / 'quad_NAME'), so
                        # both readers present mixed markers identically.
                        codes = np.fromiter((int(l.split(None, 1)[0]) for l in block),
                                            dtype=np.int64, count=len(block))
                        uniqueCodes = np.unique(codes)
                        if uniqueCodes.size <= 1:
                            arr = np.loadtxt(block, dtype=np.int64, ndmin=2)
                            markersDict[MarkerName] = arr[:, 1:].astype(np.int32)
                        else:
                            prefix = {3: 'line', 5: 'tri', 9: 'quad'}
                            for code in uniqueCodes:
                                sel = np.where(codes == code)[0]
                                sub = np.loadtxt([block[i] for i in sel],
                                                 dtype=np.int64, ndmin=2)
                                key = "%s_%s" % (prefix.get(int(code),
                                                            "type%d" % int(code)),
                                                 MarkerName)
                                markersDict[key] = sub[:, 1:].astype(np.int32)
                    break
        return markersDict, names

    def extractMeshStructure(self, zone2Read=None):

        self._read_su2_mesh()

        return self.Elements, self.cell_types
    
    def extractFields(self, zone2Read, fields2Read):
        raise ValueError("No fields to read in a SU2 file!")
        return None
    
    def appendFields(self, zone2Read, fields2Append):
        raise ValueError("No fields to append in a SU2 file!")
        return None
