try:
    import h5py
except ImportError:
    h5py = None
import os.path
import numpy as np
import time
from .ElementsUtilities import ElementsUtilities



class CGNSReader:

    def __init__(self, fileName, Logger, nDim=2):
        if h5py is None:
            raise ImportError("CGNSReader requires h5py; install cfd-mesh-tools[cgns] or cfd-mesh-tools[readers]")
        if not os.path.isfile(fileName):
            print("ERROR! Mesh file named ", fileName, " does not exist!")
            exit(1)
        else:
            print("Creating CGNS mesh struct from file", fileName)

        self.fileName = fileName
        self.nDim = nDim
        self.Logger = Logger
        self.ElementsUtilities = ElementsUtilities(Logger)
        self.Elements = {}
        self.cell_types = None

    def extractMarkersIndices(self, Markers2Extract):

        extractallMarkers=False
        if Markers2Extract == None:
            print("Extracting all Markers...")
            extractallMarkers=True
        else:
            print("Extracting indices", Markers2Extract, "...")
        start = time.time()

        # Handle if Markers2Extract==None, thus extracting all markers!!!

        markersDict = {}
        AllPoints = None

        f = h5py.File(self.fileName,'r')
        listOfThings = np.array(list(f['Base'].keys()))
        zone2Read = 'dom'
        if self.nDim == 3:
            zone2Read = 'blk'

        # Hopefully there is just one!
        field2Read = np.where(np.char.find(np.char.lower(listOfThings), np.char.lower(zone2Read)) > -1)
        if (len(field2Read[0]) == 0):
            print("The CGNS field named " + zone2Read + " does not exist!")
            print("Available fields are:")
            print(listOfThings)
            exit()
        else:
            field2Read = field2Read[0][0]
        field2Read = listOfThings[field2Read]

        AllMarkers = list(f['Base'][field2Read]['ZoneBC'].keys())
        # print(AllMarkers)
        # print(list(f['Base'][field2Read].keys()))
        
        for Marker2Read in AllMarkers:
            print(Marker2Read)
            save = False
            MarkerName2Save = Marker2Read
            if extractallMarkers:
                save = True
                if Marker2Read in list(f['Base'][field2Read].keys()): 
                    print("Reading marker ", Marker2Read)
                    Connectivity = np.array(f['Base'][field2Read][Marker2Read]['ElementConnectivity'][' data'], dtype=np.int32)
                    indexList = np.unique(Connectivity)-1
                else: # For sure it has a tri_ in it
                    print("Reading marker ", "tri_"+Marker2Read, "and", "quad_"+Marker2Read)
                    Connectivity_tri = np.array(f['Base'][field2Read]["tri_"+Marker2Read]['ElementConnectivity'][' data'], dtype=np.int32)
                    Connectivity_quad = np.array(f['Base'][field2Read]["quad_"+Marker2Read]['ElementConnectivity'][' data'], dtype=np.int32)
                    indexList = np.unique(np.hstack([Connectivity_tri, Connectivity_quad]))-1
                
            else:
                print("Marker2Read=",Marker2Read,"Markers2Extract=",Markers2Extract)
                if Marker2Read in Markers2Extract:
                    save = True
                    print("Reading marker ", Marker2Read)
                    print(np.array(f['Base'][field2Read][Marker2Read]['ElementConnectivity'][' data']))
                    Connectivity = np.array(f['Base'][field2Read][Marker2Read]['ElementConnectivity'][' data'], dtype=np.int32)
                    indexList = np.unique(Connectivity)-1
                elif "tri_"+Marker2Read in Markers2Extract:
                    save = False
                    print("Reading marker tri_"+Marker2Read)
                    try:
                        MarkerName2Save = "tri_"+Marker2Read
                        # First try, 'cause it might happen that the marker is now full of tria only!
                        Connectivity = np.array(f['Base'][field2Read]["tri_"+Marker2Read]['ElementConnectivity'][' data'], dtype=np.int32)
                        indexList = np.unique(Connectivity)-1

                        # Save Triangles
                        markersDict[MarkerName2Save] = indexList
                        if isinstance(AllPoints,np.ndarray):
                            AllPoints = np.hstack([AllPoints, indexList])
                        else:
                            AllPoints = indexList

                        MarkerName2Save = "quad_"+Marker2Read
                        print("Reading marker quad_"+Marker2Read)
                        Connectivity = np.array(f['Base'][field2Read]["quad_"+Marker2Read]['ElementConnectivity'][' data'], dtype=np.int32)
                        indexList = np.unique(Connectivity)-1

                        # Save Quads
                        markersDict[MarkerName2Save] = indexList
                        if isinstance(AllPoints,np.ndarray):
                            AllPoints = np.hstack([AllPoints, indexList])
                        else:
                            AllPoints = indexList
                    except:
                        print("Asked for tri_"+Marker2Read+" but not present, trying to read "+Marker2Read+" as a whole")
                        print("Reading marker ", Marker2Read)
                        print(np.array(f['Base'][field2Read][Marker2Read]['ElementConnectivity'][' data']))
                        Connectivity = np.array(f['Base'][field2Read][Marker2Read]['ElementConnectivity'][' data'], dtype=np.int32)
                        indexList = np.unique(Connectivity)-1

                        MarkerName2Save = "tri_"+Marker2Read
                        markersDict[MarkerName2Save] = indexList
                        if isinstance(AllPoints,np.ndarray):
                            AllPoints = np.hstack([AllPoints, indexList])
                        else:
                            AllPoints = indexList

                            # Now save quad_ elements as 0 elements
                        MarkerName2Save = "quad_"+Marker2Read
                        markersDict[MarkerName2Save] = np.array([], dtype=np.int32)
                
            if save:
                markersDict[MarkerName2Save] = indexList
                if isinstance(AllPoints,np.ndarray):
                    AllPoints = np.hstack([AllPoints, indexList])
                else:
                    AllPoints = indexList

        end = time.time()
        print("Done! Elapsed time ", end-start, "s")

        return markersDict, np.unique(AllPoints)


    def extractMarkerElements(self, Markers2Extract=None):
        """
        CGNS analogue of SU2Reader.extractMarkerElements: returns
        ({MarkerName: connectivity (nElems, nPtsPerElem) 0-based int array},
        [names in file order]), keeping the boundary-element connectivity (not
        just the node set as extractMarkersIndices does). 2D boundary elements
        are BAR_2 lines; 3D ones are TRI_3 / QUAD_4 faces. A physical boundary
        AA whose faces are stored split by type appears as separate tri_AA /
        quad_AA entries (same convention as the SU2 mesh files). The
        connectivity in the file is 1-based.
        """
        extractAll = Markers2Extract is None
        markersDict = {}
        names = []

        f = h5py.File(self.fileName, 'r')
        listOfThings = np.array(list(f['Base'].keys()))
        zone2Read = 'dom' if self.nDim == 2 else 'blk'
        field2Read = np.where(np.char.find(np.char.lower(listOfThings), np.char.lower(zone2Read)) > -1)[0]
        if len(field2Read) == 0:
            print("The CGNS field named " + zone2Read + " does not exist!")
            print("Available fields are:")
            print(listOfThings)
            exit()
        field2Read = listOfThings[field2Read[0]]

        # Line boundary elements appear under either type code 2 (this codebase's LINE_2, as in
        # ElementsUtilities) or 3 (ISO-CGNS BAR_2); accept both so neither convention is skipped.
        NPTS_BY_TYPE = {2: 2, 3: 2, 5: 3, 7: 4}   # 2/3 = line (LINE_2 / BAR_2), 5 = TRI_3, 7 = QUAD_4
        AllMarkers = list(f['Base'][field2Read]['ZoneBC'].keys())
        for Marker2Read in AllMarkers:
            if not (extractAll or Marker2Read in Markers2Extract):
                continue
            # A marker is stored either as one Elements_t section with the BC name,
            # or split by surface-element type into tri_NAME / quad_NAME sections.
            for section in (Marker2Read, "tri_" + Marker2Read, "quad_" + Marker2Read):
                grp = f['Base'][field2Read].get(section)
                if grp is None or 'ElementConnectivity' not in grp:
                    continue
                etype = int(np.array(grp[' data'])[0]) if ' data' in grp else -1
                nPts = NPTS_BY_TYPE.get(etype)
                if nPts is None:
                    print(f"WARNING: marker section '{section}' has unsupported element "
                          f"type code {etype}; skipping it.")
                    continue
                conn = np.array(grp['ElementConnectivity'][' data'], dtype=np.int64)
                names.append(section)
                markersDict[section] = conn.reshape(-1, nPts) - 1   # to 0-based

        f.close()
        return markersDict, names

    def extractCoordsFromIDs(self, IDs2Extract):

        ExtractedCoords = np.zeros((len(IDs2Extract), 3), dtype=np.float64)
        ToDo = np.ones((len(IDs2Extract), 3), dtype=np.int32)

        print("Extracting Mesh Coordinates at indices...")
        start = time.time()

        f = h5py.File(self.fileName,'r')
        listOfThings = np.array(list(f['Base'].keys()))
        zone2Read = 'dom'
        if self.nDim == 3:
            zone2Read = 'blk'

        # Hopefully there is just one!
        field2Read = np.where(np.char.find(np.char.lower(listOfThings), np.char.lower(zone2Read)) > -1)
        if (len(field2Read[0]) == 0):
            print("The CGNS field named " + zone2Read + " does not exist!")
            print("Available fields are:")
            print(listOfThings)
            exit()
        else:
            field2Read = field2Read[0][0]
        field2Read = listOfThings[field2Read]

        ExtractedCoords[:, 0] = np.transpose(np.array(f['Base'][field2Read]['GridCoordinates']['CoordinateX'][' data'])[IDs2Extract])
        ExtractedCoords[:, 1] = np.transpose(np.array(f['Base'][field2Read]['GridCoordinates']['CoordinateY'][' data'])[IDs2Extract])
        if self.nDim == 3:
            ExtractedCoords[:, 2] = np.transpose(np.array(f['Base'][field2Read]['GridCoordinates']['CoordinateZ'][' data'])[IDs2Extract])
            
        end = time.time()
        print("Done! Elapsed time ", end-start, "s")

        return ExtractedCoords

    def extractAllCoords(self, zone2Read = None):

        print("Reading coordinates from ", self.fileName, "...")
        start = time.time()

        f = h5py.File(self.fileName,'r')
        listOfThings = np.array(list(f['Base'].keys()))
        if zone2Read == None:
            zone2Read = 'dom'
            if self.nDim == 3:
                zone2Read = 'blk'

        # Hopefully there is just one!
        field2Read = np.where(np.char.find(np.char.lower(listOfThings), np.char.lower(zone2Read)) > -1)
        if (len(field2Read[0]) == 0):
            print("The CGNS field named " + zone2Read + " does not exist!")
            print("Available fields are:")
            print(listOfThings)
            exit()
        else:
            field2Read = field2Read[0][0]
        field2Read = listOfThings[field2Read]

        x = np.array(f['Base'][field2Read]['GridCoordinates']['CoordinateX'][' data'])
        y = np.array(f['Base'][field2Read]['GridCoordinates']['CoordinateY'][' data'])
        z = np.zeros((len(x),))
        if self.nDim == 3:
            z = np.array(f['Base'][field2Read]['GridCoordinates']['CoordinateZ'][' data'])

        ExtractedCoords = np.transpose(np.stack((x, y, z)))

        print("Done! Elapsed time = ", time.time()-start, "s")

        return ExtractedCoords

    def extractUnpackedMeshStructure(self, zone2Read):

        # Parse the element sections once into self.Elements (guarded inside
        # _read_cgns_h5py), then build the unpacked PointIDs/CellTypes from it --
        # no second pass over the file.
        self._read_cgns_h5py(zoneName=zone2Read)

        print("Reading point indices...")
        start = time.time()

        # CGNS element type code -> output VTK type.
        # (CGNS: TRI_3=5, QUAD_4=7, TETRA_4=10, PYRA_5=12, PENTA_6=14, HEXA_8=17)
        vtkOfCGNS = {5: 5, 7: 9, 10: 10, 12: 14, 14: 13, 17: 12}
        perType = {5: [], 9: [], 10: [], 12: [], 13: [], 14: []}   # keyed by VTK code

        sections = []   # (start_idx, end_idx, vtkType, connectivity)
        for name, data in self.Elements.items():
            ctype = data['type']
            if ctype not in vtkOfCGNS:
                continue
            vtk = vtkOfCGNS[ctype]
            conn = data['connectivity']
            start_idx, end_idx = data['range']
            sections.append((start_idx, end_idx, vtk, conn))
            if vtk in perType:
                perType[vtk] = conn

        # Place every section at its global cell-id range so the cell numbering
        # matches buildCellsOfPoints (which assigns ids from ElementRange too).
        # Works uniformly for 2D, 3D volume, and 3D surface (tri/quad) meshes.
        nCells = len(self.cell_types)
        maxNodes = max((conn.shape[1] for _, _, _, conn in sections), default=0)

        PointIDs = np.zeros((nCells, maxNodes), dtype=np.int32)
        CellTypes = np.zeros((nCells,), dtype=int)
        for start_idx, end_idx, vtk, conn in sections:
            npe = conn.shape[1]
            PointIDs[start_idx:end_idx, :npe] = conn
            CellTypes[start_idx:end_idx] = vtk

        Trias = perType[5]; Quads = perType[9]; Tetras = perType[10]
        Hexas = perType[12]; Wedges = perType[13]; Pyras = perType[14]

        print("Done! Elapsed time = ", time.time()-start, "s")

        return Trias, Quads, Hexas, Tetras, Wedges, Pyras, PointIDs, CellTypes

    def extractPointDataFromFieldNames(self, zone2Read, ListOfFieldsNames):

        print("Reading fields from ", self.fileName, "...")
        start = time.time()

        f = h5py.File(self.fileName,'r')

        fieldDict = {}

        # Resolve the zone (exact, else substring) so 'Zone 1' and 'Zone' both work.
        baseKeys = np.array(list(f['Base'].keys()))
        if zone2Read in f['Base']:
            zoneName = zone2Read
        else:
            hit = np.where(np.char.find(np.char.lower(baseKeys), np.char.lower(str(zone2Read))) > -1)[0]
            if len(hit) == 0:
                print("The CGNS zone named " + str(zone2Read) + " does not exist! Available:", list(baseKeys))
                exit(1)
            zoneName = baseKeys[hit[0]]
        zone = f['Base'][zoneName]

        # Solution fields live under 'PointData' (volume exports) or 'Fields'
        # (SU2 surface_flow exports). Use whichever is present.
        if 'PointData' in zone:
            fieldGroup = zone['PointData']
        elif 'Fields' in zone:
            fieldGroup = zone['Fields']
        else:
            print("ERROR! Neither 'PointData' nor 'Fields' found in zone", zoneName)
            exit(1)

        availableFields = np.array(list(fieldGroup.keys()))

        fieldsPresent = []
        for field in ListOfFieldsNames:
            index = np.where(availableFields == field)[0]
            if len(index) == 1:
                fieldsPresent.append(field)

        print("Fields 2 read =", fieldsPresent)
        for field in fieldsPresent:
            fieldDict[field] = np.array(fieldGroup[field][' data'])


        print("Done! Elapsed time = ", time.time()-start, "s")

        return fieldDict
    

    def _read_cgns_h5py(self, zoneName = None):
        """Read CGNS using h5py - NumPy 2.0 compatible"""
        # Cache guard: skip re-reading the element sections if they have already
        # been parsed (e.g. extractUnpackedMeshStructure followed by
        # extractMeshStructure via buildNodesConnections).
        if self.Elements and self.cell_types is not None:
            return

        try:
            import h5py
        except ImportError:
            self.Logger.error("h5py not installed. Run: pip install h5py")
            raise


        with h5py.File(self.fileName, 'r') as f:
            # Navigate CGNS tree structure
            base_name = 'Base'
            base = f[base_name]

            listOfThings = np.array(list(base.keys()))
            zone2Read = 'dom'
            if self.nDim == 3:
                zone2Read = 'blk'

            if zoneName is not None:
                zone2Read = zoneName

            # Hopefully there is just one!
            field2Read = np.where(np.char.find(np.char.lower(listOfThings), np.char.lower(zone2Read)) > -1)
            if (len(field2Read[0]) == 0):
                print("The CGNS field named " + zone2Read + " does not exist!")
                print("Available fields are:")
                print(listOfThings)
                exit()
            else:
                field2Read = field2Read[0][0]
            zone2Read = listOfThings[field2Read]

            zone = base[zone2Read]

            # Read grid coordinates
            if 'GridCoordinates' not in zone:
                raise ValueError("GridCoordinates not found in zone")

            AllCell_types = np.zeros((np.array(zone[' data'])[1][0], 1), dtype=int)
            
            # Read element sections. Identify them robustly as any group holding
            # an 'ElementConnectivity' array (covers 'Elem_Triangles', bare
            # 'Triangles'/'Quadrilaterals' from surface files, volume sections, etc.).
            sections = []
            for key in zone.keys():

                if hasattr(zone[key], 'keys') and 'ElementConnectivity' in zone[key]:
                    try:
                        elem_section = zone[key]

                        # Read element type
                        if ' data' in elem_section:
                            elem_type = int(elem_section[' data'][0])
                        else:
                            continue

                        # Read connectivity
                        if 'ElementConnectivity' not in elem_section:
                            continue

                        conn_data = elem_section['ElementConnectivity']
                        if ' data' in conn_data:
                            connectivity = np.array(conn_data[' data'])
                        else:
                            connectivity = np.array(conn_data[()])
                        
                        # Read element range
                        if 'ElementRange' in elem_section:
                            range_data = elem_section['ElementRange']
                            if ' data' in range_data:
                                elem_range = range_data[' data']
                            else:
                                elem_range = range_data[()]
                            start_idx = int(elem_range[0]) - 1
                            end_idx = int(elem_range[1])
                        else:
                            start_idx = 0
                            end_idx = len(connectivity)

                        # Zone_t counts the volume/area cells only. Boundary
                        # sections follow that range and are read separately by
                        # extractMarkerElements; including them here corrupts
                        # metric neighbours and cell-of-node indices.
                        if start_idx >= len(AllCell_types):
                            continue
                        AllCell_types[start_idx:end_idx] = elem_type
                        nPointsPerElement = self.ElementsUtilities.NODES_PER_ELEMENT['CGNS'][elem_type]
                        connectivity = np.reshape(connectivity-1, (int(len(connectivity)/nPointsPerElement), nPointsPerElement))

                        section_name = self.ElementsUtilities.ELEMENT_NAMES_FROM_TYPES['CGNS'][str(elem_type)]
                        sections.append((start_idx, str(key), section_name, {
                            'type': elem_type,
                            'connectivity': connectivity,  # Convert to 0-based
                            'range': (start_idx, end_idx)
                        }))
                    except Exception as e:
                        self.Logger.error(f"  Warning: Could not read section '{key}': {e}")
                        continue

            # HDF5 group iteration order is not CGNS cell order. ElementRange
            # is authoritative; preserving it also keeps provenance cell ids
            # stable across a read/write round trip.
            for _start, key, section_name, data in sorted(sections):
                store_name = section_name
                if store_name in self.Elements:
                    store_name = section_name + '@' + key
                self.Elements[store_name] = data
                self.Logger.info(
                    f"  Section '{section_name}': type={data['type']}, "
                    f"elements={data['range'][1] - data['range'][0]}")
            
            self.cell_types = AllCell_types

    def extractMeshStructure(self, zone2Read = None):

        self._read_cgns_h5py(zoneName = zone2Read)

        return self.Elements, self.cell_types
    
    def extractFields(self, zone2Read, fields2Read):

        print("Reading fields from ", self.fileName, "...")
        start = time.time()

        f = h5py.File(self.fileName,'r')
        listOfThings = np.array(list(f['Base'].keys()))
        
        # Hopefully there is just one!
        zone2Read = np.where(np.char.find(np.char.lower(listOfThings), np.char.lower(zone2Read)) > -1)
        if (len(zone2Read[0]) == 0):
            print("The CGNS field named " + zone2Read + " does not exist!")
            print("Available fields are:")
            print(listOfThings)
            exit(1)
        else:
            zone2Read = zone2Read[0][0]
        zone2Read = listOfThings[zone2Read]

        FieldsDict = {}
        for field2Read in fields2Read:
            Field = np.array(f['Base'][zone2Read]['Fields'][field2Read][' data'])
            FieldsDict[field2Read] = Field

        print("Done! Elapsed time = ", time.time()-start, "s")

        f.close()
        
        return FieldsDict
    
    def appendFields(self, zone2Read, fields2Append):

        print("Append fields to ", self.fileName, "...")
        start = time.time()

        file = h5py.File(self.fileName,'r+')
        listOfThings = np.array(list(file['Base'].keys()))
        
        # Hopefully there is just one!
        zone2Read = np.where(np.char.find(np.char.lower(listOfThings), np.char.lower(zone2Read)) > -1)
        if (len(zone2Read[0]) == 0):
            print("The CGNS field named " + zone2Read + " does not exist!")
            print("Available fields are:")
            print(listOfThings)
            exit(1)
        else:
            zone2Read = zone2Read[0][0]
        zone2Read = listOfThings[zone2Read]

        DimsNames = ['x', 'y', 'z']
        for fieldName2Append in fields2Append:
            field2Append = fields2Append[fieldName2Append]

            for iDim in range(3):
                actualFieldName = fieldName2Append+"_"+DimsNames[iDim]

                file.copy(
                       f"/Base/{zone2Read}/Fields/Density",
                       f"/Base/{zone2Read}/Fields/{actualFieldName}"
                      )
                
                file["Base"][zone2Read]["Fields"][actualFieldName][" data"][...] = field2Append[:, iDim]

                # Update the name attribute (CGNS requires 33-character null-terminated string)
                file["Base"][zone2Read]["Fields"][actualFieldName].attrs["name"] = np.bytes_(
                    bytes(actualFieldName, 'ascii').ljust(33, b"\x00")
                    )
            
            # Field = np.array(f['Base'][zone2Read]['Fields'][field2Read][' data'])
            # FieldsDict[field2Read] = Field

        print("Done! Elapsed time = ", time.time()-start, "s")

        file.close()
