"""Mesh writing, shared. The counterpart to MeshReader, which is read-only.

Before this, writing was scattered and format-specific:

    FunScripts/MeshConvert.write_su2_2d     2D triangles only, from arrays
    FunScripts/MeshConvert.write_vtk_2d     2D triangles only
    RandomMeshMovements.write_su2           rewrites coords into an existing file
    AnisotropicMetric.write_medit_metric    metric sidecar

and there was NO 3D mesh writer at all, so any 3D output had to invent a fourth
one. This module is the single implementation: 2D and 3D, mixed cell types,
named markers.

Element codes are VTK codes, which is what SU2 uses natively (see
Readers/SU2Reader.py: "SU2 element type codes are VTK codes already").

`cells` and each marker accept either

    an (n, k) integer array          -- one cell type, inferred from k and nDim
    [(typeCode, (n, k) array), ...]  -- mixed, given explicitly

so the common single-type case stays a one-liner while mixed grids -- which is
the whole reason this exists -- are expressible without a second API.

Deliberately validating rather than trusting: out-of-range connectivity, wrong
vertex counts for a declared type, and non-finite coordinates are all caught
here. A malformed SU2 mesh otherwise surfaces as a segfault or a nonsense
solution deep inside the solver, a long way from the code that wrote it.
"""

import numpy as np

# VTK / SU2 element codes
LINE, TRI, QUAD, TET, HEX, PRISM, PYRA = 3, 5, 9, 10, 12, 13, 14

NODES_PER_TYPE = {LINE: 2, TRI: 3, QUAD: 4, TET: 4, HEX: 8, PRISM: 6, PYRA: 5}
TYPE_NAME = {LINE: "LINE", TRI: "TRI", QUAD: "QUAD", TET: "TET",
             HEX: "HEX", PRISM: "PRISM", PYRA: "PYRA"}

# (nDim, nodes-per-cell) -> type, for the single-type shorthand
_VOLUME_BY_SHAPE = {(2, 3): TRI, (2, 4): QUAD,
                    (3, 4): TET, (3, 5): PYRA, (3, 6): PRISM, (3, 8): HEX}
_FACE_BY_SHAPE = {(2, 2): LINE, (3, 3): TRI, (3, 4): QUAD}


def _asBlocks(cells, nDim, shapeTable, what):
    """Normalise either form of the cells argument to [(typeCode, array), ...]."""
    if cells is None:
        return []
    if isinstance(cells, (list, tuple)) and len(cells) \
            and isinstance(cells[0], (list, tuple)) and len(cells[0]) == 2 \
            and np.isscalar(cells[0][0]):
        blocks = [(int(t), np.asarray(c, dtype=np.int64)) for t, c in cells]
    else:
        arr = np.asarray(cells, dtype=np.int64)
        if arr.ndim != 2:
            raise ValueError(f"{what}: expected a 2D (n,k) array, got shape {arr.shape}")
        key = (nDim, arr.shape[1])
        if key not in shapeTable:
            raise ValueError(
                f"{what}: cannot infer the element type for {arr.shape[1]} nodes in "
                f"{nDim}D. Pass explicit [(typeCode, conn), ...] instead.")
        blocks = [(shapeTable[key], arr)]
    for t, c in blocks:
        if t not in NODES_PER_TYPE:
            raise ValueError(f"{what}: unknown element type code {t}")
        if c.ndim != 2 or c.shape[1] != NODES_PER_TYPE[t]:
            raise ValueError(
                f"{what}: type {TYPE_NAME[t]} needs {NODES_PER_TYPE[t]} nodes per "
                f"element but the array has shape {c.shape}")
    return blocks


class MeshWriter:
    """Write a mesh. Symmetric with MeshReader(fileName, meshType, nDim)."""

    def __init__(self, fileName, meshType="SU2", nDim=3, zoneName=None):
        self.fileName = str(fileName)
        self.meshType = str(meshType).upper()
        self.nDim = int(nDim)
        self.zoneName = zoneName        # CGNS only; see _writeCGNS on naming
        if self.nDim not in (2, 3):
            raise ValueError(f"nDim must be 2 or 3, got {self.nDim}")

    # -- public ---------------------------------------------------------- #
    def write(self, points, cells, markers=None):
        """points (nPts, nDim>=), cells, markers {name: faces} -> file.

        Extra coordinate columns are ignored, so a 3-column array can be written
        as a 2D mesh without the caller slicing it first.
        """
        P = np.asarray(points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] < self.nDim:
            raise ValueError(f"points must be (nPts, >={self.nDim}), got {P.shape}")
        P = P[:, :self.nDim]
        if not np.isfinite(P).all():
            bad = int((~np.isfinite(P)).any(axis=1).sum())
            raise ValueError(f"{bad} point(s) have non-finite coordinates")

        vol = _asBlocks(cells, self.nDim, _VOLUME_BY_SHAPE, "cells")
        mk = {}
        for name, faces in (markers or {}).items():
            mk[str(name)] = _asBlocks(faces, self.nDim, _FACE_BY_SHAPE,
                                      f"marker '{name}'")

        nPts = P.shape[0]
        for t, c in vol:
            self._checkRange(c, nPts, f"cells ({TYPE_NAME[t]})")
        for name, blocks in mk.items():
            for t, c in blocks:
                self._checkRange(c, nPts, f"marker '{name}' ({TYPE_NAME[t]})")

        if self.meshType == "SU2":
            self._writeSU2(P, vol, mk)
        elif self.meshType in ("VTK", "VTU"):
            self._writeVTK(P, vol)
        elif self.meshType == "CGNS":
            self._writeCGNS(P, vol, mk)
        else:
            raise ValueError(
                f"meshType '{self.meshType}' is not supported for writing "
                f"(have: SU2, CGNS, VTK).")
        return self.fileName

    # -- internals ------------------------------------------------------- #
    @staticmethod
    def _checkRange(conn, nPts, what):
        if conn.size == 0:
            return
        lo, hi = int(conn.min()), int(conn.max())
        if lo < 0 or hi >= nPts:
            raise ValueError(
                f"{what}: connectivity references node index range [{lo}, {hi}] "
                f"but the mesh has {nPts} points. Writing this would produce a "
                f"file the solver reads as corrupt.")

    def _writeSU2(self, P, vol, mk):
        nElem = sum(c.shape[0] for _, c in vol)
        with open(self.fileName, "w") as f:
            f.write(f"NDIME= {self.nDim}\n")
            f.write(f"NELEM= {nElem}\n")
            k = 0
            for t, conn in vol:
                # Inverse of SU2Reader's prism normalization. The layer swap
                # is self-inverse and matches the paired Pointwise SU2 export.
                fileConn = conn[:, [3, 4, 5, 0, 1, 2]] if t == PRISM else conn
                for row in fileConn:
                    f.write(f"{t} " + " ".join(str(int(v)) for v in row) + f" {k}\n")
                    k += 1
            f.write(f"NPOIN= {P.shape[0]}\n")
            # 17 significant digits: a binary64 -> text -> binary64 round trip
            # is exact, so a zero-motion rewrite reproduces coordinates bitwise.
            fmt = " ".join(["%.17g"] * self.nDim)
            for i, p in enumerate(P):
                f.write((fmt % tuple(p)) + f" {i}\n")
            f.write(f"NMARK= {len(mk)}\n")
            for name, blocks in mk.items():
                n = sum(c.shape[0] for _, c in blocks)
                f.write(f"MARKER_TAG= {name}\n")
                f.write(f"MARKER_ELEMS= {n}\n")
                for t, conn in blocks:
                    for row in conn:
                        f.write(f"{t} " + " ".join(str(int(v)) for v in row) + "\n")

    # meshio names for the cell types we emit
    _MESHIO_NAME = {LINE: "line", TRI: "triangle", QUAD: "quad", TET: "tetra",
                    PYRA: "pyramid", PRISM: "wedge", HEX: "hexahedron"}

    # CGNS ElementType_t codes -- NOT the VTK codes used elsewhere in this file.
    # Verified against TestInterp/Meshes/Mixed_16x16x16.cgns, which carries
    # TetElements code 10, PyramidElements 12, tri_FARFIELD 5, quad_FARFIELD 7.
    _CGNS_CODE = {LINE: 3, TRI: 5, QUAD: 7, TET: 10, PYRA: 12, PRISM: 14, HEX: 17}
    _CGNS_SECTION = {TET: "TetElements", PYRA: "PyramidElements",
                     PRISM: "PrismElements", HEX: "HexElements",
                     TRI: "TriElements", QUAD: "QuadElements"}
    _CGNS_FACE_PREFIX = {TRI: "tri", QUAD: "quad", LINE: "line"}

    @staticmethod
    def _cgnsNode(parent, name, label, typ, data=None):
        """One CGNS/HDF5 node: value in a ' data' dataset, metadata in attrs.

        Mirrors exactly what the existing meshes carry, because
        Readers/CGNSReader navigates by these names and reads group[' data'].
        """
        import numpy as _np
        if len(name.encode('ascii')) > 32 or len(label.encode('ascii')) > 32:
            raise ValueError("CGNS node names and labels are limited to 32 ASCII bytes")
        g = parent.create_group(name)
        g.attrs.create('flags', _np.array([1], dtype=_np.int32))
        # CGIO reads these as fixed C strings. Short HDF5 string datatypes
        # make it scan past the attribute into garbage (e.g. ElementRange<U)
        # and abort before discovering any CGNSBase_t nodes.
        g.attrs.create('label', _np.bytes_(label), dtype='S33')
        g.attrs.create('name', _np.bytes_(name), dtype='S33')
        g.attrs.create('type', _np.bytes_(typ), dtype='S3')
        if data is not None:
            g.create_dataset(' data', data=data)
        return g

    def _writeCGNS(self, P, vol, mk):
        """CGNS/HDF5 written directly with h5py.

        meshio is NOT used: given several cell blocks it emits a single
        GridElements section and silently discards the rest. Measured with
        meshio 4.3.13 on a mixed mesh (42269 tets + 1752 pyramids + 2720 marker
        faces) it wrote 169076 connectivity entries -- exactly the tets --
        instead of 177836, losing every pyramid and every boundary face without
        an error, and MeshReader then exits status 0 reading it back. That
        failure is silent, so the format is built here instead.

        Layout, taken from the meshes the pipeline already reads:
          /CGNSLibraryVersion, /Base (CGNSBase_t, ' data' = [cellDim, physDim])
          /Base/<zone> (Zone_t, ' data' = [[nVert],[nCell],[0]])
              ZoneType = 'Unstructured'
              GridCoordinates/CoordinateX|Y|Z (DataArray_t, R8)
              <VolumeSection> (Elements_t, ' data' = [code, 0])
                  ElementConnectivity (flat, ONE-BASED), ElementRange
              <MARKER> (or tri_<MARKER> / quad_<MARKER> for mixed faces)
              ZoneBC/<MARKER> (BC_t) with GridLocation + PointRange
        Element ranges are globally unique and contiguous: volume cells first,
        then the boundary sections, which is what the reader assumes when it
        assigns cell ids from ElementRange.
        """
        try:
            import h5py
        except ImportError:
            raise ValueError("CGNS output needs h5py, which is not installed.")
        for t, _ in vol:
            if t not in self._CGNS_CODE:
                raise ValueError(f"no CGNS code for element type {TYPE_NAME.get(t, t)}")

        def cstr(txt):
            return np.frombuffer((txt + "\x00").encode('ascii'), dtype=np.int8)

        def chars(txt):
            return np.frombuffer(txt.encode('ascii'), dtype=np.int8)

        nVert = P.shape[0]
        nCell = sum(c.shape[0] for _, c in vol)
        # Match the CGNS/HDF5 mapping used by the Pointwise-generated inputs.
        # In particular, Pointwise does not recognize a CGNS base in files
        # created with h5py's default (HDF5 superblock v0) root metadata.
        with h5py.File(self.fileName, 'w', libver='v108') as f:
            f.attrs.create('label', np.bytes_('Root Node of HDF5 File'), dtype='S33')
            f.attrs.create('name', np.bytes_('HDF5 MotherNode'), dtype='S33')
            f.attrs.create('type', np.bytes_('MT'), dtype='S3')
            f.create_dataset(' format', data=cstr('IEEE_LITTLE_32'))
            hdfVersion = (f"HDF5 Version {h5py.version.hdf5_version}\x00"
                          .encode('ascii')[:33].ljust(33, b'\x00'))
            f.create_dataset(' hdf5version',
                             data=np.frombuffer(hdfVersion, dtype=np.int8))
            self._cgnsNode(f, 'CGNSLibraryVersion', 'CGNSLibraryVersion_t', 'R4',
                           np.array([3.31], dtype=np.float32))
            # CGNSBase_t ' data' = [cellDim, physDim]. In 2D both are 2 and only
            # CoordinateX/Y are written, which is what CGNSReader expects when it
            # is constructed with nDim=2.
            base = self._cgnsNode(f, 'Base', 'CGNSBase_t', 'I4',
                                  np.array([self.nDim, self.nDim], dtype=np.int32))
            # ZONE NAME IS NOT FREE. Readers/CGNSReader selects the zone by
            # SUBSTRING: `zone2Read = 'dom' if nDim == 2 else 'blk'`. A 2D file
            # whose zone is called 'blk-1' is rejected with "The CGNS field named
            # dom does not exist!", so the name follows the dimension.
            zoneName = self.zoneName or ('dom-1' if self.nDim == 2 else 'blk-1')
            zone = self._cgnsNode(base, zoneName, 'Zone_t', 'I8',
                                  np.array([[nVert], [nCell], [0]], dtype=np.int64))
            self._cgnsNode(zone, 'ZoneType', 'ZoneType_t', 'C1', chars('Unstructured'))
            gc = self._cgnsNode(zone, 'GridCoordinates', 'GridCoordinates_t', 'MT')
            for k, nm in enumerate(('CoordinateX', 'CoordinateY', 'CoordinateZ')[:self.nDim]):
                self._cgnsNode(gc, nm, 'DataArray_t', 'R8',
                               np.ascontiguousarray(P[:, k], dtype=np.float64))

            first = 1
            sectionCounts = {}
            def section(parent, name, code, conn):
                nonlocal first
                sectionCounts[name] = sectionCounts.get(name, 0) + 1
                if sectionCounts[name] > 1:
                    name = f"{name}_{sectionCounts[name]}"
                last = first + conn.shape[0] - 1
                g = self._cgnsNode(parent, name, 'Elements_t', 'I4',
                                   np.array([code, 0], dtype=np.int32))
                # CGNS connectivity is ONE-BASED
                self._cgnsNode(g, 'ElementConnectivity', 'DataArray_t', 'I8',
                               (np.asarray(conn, dtype=np.int64).ravel() + 1))
                self._cgnsNode(g, 'ElementRange', 'IndexRange_t', 'I8',
                               np.array([first, last], dtype=np.int64))
                first = last + 1
                return first - conn.shape[0] - 1 + 1, last

            for t, conn in vol:
                if len(conn):
                    section(zone, self._CGNS_SECTION[t], self._CGNS_CODE[t], conn)

            bcRanges = {}
            for name, blocks in mk.items():
                # Both CGNS boundary readers look up one section per face type
                # (tri_NAME/quad_NAME), not suffixed repeats such as tri_NAME_2.
                # Repairs can append several blocks of the same type. Combine
                # them here so every writer caller preserves every marker face.
                byType = {}
                for t, conn in blocks:
                    pre = self._CGNS_FACE_PREFIX.get(t)
                    valid = (LINE,) if self.nDim == 2 else (TRI, QUAD)
                    if pre is None or t not in valid:
                        raise ValueError(f"marker '{name}': type {TYPE_NAME.get(t, t)} "
                                         f"is not a valid {self.nDim}D boundary face")
                    if len(conn):
                        byType.setdefault(t, []).append(conn)
                if len(blocks) != len(byType):
                    print(f"[decision] CGNS marker '{name}': combine {len(blocks)} "
                          f"input blocks into {len(byType)} nonempty face-type "
                          "sections. Reason: the boundary readers look up one "
                          "section per type; suffixed repeats would be missed. "
                          "No faces, node IDs or face orientations are changed.")
                for t, parts in byType.items():
                    conn = np.vstack(parts) if len(parts) > 1 else parts[0]
                    pre = self._CGNS_FACE_PREFIX[t]
                    # SU2 uses the section name as its marker tag. Homogeneous
                    # boundaries (including MMG's triangles) need the exact BC
                    # name. Mixed TRI/QUAD boundaries retain separate sections.
                    secName = name if len(byType) == 1 else f"{pre}_{name}"
                    lo, hi = section(zone, secName, self._CGNS_CODE[t], conn)
                    r = bcRanges.setdefault(name, [lo, hi])
                    r[0], r[1] = min(r[0], lo), max(r[1], hi)

            if mk:
                zbc = self._cgnsNode(zone, 'ZoneBC', 'ZoneBC_t', 'MT')
                for name, (lo, hi) in bcRanges.items():
                    # The marker name already identifies the boundary. BCGeneral
                    # is self-contained; FamilySpecified would require a full
                    # Family_t/FamilyName_t tree and is invalid without it.
                    bc = self._cgnsNode(zbc, name, 'BC_t', 'C1', chars('BCGeneral'))
                    self._cgnsNode(bc, 'GridLocation', 'GridLocation_t', 'C1',
                                   chars('EdgeCenter' if self.nDim == 2
                                         else 'FaceCenter'))
                    self._cgnsNode(bc, 'PointRange', 'IndexRange_t', 'I8',
                                   np.array([[lo], [hi]], dtype=np.int64))

    def _writeVTK(self, P, vol):
        """Legacy ASCII VTK, for eyeballing a mesh. Markers are not written."""
        nElem = sum(c.shape[0] for _, c in vol)
        size = sum(c.shape[0] * (c.shape[1] + 1) for _, c in vol)
        with open(self.fileName, "w") as f:
            f.write("# vtk DataFile Version 3.0\nmesh\nASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            f.write(f"POINTS {P.shape[0]} double\n")
            for p in P:
                z = 0.0 if self.nDim == 2 else p[2]
                f.write(f"{p[0]:.16g} {p[1]:.16g} {z:.16g}\n")
            f.write(f"CELLS {nElem} {size}\n")
            for t, conn in vol:
                for row in conn:
                    f.write(f"{conn.shape[1]} " + " ".join(str(int(v)) for v in row) + "\n")
            f.write(f"CELL_TYPES {nElem}\n")
            for t, conn in vol:
                for _ in range(conn.shape[0]):
                    f.write(f"{t}\n")


def writeMesh(fileName, points, cells, markers=None, meshType="SU2", nDim=3):
    """Functional shorthand for the common one-shot case."""
    return MeshWriter(fileName, meshType=meshType, nDim=nDim).write(
        points, cells, markers)
