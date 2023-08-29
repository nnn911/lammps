////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright 2022 OVITO GmbH, Germany
//
//  This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY KIND,
//  either express or implied. See the GPL or the MIT License for the specific language
//  governing rights and limitations.
//
////////////////////////////////////////////////////////////////////////////////////////

#include "fix_dxa_delaunay.h"

namespace LAMMPS_NS {
namespace FIXDXA_NS {

  FacetCirculator::FacetCirculator(const Delaunay &tessellation, SignedVertexHandle s,
                                   SignedVertexHandle t, CellHandle start, LocalFacetHandle f) :
      _tessellation{tessellation},
      _s{s}, _t{t}
  {
    LocalVertexHandle i = _tessellation.localVertexIndex(start, _s);
    LocalVertexHandle j = _tessellation.localVertexIndex(start, _t);
    if (f == next_around_edge(i, j))
      _pos = start;
    else
      _pos = _tessellation.oppositeCell(start, f);
  }

  FacetCirculator &FacetCirculator::operator--()
  {
    _pos = _tessellation.oppositeCell(_pos,
                                      next_around_edge(_tessellation.localVertexIndex(_pos, _t),
                                                       _tessellation.localVertexIndex(_pos, _s)));
    return *this;
  }
  FacetCirculator FacetCirculator::operator--(int)
  {
    FacetCirculator tmp(*this);
    --(*this);
    return tmp;
  }
  FacetCirculator &FacetCirculator::operator++()
  {
    _pos = _tessellation.oppositeCell(_pos,
                                      next_around_edge(_tessellation.localVertexIndex(_pos, _s),
                                                       _tessellation.localVertexIndex(_pos, _t)));
    return *this;
  }
  FacetCirculator FacetCirculator::operator++(int)
  {
    FacetCirculator tmp(*this);
    ++(*this);
    return tmp;
  }
  [[nodiscard]] Facet FacetCirculator::operator*() const
  {
    return {cell(), facet()};
  }
  [[nodiscard]] Facet FacetCirculator::operator->() const
  {
    return {cell(), facet()};
  }

  [[nodiscard]] CellHandle FacetCirculator::cell() const
  {
    return _pos;
  }
  [[nodiscard]] LocalFacetHandle FacetCirculator::facet() const
  {
    return next_around_edge(_tessellation.localVertexIndex(_pos, _s),
                            _tessellation.localVertexIndex(_pos, _t));
  }
  [[nodiscard]] bool FacetCirculator::operator==(const FacetCirculator &other) const
  {
    return _pos == other._pos && _s == other._s && _t == other._t;
  }
  [[nodiscard]] bool FacetCirculator::operator!=(const FacetCirculator &other) const
  {
    return _pos != other._pos || _s != other._s || _t != other._t;
  }

  LocalFacetHandle FacetCirculator::next_around_edge(LocalVertexHandle i, LocalVertexHandle j)
  {
    assert(i >= 0 && i < 4);
    assert(j >= 0 && j < 4);
    static constexpr LocalVertexHandle tab_next_around_edge[4][4] = {
        {5, 2, 3, 1}, {3, 5, 0, 2}, {1, 3, 5, 0}, {2, 0, 1, 5}};
    return tab_next_around_edge[i][j];
  }

  void Delaunay::init()
  {
    GEO::initialize();
    GEO::set_assert_mode(GEO::ASSERT_ABORT);

    _dt = GEO::Delaunay::create(3, "BDEL");
    _dt->set_keeps_infinite(true);
    // required for next around vertex
    _dt->set_stores_cicl(true);
    _dt->set_stores_neighbors(true);
    _dt->set_reorder(true);
    _init = true;
  }
  void Delaunay::generateTessellation(size_t nlocal, size_t nghost, const double *const points,
                                      const tagint *const tags)
  {
    _nlocal = nlocal;
    _nghost = nghost;
    _tags = tags;

    if (!_init) { init(); }
    std::srand(1323);
    GEO::Numeric::random_reset();

    _dt->set_vertices(_nlocal + _nghost, points);

    _setOwned();
    _setRequired();

    _validCells.clear();
    _validCells.resize(numCells(), CellValidity::UNPROCESSED);

    _isValid = true;
  }

  const GEO::Delaunay *const Delaunay::get() const
  {
    return _dt;
  };

  size_t Delaunay::numCells() const
  {
    return _dt->nb_cells();
  };
  size_t Delaunay::numFiniteCells() const
  {
    return _dt->nb_finite_cells();
  };
  size_t Delaunay::numVertices() const
  {
    return _dt->nb_vertices();
  };
  size_t Delaunay::numOwnedCells() const
  {
    size_t count = 0;
    for (size_t cell = 0; cell < numCells(); ++cell) { count += cellIsOwned(cell); }
    return count;
  };
  size_t Delaunay::numOwnedFacets() const
  {
    return _numOwnedFacets;
  };

  // takes a local facet index [0,4) and a local index around that facet [0,3) and returns the
  // local index into the cell
  LocalVertexHandle Delaunay::facetLocalVertex(LocalFacetHandle facet,
                                               LocalVertexHandle localIdx) const
  {
    assert(facet >= 0 && facet < 4);
    assert(localIdx >= 0 && localIdx < 3);
    static constexpr LocalVertexHandle facetMap[4][3] = {
        {1, 3, 2},
        {0, 2, 3},
        {0, 3, 1},
        {0, 1, 2},
    };
    return facetMap[facet][localIdx];
  }
  LocalVertexHandle Delaunay::localVertexIndex(CellHandle cell, SignedVertexHandle vertex) const
  {
    return _dt->index(cell, vertex);
  }
  SignedVertexHandle Delaunay::facetVertex(CellHandle cell, LocalFacetHandle facet,
                                           LocalVertexHandle localIdx) const
  {
    return cellVertex(cell, facetLocalVertex(facet, localIdx));
  }
  SignedVertexHandle Delaunay::cellVertex(CellHandle cell, LocalVertexHandle localIdx) const
  {
    return _dt->cell_vertex(cell, localIdx);
  }

  // Takes a cell and facet index [0,4) and returns the cell on the opposite side of that facet
  SignedCellHandle Delaunay::oppositeCell(CellHandle cell, LocalFacetHandle facet) const
  {
    return _dt->cell_adjacent(cell, facet);
  }

  double Delaunay::getVertexPos(VertexHandle vertexIndex, size_t coord) const
  {
    // x -> coord == 0, y -> coord == 1, z -> coord == 2
    return *(_dt->vertex_ptr(vertexIndex) + coord);
  }
  Vector3d Delaunay::getVertexPos(VertexHandle vertexIndex) const
  {
    return {*(_dt->vertex_ptr(vertexIndex)), *(_dt->vertex_ptr(vertexIndex) + 1),
            *(_dt->vertex_ptr(vertexIndex) + 2)};
  }

  bool Delaunay::cellIsFinite(CellHandle cell) const
  {
    return _dt->cell_is_finite(cell);
  }
  bool Delaunay::cellIsOwned(CellHandle cell) const
  {
    // cell is owned if one or more of its facets are owned
    // cells can be owned by multiple processors
    // facets can only be owned by one processor
    for (LocalFacetHandle facet = 0; facet < 4; ++facet) {
      if (facetIsOwned(cell, facet)) { return true; }
    }
    return false;
  }

  bool Delaunay::cellIsRequired(CellHandle cell) const
  {
    return _requiredCells[cell];
  }
  bool Delaunay::vertexIsRequired(VertexHandle vertex) const
  {
    return _requiredVertices[vertex];
  }

  CellValidity Delaunay::cellIsValid(CellHandle cell) const
  {
    return _validCells[cell];
  }

  void Delaunay::setCellIsValid(CellHandle cell, CellValidity valid)
  {
    assert(cell < _validCells.size());
    _validCells[cell] = valid;
  }

  bool Delaunay::facetIsOwned(CellHandle cell, LocalFacetHandle facet) const
  {
    return _facetOwnership[4 * cell + facet];
  }

  bool Delaunay::isValid() const
  {
    return _isValid;
  }

  Delaunay::AlphaTestResult Delaunay::alphaTest(CellHandle cell, double alpha) const
  {
    std::array<Vector3d, 4> cellVerts;
    for (LocalVertexHandle vert = 0; vert < 4; ++vert) {
      cellVerts[vert] = getVertexPos(cellVertex(cell, vert));
    }
    Sphere<double> s{cellVerts[0], cellVerts[1], cellVerts[2], cellVerts[3]};
    assert(s.valid() || s.unreliable());
    if (s.unreliable()) { return Delaunay::AlphaTestResult::UNRELIABLE; }
    return (s.radius() < alpha) ? Delaunay::AlphaTestResult::INSIDE
                                : Delaunay::AlphaTestResult::OUTSIDE;
  }

  void Delaunay::_setRequired()
  {
    // // a cell is required if that cell is owned or has a neighbor that is owned by the current domain
    // // neighbor here means, that they share an edge
    // bool isRequired;
    // _requiredCells.clear();
    // _requiredCells.resize(numCells(), false);

    // // Todo: this does a lot of duplicate work!
    // for (CellHandle cell = 0; cell < numCells(); ++cell) {
    //   if (!cellIsOwned(cell)) { continue; }
    //   // cells that are owned, are required
    //   _requiredCells[cell] = true;

    //   // all cells that share an edge with an owned cell are also required
    //   for (LocalFacetHandle facet = 0; facet < 4; ++facet) {
    //     for (int edge = 0; edge < 3; ++edge) {

    //       SignedVertexHandle v1 = facetVertex(cell, facet, edge);
    //       SignedVertexHandle v2 = facetVertex(cell, facet, (edge + 1) % 3);
    //       FacetCirculator circulator{*this, v1, v2, cell, facet};
    //       do {
    //         --circulator;
    //         _requiredCells[circulator.cell()] = true;
    //       } while (circulator.cell() != cell);
    //     }
    //   }
    // }

    // a cell is required if that cell is owned or has a neighbor that is owned by the current domain
    bool isRequired;
    _requiredCells.clear();
    _requiredCells.resize(numCells(), false);
    for (size_t cell = 0; cell < numCells(); ++cell) {
      isRequired = cellIsOwned(cell);
      if (!isRequired) {
        for (size_t facet = 0; facet < 4; ++facet) {
          int oppCell = oppositeCell(cell, facet);
          assert(oppCell != -1);
          isRequired = cellIsOwned(oppCell);
          if (isRequired) { break; }
        }
      }
      _requiredCells[cell] = isRequired;
    }

    _requiredVertices.clear();
    _requiredVertices.resize(numVertices(), false);
    std::vector<bool> processed;
    processed.resize(numVertices(), false);
    for (CellHandle cell = 0; cell < numCells(); ++cell) {
      for (LocalVertexHandle lv = 0; lv < 4; ++lv) {
        int gv = cellVertex(cell, lv);
        if (gv == -1) { continue; }
        if (processed[gv]) { continue; }
        processed[gv] = true;
        _requiredVertices[gv] = _requiredCells[cell];
        if (_requiredVertices[gv]) { continue; }

        int incCell = _dt->next_around_vertex(cell, lv);
        while (incCell != -1 && incCell != cell && !_requiredVertices[gv]) {
          _requiredVertices[gv] = _requiredCells[incCell];
          incCell = _dt->next_around_vertex(incCell, _dt->index(incCell, gv));
        }
      }
    }
  }

  void Delaunay::_setOwned()
  {
    bool isOwned;
    _facetOwnership.resize(4 * numCells());
    size_t idx = 0;
    _numOwnedFacets = 0;
    for (CellHandle cell = 0; cell < numCells(); ++cell) {

      bool debugThis = false;
      {
        // static const std::array<tagint, 4> tIDs{{3487, 10582, 10583, 10584}};
        static const std::array<tagint, 4> tIDs{{3484, 3487, 3495, 10582}};
        std::array<tagint, 4> IDs{{_tags[cellVertex(cell, 0)], _tags[cellVertex(cell, 1)],
                                   _tags[cellVertex(cell, 2)], _tags[cellVertex(cell, 3)]}};
        std::sort(IDs.begin(), IDs.end());
        debugThis = tIDs == IDs;
      }

      for (LocalFacetHandle facet = 0; facet < 4; ++facet) {
        isOwned = _facetIsOwned(cell, facet);
        if (debugThis) {
          std::cout << "Cell: " << cell << " Facet: " << facet << " owned: " << isOwned << "\n";
        }
        _facetOwnership[idx++] = isOwned;
        _numOwnedFacets += isOwned;
      }
    }
    assert(std::count(_facetOwnership.begin(), _facetOwnership.end(), true) == _numOwnedFacets);
  };

  bool Delaunay::_facetIsOwned(CellHandle cell, LocalFacetHandle facet) const
  {
    // // a facet is owned if the majority of its vertices is owned
    // // facets can only be owned by one processor
    int ownedCount = 0;
    for (LocalVertexHandle lv = 0; lv < 3; ++lv) {
      int vert = facetVertex(cell, facet, lv);
      ownedCount += vert >= 0 && vert < _nlocal;
    }
    return ownedCount >= 2;
  }

}    // namespace FIXDXA_NS
}    // namespace LAMMPS_NS
