////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright 2022 OVITO GmbH, Germany
//
//  This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY KIND,
//  either express or implied. See the GPL or the MIT License for the specific language
//  governing rights and limitations.
//
////////////////////////////////////////////////////////////////////////////////////////

#ifndef LMP_FIX_DXA_DELAUNAY_H
#define LMP_FIX_DXA_DELAUNAY_H

#include "fix_dxa_Delaunay_psm.h"
#include "fix_dxa_math.h"
#include "lmptype.h"
#include <limits>

namespace LAMMPS_NS {
namespace FIXDXA_NS {

  enum class CellValidity : char { VALID, INVALID, OTHER, SURFACE, SLIVER, INFINITE, UNPROCESSED };

  class Delaunay {
   public:
    Delaunay() = default;

    void init()
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
    void generateTessellation(size_t nlocal, size_t nghost, const double *const points,
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

    const GEO::Delaunay *const get() const { return _dt; };

    size_t numCells() const { return _dt->nb_cells(); };
    size_t numFiniteCells() const { return _dt->nb_finite_cells(); };
    size_t numVertices() const { return _dt->nb_vertices(); };
    size_t numOwnedCells() const
    {
      size_t count = 0;
      for (size_t cell = 0; cell < numCells(); ++cell) { count += cellIsOwned(cell); }
      return count;
    };
    size_t numOwnedFacets() const { return _numOwnedFacets; };

    // takes a local facet index [0,4) and a local index around that facet [0,3) and returns the
    // local index into the cell
    size_t facetLocalVertex(size_t facet, size_t localIdx) const
    {
      return _facetMap[facet][localIdx];
    }
    int facetVertex(size_t cell, size_t facet, size_t localIdx) const
    {
      return cellVertex(cell, facetLocalVertex(facet, localIdx));
    }
    int cellVertex(size_t cell, size_t localIdx) const { return _dt->cell_vertex(cell, localIdx); }

    // Takes a cell and facet index [0,4) and returns the cell on the opposite side of that facet
    int oppositeCell(size_t cell, size_t facet) const { return _dt->cell_adjacent(cell, facet); }

    double getVertexPos(size_t gVertexIndex, size_t coord) const
    {
      // x -> coord == 0, y -> coord == 1, z -> coord == 2
      return *(_dt->vertex_ptr(gVertexIndex) + coord);
    }
    Vector3d getVertexPos(size_t gVertexIndex) const
    {
      return {*(_dt->vertex_ptr(gVertexIndex)), *(_dt->vertex_ptr(gVertexIndex) + 1),
              *(_dt->vertex_ptr(gVertexIndex) + 2)};
    }

    bool cellIsFinite(size_t cell) const { return _dt->cell_is_finite(cell); }
    bool cellIsOwned(size_t cell) const
    {
      // cell is owned if one or more of its facets are owned
      // cells can be owned by multiple processors
      // facets can only be owned by one processor
      for (size_t facet = 0; facet < 4; ++facet) {
        if (facetIsOwned(cell, facet)) { return true; }
      }
      return false;
    }

    bool cellIsRequired(size_t cell) const { return _requiredCells[cell]; }
    bool vertexIsRequired(size_t vertex) const { return _requiredVertices[vertex]; }

    CellValidity cellIsValid(size_t cell) const
    {
      // if (cellIsFinite(cell)) { return _validCells[cell]; }
      // return CellValidity::INFINITE;
      return _validCells[cell];
    }
    void setCellIsValid(size_t cell, CellValidity valid)
    {
      assert(cell < _validCells.size());
      _validCells[cell] = valid;
    }

    bool facetIsOwned(size_t cell, size_t facet) const { return _facetOwnership[4 * cell + facet]; }

    bool isValid() const { return _isValid; }

    enum class AlphaTestResult { INSIDE, OUTSIDE, UNRELIABLE };
    AlphaTestResult alphaTest(size_t cell, double alpha) const
    {
      std::array<Vector3d, 4> cellVerts;
      for (size_t vert = 0; vert < 4; ++vert) {
        cellVerts[vert] = getVertexPos(cellVertex(cell, vert));
      }
      Sphere<double> s{cellVerts[0], cellVerts[1], cellVerts[2], cellVerts[3]};
      assert(s.valid() || s.unreliable());
      if (s.unreliable()) { return AlphaTestResult::UNRELIABLE; }
      return (s.radius() < alpha) ? AlphaTestResult::INSIDE : AlphaTestResult::OUTSIDE;
    }

   private:
    void _setRequired()
    {
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
      for (size_t cell = 0; cell < numCells(); ++cell) {
        for (size_t lv = 0; lv < 4; ++lv) {
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

    void _setOwned()
    {
      bool isOwned;
      _facetOwnership.resize(4 * numCells());
      size_t idx = 0;
      _numOwnedFacets = 0;
      for (size_t cell = 0; cell < numCells(); ++cell) {
        for (size_t facet = 0; facet < 4; ++facet) {
          isOwned = _facetIsOwned(cell, facet);
          _facetOwnership[idx++] = isOwned;
          _numOwnedFacets += isOwned;
        }
      }
      assert(std::count(_facetOwnership.begin(), _facetOwnership.end(), true) == _numOwnedFacets);
    };

    bool _facetIsOwned(size_t cell, size_t facet) const
    {
      // a facet is owned if its vertex with the lowest tag is owned (local atoms)
      // facets can only be owned by one processor
      tagint minVert = std::numeric_limits<tagint>::max();
      bool minVertOwned = false;
      for (size_t vert = 0; vert < 3; ++vert) {
        int cellVert = cellVertex(cell, facetLocalVertex(facet, vert));
        if (_tags[cellVert] < minVert) {
          minVert = _tags[cellVert];
          minVertOwned = cellVert < _nlocal;
        }
      }
      return minVertOwned;
    }

   private:
    bool _init = false;
    bool _isValid = false;
    size_t _numOwnedFacets = 0;
    size_t _nlocal;
    size_t _nghost;
    GEO::Delaunay *_dt = nullptr;
    const tagint *_tags = nullptr;
    static constexpr size_t _facetMap[4][3] = {
        {1, 3, 2},
        {0, 2, 3},
        {0, 3, 1},
        {0, 1, 2},
    };
    // static constexpr size_t _facetMap[4][3] = {{0, 1, 2}, {1, 3, 2}, {0, 2, 3}, {0, 3, 1}};
    std::vector<bool> _facetOwnership;
    std::vector<CellValidity> _validCells;
    // required cells are either owned or have a neighbor that is owned
    // all required cells need to be valid
    std::vector<bool> _requiredCells;
    // Required atoms belong to a required cell
    std::vector<bool> _requiredVertices;
  };

}    // namespace FIXDXA_NS
}    // namespace LAMMPS_NS

#endif
