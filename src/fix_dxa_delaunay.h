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
#include <limits>

namespace LAMMPS_NS {
namespace FIXDXA_NS {

  class Delaunay {
   public:
    Delaunay() = default;

    void init()
    {
      GEO::initialize();
      GEO::set_assert_mode(GEO::ASSERT_ABORT);

      _dt = GEO::Delaunay::create(3, "BDEL");
      _dt->set_keeps_infinite(true);
      _dt->set_stores_cicl(true);
      _dt->set_reorder(true);
      _init = true;
    }
    void generateTessellation(size_t nlocal, size_t nghost, const double *const points)
    {
      _nlocal = nlocal;
      _nghost = nghost;

      if (!_init) { init(); }
      std::srand(1323);
      GEO::Numeric::random_reset();

      _dt->set_vertices(_nlocal + _nghost, points);

      // Todo consider if this is needed
      bool isOwned;
      _cellOwnership.resize(numCells());
      for (size_t cell = 0; cell < numCells(); ++cell) {
        isOwned = _cellIsOwned(cell);
        _cellOwnership[cell] = isOwned;
        _numOwnedCells += isOwned;
      }

      _facetOwnership.resize(4 * numCells());
      size_t idx = 0;
      for (size_t cell = 0; cell < numCells(); ++cell) {
        for (size_t facet = 0; facet < 4; ++facet) {
          isOwned = _facetIsOwned(cell, facet);
          _facetOwnership[idx++] = isOwned;
          _numOwnedFacets += isOwned;
        }
      }
      _isValid = true;
    }

    const GEO::Delaunay *const get() const { return _dt; };

    size_t numCells() const { return _dt->nb_cells(); };
    size_t numFiniteCells() const { return _dt->nb_finite_cells(); };
    size_t numVertices() const { return _dt->nb_vertices(); };
    size_t numOwnedCells() const { return _numOwnedCells; };
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
    double getVertexPos(size_t gVertexIndex, size_t coord) const
    {
      // x -> coord == 0, y -> coord == 1, z -> coord == 2
      return *(_dt->vertex_ptr(gVertexIndex) + coord);
    }

    bool cellIsFinite(size_t cell) const { return _dt->cell_is_finite(cell); }
    bool cellIsOwned(size_t cell) const { return _cellOwnership[cell]; }
    bool facetIsOwned(size_t cell, size_t facet) const { return _facetOwnership[4 * cell + facet]; }

    bool isValid() const { return _isValid; }

   private:
    bool _facetIsOwned(size_t cell, size_t facet) const
    {
      size_t localVerts = 0;
      for (size_t vert = 0; vert < 3; ++vert) {
        int cellVert = cellVertex(cell, facetLocalVertex(facet, vert));
        localVerts += cellVert < _nlocal;
      }
      return localVerts >= 2;
    }

    bool _cellIsOwned(size_t cell) const
    {
      size_t localVerts = 0;
      for (size_t lv = 0; lv < 4; ++lv) {
        size_t cellVert = _dt->cell_vertex(cell, lv);
        localVerts += cellVert < _nlocal;
      }
      // The current processor is owner of the cell if the majority of vertices are local vertices.
      if (localVerts > 2) { return true; }
      if (localVerts < 2) { return false; }
      // In the case of a tie (2 local, 2 ghost) the vertex with the smallest x-coordinate is used as a tie breaker.
      double minC = std::numeric_limits<double>::max();
      size_t minIdx = 0;
      for (size_t lv = 0; lv < 4; ++lv) {
        size_t cellVert = _dt->cell_vertex(cell, lv);
        double pos = getVertexPos(cellVert, 0);
        if (pos < minC) {
          minC = pos;
          minIdx = cellVert;
        }
      }
      if (minIdx < _nlocal) { return true; }
      return false;
    }

   private:
    bool _init = false;
    bool _isValid = false;
    size_t _numOwnedCells = 0;
    size_t _numOwnedFacets = 0;
    size_t _nlocal;
    size_t _nghost;
    static constexpr size_t _facetMap[4][3] = {{0, 1, 2}, {1, 3, 2}, {0, 2, 3}, {0, 3, 1}};
    std::vector<bool> _cellOwnership;
    std::vector<bool> _facetOwnership;
    GEO::Delaunay *_dt = nullptr;
  };

}    // namespace FIXDXA_NS
}    // namespace LAMMPS_NS

#endif
