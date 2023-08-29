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

  typedef GEO::index_t LocalVertexHandle;
  typedef GEO::index_t VertexHandle;
  typedef GEO::signed_index_t SignedVertexHandle;
  typedef GEO::signed_index_t SignedCellHandle;
  typedef GEO::index_t CellHandle;
  typedef GEO::index_t LocalFacetHandle;

  struct Facet {
    CellHandle cell;
    LocalFacetHandle lf;
  };

  enum class CellValidity : char { VALID, INVALID, OTHER, SURFACE, SLIVER, INFINITE, UNPROCESSED };

  class Delaunay;

  class FacetCirculator {
   public:
    FacetCirculator(const Delaunay &, SignedVertexHandle, SignedVertexHandle, CellHandle,
                    LocalFacetHandle);
    FacetCirculator &operator--();
    FacetCirculator operator--(int);
    FacetCirculator &operator++();
    FacetCirculator operator++(int);
    [[nodiscard]] Facet operator*() const;
    [[nodiscard]] Facet operator->() const;

    [[nodiscard]] CellHandle cell() const;
    [[nodiscard]] LocalFacetHandle facet() const;
    bool operator==(const FacetCirculator &other) const;
    bool operator!=(const FacetCirculator &other) const;
    static LocalFacetHandle next_around_edge(LocalVertexHandle i, LocalVertexHandle j);

   private:
    const Delaunay &_tessellation;
    SignedVertexHandle _s;
    SignedVertexHandle _t;
    CellHandle _pos;
  };

  class Delaunay {
   public:
    Delaunay() = default;

    void init();

    void generateTessellation(size_t nlocal, size_t nghost, const double *const points,
                              const tagint *const tags);

    const GEO::Delaunay *const get() const;

    size_t numCells() const;
    size_t numFiniteCells() const;
    size_t numVertices() const;
    size_t numOwnedCells() const;

    size_t numOwnedFacets() const;

    // takes a local facet index [0,4) and a local index around that facet [0,3) and returns the
    // local index into the cell
    LocalVertexHandle facetLocalVertex(LocalFacetHandle facet, LocalVertexHandle localIdx) const;

    LocalVertexHandle localVertexIndex(CellHandle cell, SignedVertexHandle vertex) const;

    SignedVertexHandle facetVertex(CellHandle cell, LocalFacetHandle facet,
                                   LocalVertexHandle localIdx) const;

    SignedVertexHandle cellVertex(CellHandle cell, LocalVertexHandle localIdx) const;

    // Takes a cell and facet index [0,4) and returns the cell on the opposite side of that facet
    SignedCellHandle oppositeCell(CellHandle cell, LocalFacetHandle facet) const;

    double getVertexPos(VertexHandle vertexIndex, size_t coord) const;

    Vector3d getVertexPos(VertexHandle vertexIndex) const;

    bool cellIsFinite(CellHandle cell) const;
    bool cellIsOwned(CellHandle cell) const;

    bool cellIsRequired(CellHandle cell) const;
    bool vertexIsRequired(VertexHandle vertex) const;

    CellValidity cellIsValid(CellHandle cell) const;

    void setCellIsValid(CellHandle cell, CellValidity valid);

    bool facetIsOwned(CellHandle cell, LocalFacetHandle facet) const;

    bool isValid() const;

    enum class AlphaTestResult { INSIDE, OUTSIDE, UNRELIABLE };

    AlphaTestResult alphaTest(CellHandle cell, double alpha) const;

   private:
    void _setRequired();

    void _setOwned();

    bool _facetIsOwned(CellHandle cell, LocalFacetHandle facet) const;

   private:
    bool _init = false;
    bool _isValid = false;
    size_t _numOwnedFacets = 0;
    size_t _nlocal;
    size_t _nghost;
    GEO::Delaunay *_dt = nullptr;
    const tagint *_tags = nullptr;
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
