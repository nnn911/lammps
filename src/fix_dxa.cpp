
#include "fix_dxa.h"
#include "atom.h"
#include "error.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "utils.h"
#include <numeric>

namespace LAMMPS_NS {
namespace FIXDXA_NS {
  [[noreturn]] static inline void unreachable(LAMMPS *lmp)
  {
#ifndef NDEBUG
    lmp->error->all(FLERR, "Reached unreachable code!\n");
#endif
#if defined(__has_builtin) && __has_builtin(__builtin_unreachable)
    __builtin_unreachable();
#elif __clang__ || __GNUC__
    __builtin_unreachable();
#elif _MSC_VER
    __assume(false);
#endif
  }

  FixDXA::FixDXA(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
  {
    if (narg < 5) error->all(FLERR, "Not enough parameters specified for fix DXA");
    int iarg = 3;

    this->nevery = utils::inumeric(FLERR, arg[iarg++], true, lmp);
    if (this->nevery < 1) error->all(FLERR, "Invalid timestep parameter for fix DXA");

    std::string inputStructure = utils::lowercase(arg[iarg++]);
    if (inputStructure == "bcc")
      _inputStructure = BCC;
    else if (inputStructure == "cubicdia")
      _inputStructure = CUBIC_DIA;
    else if (inputStructure == "fcc")
      _inputStructure = FCC;
    else if (inputStructure == "hcp")
      _inputStructure = HCP;
    else if (inputStructure == "hexdia")
      _inputStructure = HEX_DIA;
    else
      error->all(FLERR, "Invalid input structure parameter for fix DXA");

    static bool structuresInitialized = false;
    if (!structuresInitialized) {
      initializeStructures();
      structuresInitialized = true;
    }
  }

  template <typename iterator> void bitmapSort(iterator begin, iterator end, size_t size)
  {
    assert(size <= 32);
    unsigned int bitmap = 0;
    for (iterator pin = begin; pin != end; ++pin) { bitmap |= 1 << (*pin); }
    iterator pout = begin;
    for (int i = size - 1; i >= 0; i--)
      if (bitmap & (1 << i)) *pout++ = i;
    assert(pout == end);
  }

  std::array<CrystalStructure<FixDXA::_maxNeighCount>, FixDXA::MAXSTRUCTURECOUNT>
      FixDXA::_crystalStructures;

  void FixDXA::initializeStructures()
  {
    {
      _crystalStructures[OTHER].numNeighbors = 0;
    }
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Face centered cubic
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    {
      CrystalStructure<_maxNeighCount> &crystalStructure = _crystalStructures[FCC];
      crystalStructure.latticeVectors = {
          Vector3d(0.5, 0.5, 0.0),   Vector3d(0.0, 0.5, 0.5),   Vector3d(0.5, 0.0, 0.5),
          Vector3d(-0.5, -0.5, 0.0), Vector3d(0.0, -0.5, -0.5), Vector3d(-0.5, 0.0, -0.5),
          Vector3d(-0.5, 0.5, 0.0),  Vector3d(0.0, -0.5, 0.5),  Vector3d(-0.5, 0.0, 0.5),
          Vector3d(0.5, -0.5, 0.0),  Vector3d(0.0, 0.5, -0.5),  Vector3d(0.5, 0.0, -0.5)};

      crystalStructure.numNeighbors = 12;
      for (int n1 = 0; n1 < crystalStructure.numNeighbors; ++n1) {
        crystalStructure.neighborArray.setNeighborBond(n1, n1, false);
        for (int n2 = n1 + 1; n2 < crystalStructure.numNeighbors; ++n2) {
          crystalStructure.neighborArray.setNeighborBond(
              n1, n2,
              ((crystalStructure.latticeVectors[n1] - crystalStructure.latticeVectors[n2])
                   .length() < (sqrt(0.5f) + 1.0) * 0.5));
        }
        crystalStructure.cnaSignatures[n1] = 0;
      }
      crystalStructure.primitiveCell.column(0) = Vector3d(0.5, 0.5, 0.0);
      crystalStructure.primitiveCell.column(1) = Vector3d(0.0, 0.5, 0.5);
      crystalStructure.primitiveCell.column(2) = Vector3d(0.5, 0.0, 0.5);
    }
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Hexagonal closed packing
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    {
      CrystalStructure<_maxNeighCount> &crystalStructure = _crystalStructures[HCP];

      crystalStructure.latticeVectors = {
          Vector3d(sqrt(2.0) / 4.0, -sqrt(6.0) / 4.0, 0.0),
          Vector3d(-sqrt(2.0) / 2.0, 0.0, 0.0),
          Vector3d(-sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
          Vector3d(sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
          Vector3d(0.0, -sqrt(6.0) / 6.0, -sqrt(3.0) / 3.0),
          Vector3d(-sqrt(2.0) / 4.0, sqrt(6.0) / 4.0, 0.0),
          Vector3d(sqrt(2.0) / 4.0, sqrt(6.0) / 4.0, 0.0),
          Vector3d(sqrt(2.0) / 2.0, 0.0, 0.0),
          Vector3d(-sqrt(2.0) / 4.0, -sqrt(6.0) / 4.0, 0.0),
          Vector3d(0.0, -sqrt(6.0) / 6.0, sqrt(3.0) / 3.0),
          Vector3d(sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),
          Vector3d(-sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),
          Vector3d(0.0, sqrt(6.0) / 6.0, sqrt(3.0) / 3.0),
          Vector3d(-sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
          Vector3d(sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),
          Vector3d(0.0, sqrt(6.0) / 6.0, -sqrt(3.0) / 3.0),
          Vector3d(sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
          Vector3d(-sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, sqrt(3.0) / 3.0)};

      crystalStructure.numNeighbors = 12;
      for (int n1 = 0; n1 < crystalStructure.numNeighbors; n1++) {
        crystalStructure.neighborArray.setNeighborBond(n1, n1, false);
        for (int n2 = n1 + 1; n2 < crystalStructure.numNeighbors; n2++) {
          crystalStructure.neighborArray.setNeighborBond(
              n1, n2,
              ((crystalStructure.latticeVectors[n1] - crystalStructure.latticeVectors[n2])
                   .length() < (sqrt(0.5) + 1.0) * 0.5));
        }
        crystalStructure.cnaSignatures[n1] = (crystalStructure.latticeVectors[n1].z() == 0) ? 1 : 0;
      }
      crystalStructure.primitiveCell.column(0) = Vector3d(sqrt(0.5) / 2, -sqrt(6.0) / 4, 0.0);
      crystalStructure.primitiveCell.column(1) = Vector3d(sqrt(0.5) / 2, sqrt(6.0) / 4, 0.0);
      crystalStructure.primitiveCell.column(2) = Vector3d(0.0, 0.0, sqrt(8.0 / 6.0));
    }
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Body centered cubic
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    {
      CrystalStructure<_maxNeighCount> &crystalStructure = _crystalStructures[BCC];
      crystalStructure.latticeVectors = {
          Vector3d(0.5, 0.5, 0.5),    Vector3d(-0.5, 0.5, 0.5),  Vector3d(0.5, 0.5, -0.5),
          Vector3d(-0.5, -0.5, 0.5),  Vector3d(0.5, -0.5, 0.5),  Vector3d(-0.5, 0.5, -0.5),
          Vector3d(-0.5, -0.5, -0.5), Vector3d(0.5, -0.5, -0.5), Vector3d(1.0, 0.0, 0.0),
          Vector3d(-1.0, 0.0, 0.0),   Vector3d(0.0, 1.0, 0.0),   Vector3d(0.0, -1.0, 0.0),
          Vector3d(0.0, 0.0, 1.0),    Vector3d(0.0, 0.0, -1.0)};

      crystalStructure.numNeighbors = 14;
      for (int n1 = 0; n1 < crystalStructure.numNeighbors; ++n1) {
        crystalStructure.neighborArray.setNeighborBond(n1, n1, false);
        for (int n2 = n1 + 1; n2 < crystalStructure.numNeighbors; ++n2) {
          crystalStructure.neighborArray.setNeighborBond(
              n1, n2,
              ((crystalStructure.latticeVectors[n1] - crystalStructure.latticeVectors[n2])
                   .length() < (1.0 + sqrt(2.0)) * 0.5));
        }
        crystalStructure.cnaSignatures[n1] = (n1 < 8) ? 0 : 1;
      }
      crystalStructure.primitiveCell.column(0) = Vector3d(1.0, 0.0, 0.0);
      crystalStructure.primitiveCell.column(1) = Vector3d(0.0, 1.0, 0.0);
      crystalStructure.primitiveCell.column(2) = Vector3d(0.5, 0.5, 0.5);
    }
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Cubic diamond
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    {
      CrystalStructure<_maxNeighCount> &crystalStructure = _crystalStructures[CUBIC_DIA];
      crystalStructure.latticeVectors = {
          Vector3d(0.25, 0.25, 0.25),   Vector3d(0.25, -0.25, -0.25), Vector3d(-0.25, -0.25, 0.25),
          Vector3d(-0.25, 0.25, -0.25), Vector3d(0, -0.5, 0.5),       Vector3d(0.5, 0.5, 0),
          Vector3d(-0.5, 0, 0.5),       Vector3d(-0.5, 0.5, 0),       Vector3d(0, 0.5, 0.5),
          Vector3d(0.5, -0.5, 0),       Vector3d(0.5, 0, 0.5),        Vector3d(0.5, 0, -0.5),
          Vector3d(-0.5, -0.5, 0),      Vector3d(0, -0.5, -0.5),      Vector3d(0, 0.5, -0.5),
          Vector3d(-0.5, 0, -0.5),      Vector3d(0.25, -0.25, 0.25),  Vector3d(0.25, 0.25, -0.25),
          Vector3d(-0.25, 0.25, 0.25),  Vector3d(-0.25, -0.25, -0.25)};

      crystalStructure.numNeighbors = 16;

      for (int n1 = 0; n1 < 16; ++n1) {
        crystalStructure.neighborArray.setNeighborBond(n1, n1, false);
        double cutoff = (n1 < 4) ? (sqrt(3.0) * 0.25 + sqrt(0.5)) / 2 : (1.0 + sqrt(0.5)) / 2;
        double cutoffSquared = cutoff * cutoff;
        for (int n2 = n1 + 1; n2 < 4; ++n2) {
          crystalStructure.neighborArray.setNeighborBond(n1, n2, false);
        }
        for (int n2 = std::max(n1 + 1, 4); n2 < crystalStructure.numNeighbors; ++n2) {
          crystalStructure.neighborArray.setNeighborBond(
              n1, n2,
              ((crystalStructure.latticeVectors[n1] - crystalStructure.latticeVectors[n2])
                   .lengthSquared() < cutoffSquared));
        }
        crystalStructure.cnaSignatures[n1] = (n1 < 4) ? 0 : 1;
      }
      crystalStructure.primitiveCell.column(0) = Vector3d(0.5, 0.5, 0.0);
      crystalStructure.primitiveCell.column(1) = Vector3d(0.0, 0.5, 0.5);
      crystalStructure.primitiveCell.column(2) = Vector3d(0.5, 0.0, 0.5);
    }
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Hexagonal diamond
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    {
      CrystalStructure<_maxNeighCount> &crystalStructure = _crystalStructures[HEX_DIA];
      crystalStructure.latticeVectors = {
          Vector3d(-sqrt(2.0) / 4, sqrt(3.0 / 2.0) / 6, -sqrt(3.0) / 12),
          Vector3d(0, -sqrt(3.0 / 2.0) / 3, -sqrt(3.0) / 12),
          Vector3d(sqrt(2.0) / 4, sqrt(3.0 / 2.0) / 6, -sqrt(3.0) / 12),
          Vector3d(0, 0, sqrt(3.0) / 4),

          Vector3d(sqrt(2.0) / 4.0, -sqrt(6.0) / 4.0, 0.0),
          Vector3d(-sqrt(2.0) / 2.0, 0.0, 0.0),
          Vector3d(-sqrt(2.0) / 4.0, sqrt(6.0) / 4.0, 0.0),
          Vector3d(sqrt(2.0) / 4.0, sqrt(6.0) / 4.0, 0.0),
          Vector3d(sqrt(2.0) / 2.0, 0.0, 0.0),
          Vector3d(-sqrt(2.0) / 4.0, -sqrt(6.0) / 4.0, 0.0),
          Vector3d(-sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
          Vector3d(sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
          Vector3d(0.0, -sqrt(6.0) / 6.0, -sqrt(3.0) / 3.0),
          Vector3d(0.0, -sqrt(6.0) / 6.0, sqrt(3.0) / 3.0),
          Vector3d(sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),
          Vector3d(-sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),

          Vector3d(-sqrt(2.0) / 4, sqrt(3.0 / 2.0) / 6, sqrt(3.0) / 12),
          Vector3d(0, -sqrt(3.0 / 2.0) / 3, sqrt(3.0) / 12),
          Vector3d(sqrt(2.0) / 4, sqrt(3.0 / 2.0) / 6, sqrt(3.0) / 12),
          Vector3d(0, 0, -sqrt(3.0) / 4),

          Vector3d(-sqrt(2.0) / 4, -sqrt(3.0 / 2.0) / 6, -sqrt(3.0) / 12),
          Vector3d(0, sqrt(3.0 / 2.0) / 3, -sqrt(3.0) / 12),
          Vector3d(sqrt(2.0) / 4, -sqrt(3.0 / 2.0) / 6, -sqrt(3.0) / 12),

          Vector3d(-sqrt(2.0) / 4, -sqrt(3.0 / 2.0) / 6, sqrt(3.0) / 12),
          Vector3d(0, sqrt(3.0 / 2.0) / 3, sqrt(3.0) / 12),
          Vector3d(sqrt(2.0) / 4, -sqrt(3.0 / 2.0) / 6, sqrt(3.0) / 12),

          Vector3d(0.0, sqrt(6.0) / 6.0, sqrt(3.0) / 3.0),
          Vector3d(-sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
          Vector3d(sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),
          Vector3d(0.0, sqrt(6.0) / 6.0, -sqrt(3.0) / 3.0),
          Vector3d(sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
          Vector3d(-sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, sqrt(3.0) / 3.0)};

      crystalStructure.numNeighbors = 16;
      for (int n1 = 0; n1 < 16; ++n1) {
        crystalStructure.neighborArray.setNeighborBond(n1, n1, false);
        double cutoff = (n1 < 4) ? (sqrt(3.0) * 0.25 + sqrt(0.5)) / 2 : (1.0 + sqrt(0.5)) / 2;
        double cutoffSquared = cutoff * cutoff;
        for (int n2 = n1 + 1; n2 < 4; ++n2) {
          crystalStructure.neighborArray.setNeighborBond(n1, n2, false);
        }
        for (int n2 = std::max(n1 + 1, 4); n2 < 16; ++n2) {
          crystalStructure.neighborArray.setNeighborBond(
              n1, n2,
              ((crystalStructure.latticeVectors[n1] - crystalStructure.latticeVectors[n2])
                   .lengthSquared() < cutoffSquared));
        }
        crystalStructure.cnaSignatures[n1] =
            (n1 < 4) ? 0 : ((crystalStructure.latticeVectors[n1].z() == 0) ? 2 : 1);
      }
      crystalStructure.primitiveCell.column(0) = Vector3d(sqrt(0.5) / 2, -sqrt(6.0) / 4, 0.0);
      crystalStructure.primitiveCell.column(1) = Vector3d(sqrt(0.5) / 2, sqrt(6.0) / 4, 0.0);
      crystalStructure.primitiveCell.column(2) = Vector3d(0.0, 0.0, sqrt(8.0 / 6.0));
    }
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Generate symmetries
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    {
      for (auto &crystalStructure : _crystalStructures) {
        if (crystalStructure.latticeVectors.empty()) { continue; }

        // Find 3 non-coplanar lattice vectors
        std::array<int, 3> nindices;
        Matrix3d mat1;
        for (int i = 0, n = 0; i < crystalStructure.numNeighbors && n < 3; ++i) {
          mat1.column(n) = crystalStructure.latticeVectors[i];
          if (n == 1) {
            if (mat1.column(0).cross(mat1.column(1)).lengthSquared() <= EPSILON) { continue; }
          } else if (n == 2) {
            if (std::abs(mat1.determinant()) < EPSILON) { continue; }
          }
          nindices[n++] = i;
        }
        assert(std::abs(mat1.determinant()) > EPSILON);
        Matrix3d mat1Inv = mat1.inverse();

        // find symmetries
        std::vector<int> permutation(crystalStructure.latticeVectors.size());
        assert(permutation.size() == crystalStructure.latticeVectors.size());
        std::vector<int> lastPermutation(crystalStructure.latticeVectors.size(), -1);
        assert(lastPermutation.size() == crystalStructure.latticeVectors.size());
        std::iota(permutation.begin(), permutation.end(), 0);
        SymmetryPermutation<_maxNeighCount> symmetryPermutation;
        Matrix3d mat2;
        do {
          int changedFrom =
              std::mismatch(permutation.begin(), permutation.end(), lastPermutation.begin()).first -
              permutation.begin();
          assert(changedFrom < crystalStructure.numNeighbors);
          std::copy(permutation.begin(), permutation.end(), lastPermutation.begin());
          if (changedFrom <= nindices[2]) {
            mat2.column(0) = crystalStructure.latticeVectors[permutation[nindices[0]]];
            mat2.column(1) = crystalStructure.latticeVectors[permutation[nindices[1]]];
            mat2.column(2) = crystalStructure.latticeVectors[permutation[nindices[2]]];
            symmetryPermutation.transformation = mat2 * mat1Inv;
            if (!symmetryPermutation.transformation.isOrthogonal(EPSILON)) {
              bitmapSort(permutation.begin() + nindices[2] + 1, permutation.end(),
                         permutation.size());
              continue;
            }
            changedFrom = 0;
          }
          int sortFrom = nindices[2];
          int invalidFrom;
          for (invalidFrom = changedFrom; invalidFrom < crystalStructure.numNeighbors;
               invalidFrom++) {
            Vector3d v =
                symmetryPermutation.transformation * crystalStructure.latticeVectors[invalidFrom];
            if (!v.equals(crystalStructure.latticeVectors[permutation[invalidFrom]], EPSILON))
              break;
          }
          if (invalidFrom == crystalStructure.numNeighbors) {
            std::copy(permutation.begin(), permutation.begin() + crystalStructure.numNeighbors,
                      symmetryPermutation.permutation.begin());
            for (const auto &entry : crystalStructure.permutations) {
              assert(!entry.transformation.equals(symmetryPermutation.transformation, EPSILON));
            }
            crystalStructure.permutations.push_back(symmetryPermutation);
          } else {
            sortFrom = invalidFrom;
          }
          bitmapSort(permutation.begin() + sortFrom + 1, permutation.end(), permutation.size());
        } while (std::next_permutation(permutation.begin(), permutation.end()));

        assert(crystalStructure.permutations.size() >= 1);
        assert(crystalStructure.permutations.front().transformation.equals(Matrix3d::Identity(),
                                                                           EPSILON));
        // Products of symmetry transformations
        for (int s1 = 0; s1 < crystalStructure.permutations.size(); s1++) {
          for (int s2 = 0; s2 < crystalStructure.permutations.size(); s2++) {
            Matrix3d product = crystalStructure.permutations[s2].transformation *
                crystalStructure.permutations[s1].transformation;
            for (int i = 0; i < crystalStructure.permutations.size(); i++) {
              if (crystalStructure.permutations[i].transformation.equals(product, EPSILON)) {
                crystalStructure.permutations[s1].product.push_back(i);
                break;
              }
            }
            assert(crystalStructure.permutations[s1].product.size() == s2 + 1);
            Matrix3d inverseProduct = crystalStructure.permutations[s2].transformation.inverse() *
                crystalStructure.permutations[s1].transformation;
            for (int i = 0; i < crystalStructure.permutations.size(); i++) {
              if (crystalStructure.permutations[i].transformation.equals(product, EPSILON)) {
                crystalStructure.permutations[s1].inverseProduct.push_back(i);
                break;
              }
            }
            assert(crystalStructure.permutations[s1].inverseProduct.size() == s2 + 1);
          }
        }
      }
    }
  }

  bool FixDXA::getCNANeighbors(std::vector<CNANeighbor> &neighborVectors, const int index,
                               const int nn) const
  {
    double **x = atom->x;
    assert(index < _neighList->inum + _neighList->gnum);
    const int i = _neighList->ilist[index];
    const int *jlist = _neighList->firstneigh[i];
    const int jnum = _neighList->numneigh[i];

    if (jnum < nn) { return false; }
    if (jnum > neighborVectors.size()) { neighborVectors.resize(jnum); }

    for (int jj = 0; jj < jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      CNANeighbor &neigh = neighborVectors[jj];
      for (int k = 0; k < 3; ++k) { neigh.xyz[k] = x[i][k] - x[j][k]; }
      neigh.lengthSq = neigh.xyz.lengthSquared();
      neigh.idx = index;
      neigh.neighIdx = j;
    }

    std::partial_sort(neighborVectors.begin(), neighborVectors.begin() + nn,
                      neighborVectors.begin() + jnum);
    return true;
  }

  void FixDXA::identifyCrystalStructure() const
  {
    // Number of neighbors to analyze.
    const int nn = [this]() {
      if (_inputStructure == FCC || _inputStructure == HCP) {
        return 12;
      } else if (_inputStructure == BCC) {
        return 14;
      } else if (_inputStructure == CUBIC_DIA || _inputStructure == HEX_DIA) {
        return 16;
      } else {
        error->all(FLERR, "Implementation error in fix DXA");
        return -1;
      }
    }();

    // adaptive scaling
    std::vector<CNANeighbor> neighborVectors;
    std::vector<CNANeighbor> neighborVectors1;
    std::vector<CNANeighbor> neighborVectors2;
    NeighborBondArray<_maxNeighCount> neighborArray;
    const int inum = _neighList->inum;
    std::vector<StructureType> crystalStructure;
    crystalStructure.resize(inum, OTHER);
    double localCutoff = 0;
    double localScaling = 0;
    for (int ii = 0; ii < inum; ++ii) {
      localScaling = 0;
      neighborArray.reset();
      if (_inputStructure == FCC || _inputStructure == HCP) {
        if (!getCNANeighbors(neighborVectors, ii, nn)) continue;
        for (int n = 0; n < 12; ++n) { localScaling += sqrt(neighborVectors[n].lengthSq); }
        localScaling /= 12;
        localCutoff = localScaling * (1.0 + sqrt(2.0)) * 0.5;
      } else if (_inputStructure == BCC) {
        if (!getCNANeighbors(neighborVectors, ii, nn)) continue;
        for (int n = 0; n < 8; ++n) { localScaling += sqrt(neighborVectors[n].lengthSq); }
        localScaling /= 8;
        localCutoff = localScaling / (sqrt(3.0) / 2.0) * 0.5 * (1.0 + sqrt(2.0));
      } else if (_inputStructure == CUBIC_DIA || _inputStructure == HEX_DIA) {
        int outIndex = 4;
        neighborVectors.resize(16);
        if (!getCNANeighbors(neighborVectors1, ii, 4)) continue;
        for (int n = 0; n < 4; ++n) {
          neighborVectors[n] = std::move(neighborVectors1[n]);
          if (!getCNANeighbors(neighborVectors2, neighborVectors[n].neighIdx, 4)) break;
          for (int m = 0; m < 4; ++m) {
            if (outIndex == 16) break;
            if (neighborVectors2[m].neighIdx == neighborVectors[n].idx &&
                (neighborVectors[n].xyz + neighborVectors2[m].xyz).isZero(EPSILON)) {
              continue;
            }
            neighborVectors[outIndex] = std::move(neighborVectors2[m]);
            neighborVectors[outIndex].xyz = neighborVectors[outIndex].xyz + neighborVectors[n].xyz;
            neighborVectors[outIndex].lengthSq = neighborVectors[outIndex].xyz.lengthSquared();
            neighborArray.setNeighborBond(n, outIndex);
            outIndex++;
          }
          if (outIndex != n * 3 + 7) break;
        }
        if (outIndex != 16) continue;

        for (int n = 4; n < 16; n++) localScaling += sqrt(neighborVectors[n].lengthSq);
        localScaling /= 12;
        localCutoff = localScaling * 1.2071068;

      } else {
        unreachable(lmp);
      }

      double localCutoffSquared = localCutoff * localCutoff;
      // Compute common neighbor bit-flag array.
      if (_inputStructure == FCC || _inputStructure == HCP || _inputStructure == BCC) {
        for (int n1 = 0; n1 < nn; ++n1) {
          neighborArray.setNeighborBond(n1, n1, false);
          for (int n2 = n1 + 1; n2 < nn; ++n2) {
            auto v = neighborVectors[n1].xyz - neighborVectors[n2].xyz;
            if ((neighborVectors[n1].xyz - neighborVectors[n2].xyz).lengthSquared() <=
                localCutoffSquared) {
              neighborArray.setNeighborBond(n1, n2);
            }
          }
        }
      } else if (_inputStructure == CUBIC_DIA || _inputStructure == HEX_DIA) {
        for (int n1 = 4; n1 < nn; ++n1) {
          for (int n2 = n1 + 1; n2 < nn; ++n2)
            if ((neighborVectors[n1].xyz - neighborVectors[n2].xyz).lengthSquared() <=
                localCutoffSquared) {
              neighborArray.setNeighborBond(n1, n2);
            }
        }
      } else {
        unreachable(lmp);
      }

      // int templateStartIndex;
      // int templateEndIndex;
      if (_inputStructure == FCC || _inputStructure == HCP) {
        int n421 = 0;
        int n422 = 0;
        for (int ni = 0; ni < nn; ++ni) {

          int numCommonNeighbors = neighborArray.countCommonNeighbors(ni);
          if (numCommonNeighbors != 4) break;

          std::array<unsigned int, _maxNeighCount * _maxNeighCount> neighborPairBonds;
          neighborPairBonds.fill(0);
          int numNeighborBonds = neighborArray.findNeighborBonds(ni, neighborPairBonds, nn);
          if (numNeighborBonds != 2) break;
          int maxChainLength = NeighborBondArray<_maxNeighCount>::calcMaxChainLength(
              numNeighborBonds, neighborPairBonds);
          if (maxChainLength == 1) {
            n421++;
          } else if (maxChainLength == 2) {
            n422++;
          } else {
            break;
          }
        }
        if (n421 == 12) {    // FCC
          crystalStructure[ii] = FCC;
        } else if (n421 == 6 && n422 == 6) {    // HCP
          crystalStructure[ii] = HCP;
        } else {
          continue;
        }
      } else if (_inputStructure == BCC) {
        int n444 = 0;
        int n666 = 0;
        for (int ni = 0; ni < nn; ni++) {

          // Determine number of neighbors the two atoms have in common.
          unsigned int commonNeighbors;
          int numCommonNeighbors = neighborArray.countCommonNeighbors(ni);
          if (numCommonNeighbors != 4 && numCommonNeighbors != 6) { break; }

          // Determine the number of bonds among the common neighbors.
          std::array<unsigned int, _maxNeighCount * _maxNeighCount> neighborPairBonds;
          int numNeighborBonds = neighborArray.findNeighborBonds(ni, neighborPairBonds, nn);
          if (numNeighborBonds != 4 && numNeighborBonds != 6) { break; }

          // Determine the number of bonds in the longest continuous chain.
          int maxChainLength = NeighborBondArray<_maxNeighCount>::calcMaxChainLength(
              numNeighborBonds, neighborPairBonds);
          if (numCommonNeighbors == 4 && numNeighborBonds == 4 && maxChainLength == 4) {
            n444++;
          } else if (numCommonNeighbors == 6 && numNeighborBonds == 6 && maxChainLength == 6) {
            n666++;
          } else {
            break;
          }
        }
        if (n666 != 8 || n444 != 6) { continue; }
        crystalStructure[ii] = BCC;
      } else if (_inputStructure == CUBIC_DIA || _inputStructure == HEX_DIA) {
        int numCommonNeighbors = 3;
        for (int ni = 0; ni < 4; ni++) {
          int numCommonNeighbors = neighborArray.countCommonNeighbors(ni);
          if (numCommonNeighbors != 3) { break; }
        }
        if (numCommonNeighbors != 3) { continue; }
        int n543 = 0;
        int n544 = 0;
        for (int ni = 4; ni < nn; ni++) {
          unsigned int commonNeighbors;
          int numCommonNeighbors = neighborArray.countCommonNeighbors(ni);
          if (numCommonNeighbors != 5) { break; }

          std::array<unsigned int, _maxNeighCount * _maxNeighCount> neighborPairBonds;
          int numNeighborBonds = neighborArray.findNeighborBonds(ni, neighborPairBonds, nn);
          if (numNeighborBonds != 4) { break; }

          int maxChainLength = NeighborBondArray<_maxNeighCount>::calcMaxChainLength(
              numNeighborBonds, neighborPairBonds);
          if (maxChainLength == 3) {
            n543++;
          } else if (maxChainLength == 4) {
            n544++;
          } else
            break;
        }
        if (n543 == 12) {
          crystalStructure[ii] = CUBIC_DIA;
        } else if (n543 == 6 && n544 == 6) {
          crystalStructure[ii] = HEX_DIA;
        } else {
          continue;
        }
      } else {
        unreachable(lmp);
      }
    }
    std::array<size_t, 6> summary;
    std::fill(summary.begin(), summary.end(), 0);
    for (int ii = 0; ii < atom->nlocal; ++ii) {
      summary[static_cast<size_t>(crystalStructure[ii])] += 1;
    }
    for (auto s : summary) { utils::logmesg(lmp, "\nstructure: {}", s); }
    utils::logmesg(lmp, "\n");

    // Initialize permutation.
    std::array<int, _maxNeighCount> neighborMapping;
    std::iota(neighborMapping.begin(), neighborMapping.end(), 0);
    std::array<int, _maxNeighCount> previousMapping;
    std::fill(previousMapping.begin(), previousMapping.end(), -1);
  }

  void FixDXA::end_of_step()
  {
    if (_inputStructure == FCC || _inputStructure == BCC || _inputStructure == HCP) {
      neighbor->build_one(_neighList);
    }
    identifyCrystalStructure();
  }

  void FixDXA::init()
  {
    if (!(atom->tag_enable))
      error->all(FLERR, "Fix DXA requires atoms having IDs. Please use 'atom_modify id yes'");

    if (_inputStructure == FCC || _inputStructure == BCC || _inputStructure == HCP) {
      neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
    } else if (_inputStructure == CUBIC_DIA || _inputStructure == HEX_DIA) {
      neighbor->add_request(this,
                            NeighConst::REQ_FULL | NeighConst::REQ_DEFAULT | NeighConst::REQ_GHOST);
    } else {
      unreachable(lmp);
    }
  }

  void FixDXA::init_list(int, NeighList *ptr)
  {
    _neighList = ptr;
  }

  int FixDXA::setmask()
  {
    int mask = 0;
    mask |= FixConst::END_OF_STEP;
    return mask;
  }

  void FixDXA::setup(int)
  {
    utils::logmesg(lmp, "Fix DXA version {}\n", VERSION);
  }
}    // namespace FIXDXA_NS
}    // namespace LAMMPS_NS