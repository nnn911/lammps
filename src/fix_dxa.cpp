
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
                (neighborVectors[n].xyz + neighborVectors2[m].xyz).isZero(__DBL_EPSILON__)) {
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
          // templateStartIndex = 0;
          // templateEndIndex = 5;
        } else if (n421 == 6 && n422 == 6) {    // HCP
          crystalStructure[ii] = HCP;
          // templateStartIndex = 5;
          // templateEndIndex = 13;
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