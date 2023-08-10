////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright 2022 OVITO GmbH, Germany
//
//  This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY KIND,
//  either express or implied. See the GPL or the MIT License for the specific language
//  governing rights and limitations.
//
////////////////////////////////////////////////////////////////////////////////////////

#include "fix_dxa.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "random_park.h"
#include "utils.h"
#include <deque>
#include <fstream>
#include <numeric>
#include <string>

// TODO remove
#include <fmt/ranges.h>

// TODO:
// early exit if there's no atom of the input structure type
// avoid crash in distributing clusters to neighbors

// debug macro based on discussion in: https://stackoverflow.com/a/1644898
#define DEBUGLOG 1
#define debugLog(lmp, string, ...)                            \
  do {                                                        \
    if (DEBUGLOG) utils::logmesg(lmp, string, ##__VA_ARGS__); \
  } while (0)

namespace LAMMPS_NS {
namespace FIXDXA_NS {
  [[noreturn]] static void unreachable(LAMMPS *lmp)
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

  /*++++++++++++++++++++++++++++++++++++++++++++
   _                                          
  | |                                         
  | |     __ _ _ __ ___  _ __ ___  _ __  ___  
  | |    / _` | '_ ` _ \| '_ ` _ \| '_ \/ __| 
  | |___| (_| | | | | | | | | | | | |_) \__ \ 
  \_____/\__,_|_| |_| |_|_| |_| |_| .__/|___/ 
                                  | |         
                                  |_|         
  ++++++++++++++++++++++++++++++++++++++++++++*/

  [[nodiscard]] static StructureType getInputStructure(int narg, char **arg, int minNArg)
  {
    if (narg < minNArg) { return OTHER; }
    std::string inputStructure = utils::lowercase(arg[4]);
    if (inputStructure == "bcc") {
      return BCC;
    } else if (inputStructure == "cubicdia") {
      return CUBIC_DIA;
    } else if (inputStructure == "fcc") {
      return FCC;
    } else if (inputStructure == "hcp") {
      return HCP;
    } else if (inputStructure == "hexdia") {
      return HEX_DIA;
    } else {
      return OTHER;
    }
  }

  [[nodiscard]] static int getNeighCount(StructureType input)
  {
    if (input == FCC || input == HCP) {
      return 12;
    } else if (input == BCC) {
      return 14;
    } else if (input == CUBIC_DIA || input == HEX_DIA) {
      return 16;
    } else {
      return -1;
    }
  }

  FixDXA::FixDXA(LAMMPS *lmp, int narg, char **arg) :
      Fix(lmp, narg, arg), _inputStructure{getInputStructure(narg, arg, _minNarg)},
      _neighCount{getNeighCount(_inputStructure)}
  {
    debugLog(lmp, "Begin of FixDXA() on rank {}\n", me);

    if (narg < _minNarg) error->all(FLERR, "Not enough parameters specified for fix DXA");
    if (_inputStructure == OTHER)
      error->all(FLERR, "Invalid input structure parameter for fix DXA");

    this->nevery = utils::inumeric(FLERR, arg[3], true, lmp);
    if (this->nevery < 1) error->all(FLERR, "Invalid timestep parameter for fix DXA");

    static bool structuresInitialized = false;
    if (!structuresInitialized) {
      initializeStructures();
      structuresInitialized = true;
    }
    comm_forward = std::max(2, _neighCount + 1);

    peratom_flag = 1;
    size_peratom_cols = 2;
    peratom_freq = nevery;

    memory->create(_output, atom->nlocal, 2, _outputName.c_str());
    array_atom = _output;
    atom->add_callback(Atom::GROW);
    grow_arrays(atom->nmax);

    MPI_Comm_rank(world, &me);

    sleep(5);

    debugLog(lmp, "End of FixDXA() on rank {}\n", me);
  }

  FixDXA::~FixDXA()
  {
    atom->delete_callback(id, Atom::GROW);
    memory->destroy(_output);
  }

  void FixDXA::end_of_step()
  {
    // Structure Identification
    identifyCrystalStructure();
    // buildClusters();
    // for (int i = 0; i < atom->nlocal; ++i) {
    //   _output[i][0] = static_cast<double>(static_cast<int>(_structureType[i]));
    //   _output[i][1] = static_cast<double>(_atomClusterType[i]);
    // }
    // array_atom = _output;

    // connectClusters();
    // #ifndef NDEBUG
    // write_cluster_transitions();
    // write_cluster_transitions_parallel();
    // #endif
    write_per_rank_atoms();
    // Tessellation
    firstTessllation();
    validateTessllation();
#ifndef NDEBUG
    write_tessellation_parallel();
    write_per_rank_tessellation();
#endif

    buildClustersPostTess();
    for (int i = 0; i < atom->nlocal; ++i) {
      _output[i][0] = static_cast<double>(static_cast<int>(_structureType[i]));
      _output[i][1] = static_cast<double>(_atomClusterType[i]);
    }
    array_atom = _output;

    connectClustersPostTess();
#ifndef NDEBUG
    write_cluster_transitions();
    write_cluster_transitions_parallel();
#endif

    buildEdges();
    updateClustersFromNeighbors();
    assignIdealLatticeVectorsToEdges();
#ifndef NDEBUG
    write_per_rank_edges();
    classifyRegions();
    constructMesh();
#endif
  }

  void FixDXA::init()
  {
    if (!(atom->tag_enable))
      error->all(FLERR, "Fix DXA requires atoms having IDs. Please use 'atom_modify id yes'");
    neighbor->add_request(this,
                          NeighConst::REQ_FULL | NeighConst::REQ_DEFAULT | NeighConst::REQ_GHOST);
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
    utils::logmesg(lmp, "Fix DXA version: {}\n", VERSION);
    end_of_step();
  }

  double FixDXA::memory_usage()
  {
    return 0;
  }

  void FixDXA::grow_arrays(int nmax)
  {
    memory->grow(_output, nmax, 2, _outputName.c_str());
  }

  void FixDXA::copy_arrays(int i, int j, int delflag)
  {
    _output[j][0] = _output[i][0];
    _output[j][1] = _output[i][1];
  }

  void FixDXA::set_arrays(int i)
  {
    _output[i][0] = -1;
    _output[i][1] = -1;
  }

  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    _____ _                   _                    _____    _            _   _  __ _           _   _
   /  ___| |                 | |                  |_   _|  | |          | | (_)/ _(_)         | | (_)
   \ `--.| |_ _ __ _   _  ___| |_ _   _ _ __ ___    | |  __| | ___ _ __ | |_ _| |_ _  ___ __ _| |_ _  ___  _ __
    `--. \ __| '__| | | |/ __| __| | | | '__/ _ \   | | / _` |/ _ \ '_ \| __| |  _| |/ __/ _` | __| |/ _ \| '_ \ 
   /\__/ / |_| |  | |_| | (__| |_| |_| | | |  __/  _| || (_| |  __/ | | | |_| | | | | (_| (_| | |_| | (_) | | | |
   \____/ \__|_|   \__,_|\___|\__|\__,_|_|  \___|  \___/\__,_|\___|_| |_|\__|_|_| |_|\___\__,_|\__|_|\___/|_| |_|
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

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

  std::array<CrystalStructure<FixDXA::_maxNeighCount>, MAXSTRUCTURECOUNT>
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
    // Tabulate common neighbors
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    {
      for (auto &crystalStructure : _crystalStructures) {
        for (int neighIndex = 0; neighIndex < crystalStructure.numNeighbors; neighIndex++) {
          Matrix3d mat;
          mat.column(0) = crystalStructure.latticeVectors[neighIndex];
          bool found = false;
          for (int i1 = 0; i1 < crystalStructure.numNeighbors && !found; i1++) {
            if (!crystalStructure.neighborArray.areNeighbors(neighIndex, i1)) { continue; }
            mat.column(1) = crystalStructure.latticeVectors[i1];
            for (int i2 = i1 + 1; i2 < crystalStructure.numNeighbors; i2++) {
              if (!crystalStructure.neighborArray.areNeighbors(neighIndex, i2)) continue;
              mat.column(2) = crystalStructure.latticeVectors[i2];
              if (std::abs(mat.determinant()) > EPSILON) {
                crystalStructure.commonNeighbors[neighIndex][0] = i1;
                crystalStructure.commonNeighbors[neighIndex][1] = i2;
                found = true;
                break;
              }
            }
          }
          assert(found);
        }
      }
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

  void FixDXA::buildNNList(int ii, int numNeigh)
  {
    // if (atom->tag[ii] == atom->tag[50505]) {
    //   auto p = 5;
    //   ;
    // }
    double **x = atom->x;
    const int inum = _neighList->inum;
    assert(_neighList->inum == atom->nlocal);
    const int gnum = _neighList->gnum;
    assert(_neighList->gnum == atom->nghost);
    const int nmax = inum + gnum;

    const int i = _neighList->ilist[ii];
    const int *jlist = _neighList->firstneigh[i];
    const int jnum = _neighList->numneigh[i];
    assert(jnum > numNeigh);

    _nnListBuffer.resize(std::max(jnum, numNeigh));
    for (int jj = 0; jj < std::max(jnum, numNeigh); ++jj) {
      if (jj < jnum) {
        int j = jlist[jj];
        j &= NEIGHMASK;
        _nnListBuffer[jj] = {j, 0.0};
        for (int k = 0; k < 3; ++k) {
          _nnListBuffer[jj].second += (x[i][k] - x[j][k]) * (x[i][k] - x[j][k]);
        }
      } else {
        _nnListBuffer[jj] = {-1, std::numeric_limits<double>::max()};
      }
    }
    std::partial_sort(_nnListBuffer.begin(), _nnListBuffer.begin() + numNeigh, _nnListBuffer.end(),
                      [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
                        return a.second < b.second;
                      });
    for (int jj = 0; jj < numNeigh; ++jj) { _nnList[jj] = _nnListBuffer[jj].first; }
  }

  // TODO -> this can be a std::array<CNANeighbor. _maxNeighborCount> &neighborVectors
  bool FixDXA::getCNANeighbors(std::vector<CNANeighbor> &neighborVectors, const int ii,
                               const int nn) const
  {
    double **x = atom->x;
    neighborVectors.resize(nn);
    for (int i = 0; i < nn; ++i) {
      CNANeighbor &neigh = neighborVectors[i];
      assert(_nnList[i] != -1);
      for (int k = 0; k < 3; ++k) { neigh.xyz[k] = x[_nnList[i]][k] - x[ii][k]; }
      neigh.lengthSq = neigh.xyz.lengthSquared();
      neigh.idx = ii;
      neigh.neighIdx = _nnList[i];
    }
    return true;
  }

  double FixDXA::getSqNeighDistance(int ii, int jj)
  {
    double **x = atom->x;
    assert(jj < _nnList.size());
    double sqDistance = 0;
    for (int k = 0; k < 3; ++k) {
      sqDistance += (x[_nnList[jj]][k] - x[ii][k]) * (x[_nnList[jj]][k] - x[ii][k]);
    }
    return sqDistance;
  }

  void FixDXA::initialize_neighborIndices(size_t numElements)
  {
    _neighborIndices.resize(numElements);
    for (auto &ni : _neighborIndices) { std::fill(ni.begin(), ni.end(), -1); }
  }

  void FixDXA::identifyCrystalStructure()
  {
    debugLog(lmp, "Start of identifyCrystalStructure() on rank {}\n", me);
    tagint *atomTags = atom->tag;
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Adaptive neighbor cutoff
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    std::vector<CNANeighbor> neighborVectors;
    std::vector<CNANeighbor> neighborVectors1;
    std::vector<CNANeighbor> neighborVectors2;
    NeighborBondArray<_maxNeighCount> neighborArray;

    const int ntotal = atom->nlocal + atom->nghost;
    const int nlocal = atom->nlocal;
    std::array<int, _maxNeighCount> cnaSignatures;
    cnaSignatures.fill(-1);
    _structureType.resize(ntotal, OTHER);
    std::fill(_structureType.begin(), _structureType.end(), OTHER);

    double localCutoffSquared = 0;
    initialize_neighborIndices(ntotal);

    // TODO: This only needs to run for nlocal!
    for (int ii = 0; ii < nlocal; ++ii) {
      {
        auto atomTag = atomTags[ii];
        double localCutoff = 0;
        double localScaling = 0;
        neighborArray.reset();
        if (_inputStructure == FCC || _inputStructure == HCP) {
          buildNNList(ii, _neighCount + 1);
          if (!getCNANeighbors(neighborVectors, ii, _neighCount)) continue;
          for (int n = 0; n < 12; ++n) { localScaling += sqrt(neighborVectors[n].lengthSq); }
          localScaling /= 12;
          localCutoff = localScaling * (1.0 + sqrt(2.0)) * 0.5;
          localCutoffSquared = localCutoff * localCutoff;
          // Skip over coordinated atoms
          if (getSqNeighDistance(ii, _neighCount) < localCutoffSquared) {
            _structureType[ii] = OTHER;
            continue;
          }
        } else if (_inputStructure == BCC) {
          buildNNList(ii, _neighCount + 1);
          if (!getCNANeighbors(neighborVectors, ii, _neighCount)) continue;
          for (int n = 0; n < 8; ++n) { localScaling += sqrt(neighborVectors[n].lengthSq); }
          localScaling /= 8;
          localCutoff = localScaling / (sqrt(3.0) / 2.0) * 0.5 * (1.0 + sqrt(2.0));
          localCutoffSquared = localCutoff * localCutoff;
          // Skip over coordinated atoms
          if (getSqNeighDistance(ii, _neighCount) < localCutoffSquared) {
            _structureType[ii] = OTHER;
            continue;
          }
        } else if (_inputStructure == CUBIC_DIA || _inputStructure == HEX_DIA) {
          int outIndex = 4;
          neighborVectors.resize(16);
          buildNNList(ii, 4);
          if (!getCNANeighbors(neighborVectors1, ii, 4)) continue;
          for (int n = 0; n < 4; ++n) {
            neighborVectors[n] = std::move(neighborVectors1[n]);
            buildNNList(neighborVectors[n].neighIdx, 4);
            if (!getCNANeighbors(neighborVectors2, neighborVectors[n].neighIdx, 4)) { break; }
            for (int m = 0; m < 4; ++m) {
              if (neighborVectors2[m].neighIdx == neighborVectors[n].idx &&
                  (neighborVectors[n].xyz + neighborVectors2[m].xyz).isZero(EPSILON)) {
                continue;
              }
              if (outIndex == 16) {
                outIndex++;
                break;
              }
              neighborVectors[outIndex] = std::move(neighborVectors2[m]);
              neighborVectors[outIndex].xyz =
                  neighborVectors[outIndex].xyz + neighborVectors[n].xyz;
              neighborVectors[outIndex].lengthSq = neighborVectors[outIndex].xyz.lengthSquared();
              neighborArray.setNeighborBond(n, outIndex);
              outIndex++;
            }
            if (outIndex != n * 3 + 7) { break; }
          }
          if (outIndex != 16) { continue; }

          for (int n = 4; n < 16; n++) { localScaling += sqrt(neighborVectors[n].lengthSq); }
          localScaling /= 12;
          localCutoff = localScaling * 1.2071068;
          localCutoffSquared = localCutoff * localCutoff;
        } else {
          unreachable(lmp);
        }
      }

      {
        // Compute common neighbor bit-flag array.
        if (_inputStructure == FCC || _inputStructure == HCP || _inputStructure == BCC) {
          for (int n1 = 0; n1 < _neighCount; ++n1) {
            neighborArray.setNeighborBond(n1, n1, false);
            for (int n2 = n1 + 1; n2 < _neighCount; ++n2) {
              auto v = neighborVectors[n1].xyz - neighborVectors[n2].xyz;
              if ((neighborVectors[n1].xyz - neighborVectors[n2].xyz).lengthSquared() <=
                  localCutoffSquared) {
                neighborArray.setNeighborBond(n1, n2);
              }
            }
          }
        } else if (_inputStructure == CUBIC_DIA || _inputStructure == HEX_DIA) {
          for (int n1 = 4; n1 < _neighCount; ++n1) {
            for (int n2 = n1 + 1; n2 < _neighCount; ++n2)
              if ((neighborVectors[n1].xyz - neighborVectors[n2].xyz).lengthSquared() <=
                  localCutoffSquared) {
                neighborArray.setNeighborBond(n1, n2);
              }
          }
        } else {
          unreachable(lmp);
        }
      }

      //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      // Core CNA
      //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      StructureType structureType = OTHER;
      {
        if (_inputStructure == FCC || _inputStructure == HCP) {
          int n421 = 0;
          int n422 = 0;
          for (int ni = 0; ni < _neighCount; ++ni) {

            int numCommonNeighbors = neighborArray.countCommonNeighbors(ni);
            if (numCommonNeighbors != 4) break;

            std::array<unsigned int, _maxNeighCount * _maxNeighCount> neighborPairBonds;
            neighborPairBonds.fill(0);
            int numNeighborBonds =
                neighborArray.findNeighborBonds(ni, neighborPairBonds, _neighCount);
            if (numNeighborBonds != 2) break;
            int maxChainLength = NeighborBondArray<_maxNeighCount>::calcMaxChainLength(
                numNeighborBonds, neighborPairBonds);
            if (maxChainLength == 1) {
              cnaSignatures[ni] = 0;
              n421++;
            } else if (maxChainLength == 2) {
              cnaSignatures[ni] = 1;
              n422++;
            } else {
              break;
            }
          }
          if (n421 == 12) {    // FCC
            structureType = FCC;
          } else if (n421 == 6 && n422 == 6) {    // HCP
            structureType = HCP;
          } else {
            continue;
          }
        } else if (_inputStructure == BCC) {
          int n444 = 0;
          int n666 = 0;
          for (int ni = 0; ni < _neighCount; ni++) {

            // Determine number of neighbors the two atoms have in common.
            unsigned int commonNeighbors;
            int numCommonNeighbors = neighborArray.countCommonNeighbors(ni);
            if (numCommonNeighbors != 4 && numCommonNeighbors != 6) { break; }

            // Determine the number of bonds among the common neighbors.
            std::array<unsigned int, _maxNeighCount * _maxNeighCount> neighborPairBonds;
            int numNeighborBonds =
                neighborArray.findNeighborBonds(ni, neighborPairBonds, _neighCount);
            if (numNeighborBonds != 4 && numNeighborBonds != 6) { break; }

            // Determine the number of bonds in the longest continuous chain.
            int maxChainLength = NeighborBondArray<_maxNeighCount>::calcMaxChainLength(
                numNeighborBonds, neighborPairBonds);
            if (numCommonNeighbors == 4 && numNeighborBonds == 4 && maxChainLength == 4) {
              cnaSignatures[ni] = 1;
              n444++;
            } else if (numCommonNeighbors == 6 && numNeighborBonds == 6 && maxChainLength == 6) {
              cnaSignatures[ni] = 0;
              n666++;
            } else {
              break;
            }
          }
          if (n666 != 8 || n444 != 6) { continue; }
          structureType = BCC;
        } else if (_inputStructure == CUBIC_DIA || _inputStructure == HEX_DIA) {
          int numCommonNeighbors;
          for (int ni = 0; ni < 4; ni++) {
            cnaSignatures[ni] = 0;
            numCommonNeighbors = neighborArray.countCommonNeighbors(ni);
            if (numCommonNeighbors != 3) { break; }
          }
          if (numCommonNeighbors != 3) { continue; }
          int n543 = 0;
          int n544 = 0;
          for (int ni = 4; ni < _neighCount; ni++) {
            unsigned int commonNeighbors;
            numCommonNeighbors = neighborArray.countCommonNeighbors(ni);
            if (numCommonNeighbors != 5) { break; }

            std::array<unsigned int, _maxNeighCount * _maxNeighCount> neighborPairBonds;
            int numNeighborBonds =
                neighborArray.findNeighborBonds(ni, neighborPairBonds, _neighCount);
            if (numNeighborBonds != 4) { break; }

            int maxChainLength = NeighborBondArray<_maxNeighCount>::calcMaxChainLength(
                numNeighborBonds, neighborPairBonds);
            if (maxChainLength == 3) {
              cnaSignatures[ni] = 1;
              n543++;
            } else if (maxChainLength == 4) {
              cnaSignatures[ni] = 2;
              n544++;
            } else
              break;
          }
          if (n543 == 12) {
            structureType = CUBIC_DIA;
          } else if (n543 == 6 && n544 == 6) {
            structureType = HEX_DIA;
          } else {
            continue;
          }
        } else {
          unreachable(lmp);
        }

        assert(structureType != OTHER);
        if (_maxNeighDistance < _nnListBuffer[_neighCount - 1].second) {
          _maxNeighDistance = _nnListBuffer[_neighCount - 1].second;
        }
      }
      //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      // Map neighbor crystal directions
      //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      {
        std::array<int, _maxNeighCount> neighborMapping;
        std::iota(neighborMapping.begin(), neighborMapping.end(), 0);
        std::array<int, _maxNeighCount> previousMapping;
        std::fill(previousMapping.begin(), previousMapping.end(), -1);
        const auto &crystalStructure = _crystalStructures[structureType];

        while (true) {
          int n1 = 0;
          while (neighborMapping[n1] == previousMapping[n1]) { n1++; }
          for (; n1 < _neighCount; n1++) {
            int atomNeighborIndex1 = neighborMapping[n1];
            previousMapping[n1] = atomNeighborIndex1;
            if (cnaSignatures[atomNeighborIndex1] != crystalStructure.cnaSignatures[n1]) { break; }
            int n2;
            for (n2 = 0; n2 < n1; n2++) {
              int atomNeighborIndex2 = neighborMapping[n2];
              if (neighborArray.areNeighbors(atomNeighborIndex1, atomNeighborIndex2) !=
                  crystalStructure.neighborArray.areNeighbors(n1, n2)) {
                break;
              }
            }
            if (n2 != n1) { break; }
          }
          if (n1 == _neighCount) {
            _structureType[ii] = structureType;
            // Save the atom's neighbor list.
            for (int i = 0; i < _neighCount; i++) {
              assert(neighborMapping[i] < _neighCount);
#ifndef NDEBUG
              for (int boxIdx = 0; boxIdx < 3; ++boxIdx) {
                if (domain->periodicity[boxIdx] &&
                    ((domain->prd[boxIdx] / 2) < neighborVectors[neighborMapping[i]].xyz[boxIdx])) {
                  lmp->error->all(FLERR, "Simulation cell too small!\n");
                }
              }
#endif
              _neighborIndices[ii][i] = neighborVectors[neighborMapping[i]].neighIdx;
            }
            break;
          } else {
            bitmapSort(neighborMapping.begin() + n1 + 1, neighborMapping.begin() + _neighCount,
                       _neighCount);
            if (!std::next_permutation(neighborMapping.begin(),
                                       neighborMapping.begin() + _neighCount)) {
              unreachable(lmp);
            }
          }
        }
      }
    }
    std::array<size_t, MAXSTRUCTURECOUNT> summary;
    std::fill(summary.begin(), summary.end(), 0);
    for (int ii = 0; ii < atom->nlocal; ++ii) {
      summary[static_cast<size_t>(_structureType[ii])] += 1;
    }
    debugLog(lmp, "Rank {}:\n", me);
    for (int i = 0; i < summary.size(); ++i) {
      debugLog(lmp, "\nstructure {}: {} / {}", i, summary[i], atom->nlocal);
    }
    debugLog(lmp, "\n");

    pack_neighborIndices_forward_comm();
    _commStep = STRUCTURE_NEIGHS;
    comm->forward_comm(this, _neighCount + 1);
    _commStep = NOCOM;
    unpack_neighborIndices_forward_comm();

    debugLog(lmp, "End of identifyCrystalStructure() on rank {}\n", me);
  }

  static inline Vector3d xToVector(double *x)
  {
    return {x[0], x[1], x[2]};
  }

  void FixDXA::buildClusters()
  {
    debugLog(lmp, "Start of buildClusters() on rank {}\n", me);

    tagint *atomTags = atom->tag;
    double **x = atom->x;

    _atomClusterType.resize(atom->nmax);

    std::fill(_atomClusterType.begin(), _atomClusterType.end(), INVALID);

    _atomSymmetryPermutations.resize(atom->nmax);
    std::fill(_atomSymmetryPermutations.begin(), _atomSymmetryPermutations.end(), 0);

    std::deque<int> atomQueue{};

    for (int ii = 0; ii < atom->nlocal; ++ii) {

      if (_atomClusterType[ii] != INVALID) continue;
      if (_structureType[ii] == OTHER) continue;
      int clusterIndex = _clusterGraph.addCluster(atomTags[ii], _structureType[ii]);
      int clusterId = _clusterGraph.getCluster(clusterIndex).id;

      _atomClusterType[ii] = clusterId;
      Matrix3d orientationV = Matrix3d::Zero();
      Matrix3d orientationW = Matrix3d::Zero();
      const auto &crystalStructure = _crystalStructures[_structureType[ii]];

      atomQueue.clear();
      atomQueue.push_back(ii);
      while (!atomQueue.empty()) {
        int currentAtom = atomQueue.front();
        atomQueue.pop_front();
        const int symmetryPermutation = _atomSymmetryPermutations[currentAtom];
        const auto &permutation = crystalStructure.permutations[symmetryPermutation].permutation;

        const Vector3d iiPosition = xToVector(x[currentAtom]);
        for (int jj = 0; jj < _neighCount; ++jj) {

          // TODO: HERE we might be able to skip all ghost atoms -> because we don't need their orientation etc. It will be replaced in the comm step!

          const int neighIdx = _neighborIndices[currentAtom][jj];
          assert(neighIdx != -1);
          const Vector3d &latticeVector = crystalStructure.latticeVectors[permutation[jj]];
          Vector3d spatialVector = xToVector(x[neighIdx]) - iiPosition;
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              orientationV(i, j) += (latticeVector[j] * latticeVector[i]);
              orientationW(i, j) += (latticeVector[j] * spatialVector[i]);
            }
          }

          if (_atomClusterType[neighIdx] != INVALID) { continue; }
          if (_structureType[neighIdx] != _inputStructure) { continue; }

          Matrix3d tm1, tm2;
          bool overlap = true;
          for (int i = 0; i < 3; i++) {
            int atomIndex;
            if (i != 2) {
              atomIndex = _neighborIndices[currentAtom][crystalStructure.commonNeighbors[jj][i]];
              assert(atomIndex != -1);
              tm1.column(i) =
                  crystalStructure
                      .latticeVectors[permutation[crystalStructure.commonNeighbors[jj][i]]] -
                  crystalStructure.latticeVectors[permutation[jj]];
            } else {
              atomIndex = currentAtom;
              tm1.column(i) = -crystalStructure.latticeVectors[permutation[jj]];
            }
            auto pos = std::find(_neighborIndices[neighIdx].begin(),
                                 _neighborIndices[neighIdx].begin() + _neighCount, atomIndex);
            if (*pos != atomIndex) {
              overlap = false;
              break;
            }
            auto d = std::distance(_neighborIndices[neighIdx].begin(), pos);
            assert(d < _neighCount);
            tm2.column(i) = crystalStructure.latticeVectors[d];
          }
          if (!overlap) { continue; }
          // return;
          assert(tm1.invertible());
          Matrix3d tm2inverse;
          if (!tm2.inverse(tm2inverse)) { continue; }
          Matrix3d transition = tm1 * tm2inverse;

          for (int i = 0; i < crystalStructure.permutations.size(); ++i) {
            if (transition.equals(crystalStructure.permutations[i].transformation,
                                  TRANSITION_MATRIX_EPSILON)) {
              assert(_atomClusterType[ii] == _atomClusterType[currentAtom]);
              _atomClusterType[neighIdx] = _atomClusterType[ii];
              _atomSymmetryPermutations[neighIdx] = i;
              if (neighIdx < atom->nlocal) { atomQueue.push_back(neighIdx); }
              break;
            }
          }
        }
      }    // end of while loop
      assert(std::abs(orientationV.determinant()) > EPSILON);
      _clusterGraph.setClusterOrientation(clusterIndex, orientationW * orientationV.inverse());
#if 0
      const Matrix3d alignedOrientation = Matrix3d::Identity();
      if (_structureType[ii] == _inputStructure) {
        Cluster &currentCluster = _clusterGraph.clusters[clusterIndex];
        double bestDeviation = std::numeric_limits<double>::max();
        const Matrix3d &originalOrientation = _clusterGraph.clusters[clusterIndex].orientation;
        for (size_t symPermIdx = 0; symPermIdx < crystalStructure.permutations.size();
             ++symPermIdx) {
          const Matrix3d &symmetryTMatrix =
              crystalStructure.permutations[symPermIdx].transformation;
          const Matrix3d newOrientation = originalOrientation * symmetryTMatrix.inverse();
          const double scaling = std::cbrt(std::abs(newOrientation.determinant()));
          double deviation = 0;
          for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
              deviation += std::abs(newOrientation(i, j) / scaling - alignedOrientation(i, j));
            }
          }
          if (deviation < bestDeviation) {
            bestDeviation = deviation;
            currentCluster.symmetryPermutationIndex = symPermIdx;
            currentCluster.orientation = newOrientation;
          }
          if (bestDeviation == 0) { break; }
        }
      }
#endif
    }    // end of for loop
#if 0
    for (int ii = 0; ii < atom->nlocal; ++ii) {
      const int cluster = _atomClusterType[ii];
      if (cluster == 0) { continue; }
      const int clusterIdx = _clusterGraph.findCluster(cluster);
      assert(clusterIdx < _clusterGraph.clusters.size());

      Cluster &currentCluster = _clusterGraph.clusters[clusterIdx];
      if (currentCluster.symmetryPermutationIndex == 0) { continue; }

      const auto &crystalStructure = _crystalStructures[currentCluster.structure];
      int oldSymmetryPermutation = _atomSymmetryPermutations[ii];
      int newSymmetryPermutation = crystalStructure.permutations[oldSymmetryPermutation]
                                       .inverseProduct[currentCluster.symmetryPermutationIndex];
      _atomSymmetryPermutations[ii] = newSymmetryPermutation;
    }
#endif

    _commStep = CLUSTER;
    comm->forward_comm(this, 2);
    _commStep = NOCOM;

    debugLog(lmp, "End of buildClusters() on rank {}\n", me);
  }

  void FixDXA::buildClustersPostTess()
  {
    debugLog(lmp, "Start of buildClustersPostTess() on rank {}\n", me);

    tagint *atomTags = atom->tag;
    double **x = atom->x;

    _atomClusterType.resize(atom->nmax);

    std::fill(_atomClusterType.begin(), _atomClusterType.end(), INVALID);

    _atomSymmetryPermutations.resize(atom->nmax);
    std::fill(_atomSymmetryPermutations.begin(), _atomSymmetryPermutations.end(), 0);

    std::deque<int> atomQueue{};
    int ntotal = atom->nlocal + atom->nghost;
    for (int ii = 0; ii < ntotal; ++ii) {
      if (!_dt.vertexIsRequired(ii)) { continue; }
      if (_atomClusterType[ii] != INVALID) continue;
      if (_structureType[ii] == OTHER) continue;
      int clusterIndex = _clusterGraph.addCluster(atomTags[ii], _structureType[ii]);
      int clusterId = _clusterGraph.getCluster(clusterIndex).id;

      _atomClusterType[ii] = clusterId;
      Matrix3d orientationV = Matrix3d::Zero();
      Matrix3d orientationW = Matrix3d::Zero();
      const auto &crystalStructure = _crystalStructures[_structureType[ii]];

      atomQueue.clear();
      atomQueue.push_back(ii);
      while (!atomQueue.empty()) {
        int currentAtom = atomQueue.front();
        atomQueue.pop_front();
        const int symmetryPermutation = _atomSymmetryPermutations[currentAtom];
        const auto &permutation = crystalStructure.permutations[symmetryPermutation].permutation;

        const Vector3d iiPosition = xToVector(x[currentAtom]);
        for (int jj = 0; jj < _neighCount; ++jj) {

          // TODO: HERE we might be able to skip all ghost atoms -> because we don't need their orientation etc. It will be replaced in the comm step!

          const int neighIdx = _neighborIndices[currentAtom][jj];
          if (neighIdx == -1 && currentAtom > atom->nlocal) { continue; }
          assert(neighIdx != -1);

          const Vector3d &latticeVector = crystalStructure.latticeVectors[permutation[jj]];
          Vector3d spatialVector = xToVector(x[neighIdx]) - iiPosition;
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              orientationV(i, j) += (latticeVector[j] * latticeVector[i]);
              orientationW(i, j) += (latticeVector[j] * spatialVector[i]);
            }
          }

          if (_atomClusterType[neighIdx] != INVALID) { continue; }
          if (_structureType[neighIdx] != _inputStructure) { continue; }

          Matrix3d tm1, tm2;
          bool overlap = true;
          for (int i = 0; i < 3; i++) {
            int atomIndex;
            if (i != 2) {
              atomIndex = _neighborIndices[currentAtom][crystalStructure.commonNeighbors[jj][i]];
              // assert(atomIndex != -1);
              if (atomIndex == -1) {
                overlap = false;
                break;
              }
              tm1.column(i) =
                  crystalStructure
                      .latticeVectors[permutation[crystalStructure.commonNeighbors[jj][i]]] -
                  crystalStructure.latticeVectors[permutation[jj]];
            } else {
              atomIndex = currentAtom;
              tm1.column(i) = -crystalStructure.latticeVectors[permutation[jj]];
            }
            auto pos = std::find(_neighborIndices[neighIdx].begin(),
                                 _neighborIndices[neighIdx].begin() + _neighCount, atomIndex);
            if (*pos != atomIndex) {
              overlap = false;
              break;
            }
            auto d = std::distance(_neighborIndices[neighIdx].begin(), pos);
            assert(d < _neighCount);
            tm2.column(i) = crystalStructure.latticeVectors[d];
          }
          if (!overlap) { continue; }
          // return;
          assert(tm1.invertible());
          Matrix3d tm2inverse;
          if (!tm2.inverse(tm2inverse)) { continue; }
          Matrix3d transition = tm1 * tm2inverse;

          for (int i = 0; i < crystalStructure.permutations.size(); ++i) {
            if (transition.equals(crystalStructure.permutations[i].transformation,
                                  TRANSITION_MATRIX_EPSILON)) {
              assert(_atomClusterType[ii] == _atomClusterType[currentAtom]);
              _atomClusterType[neighIdx] = _atomClusterType[ii];
              _atomSymmetryPermutations[neighIdx] = i;
              if (_dt.vertexIsRequired(neighIdx)) { atomQueue.push_back(neighIdx); }
              break;
            }
          }
        }
      }    // end of while loop
      assert(std::abs(orientationV.determinant()) > EPSILON);
      _clusterGraph.setClusterOrientation(clusterIndex, orientationW * orientationV.inverse());
#if 0
      const Matrix3d alignedOrientation = Matrix3d::Identity();
      if (_structureType[ii] == _inputStructure) {
        Cluster &currentCluster = _clusterGraph.clusters[clusterIndex];
        double bestDeviation = std::numeric_limits<double>::max();
        const Matrix3d &originalOrientation = _clusterGraph.clusters[clusterIndex].orientation;
        for (size_t symPermIdx = 0; symPermIdx < crystalStructure.permutations.size();
             ++symPermIdx) {
          const Matrix3d &symmetryTMatrix =
              crystalStructure.permutations[symPermIdx].transformation;
          const Matrix3d newOrientation = originalOrientation * symmetryTMatrix.inverse();
          const double scaling = std::cbrt(std::abs(newOrientation.determinant()));
          double deviation = 0;
          for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
              deviation += std::abs(newOrientation(i, j) / scaling - alignedOrientation(i, j));
            }
          }
          if (deviation < bestDeviation) {
            bestDeviation = deviation;
            currentCluster.symmetryPermutationIndex = symPermIdx;
            currentCluster.orientation = newOrientation;
          }
          if (bestDeviation == 0) { break; }
        }
      }
#endif
    }    // end of for loop
#if 0
    for (int ii = 0; ii < atom->nlocal; ++ii) {
      const int cluster = _atomClusterType[ii];
      if (cluster == 0) { continue; }
      const int clusterIdx = _clusterGraph.findCluster(cluster);
      assert(clusterIdx < _clusterGraph.clusters.size());

      Cluster &currentCluster = _clusterGraph.clusters[clusterIdx];
      if (currentCluster.symmetryPermutationIndex == 0) { continue; }

      const auto &crystalStructure = _crystalStructures[currentCluster.structure];
      int oldSymmetryPermutation = _atomSymmetryPermutations[ii];
      int newSymmetryPermutation = crystalStructure.permutations[oldSymmetryPermutation]
                                       .inverseProduct[currentCluster.symmetryPermutationIndex];
      _atomSymmetryPermutations[ii] = newSymmetryPermutation;
    }
#endif

    _commStep = CLUSTER;
    comm->forward_comm(this, 2);
    _commStep = NOCOM;

    debugLog(lmp, "End of buildClustersPostTess() on rank {}\n", me);
  }

  bool FixDXA::addNeighborIndex(int neighListIndex, int indexToAdd)
  {
    for (size_t i = 0; i < _neighCount; ++i) {
      if (_neighborIndices[neighListIndex][i] == -1) {
        _neighborIndices[neighListIndex][i] = indexToAdd;
        return true;
      }
    }
    return false;
  }

  void FixDXA::connectClusters()
  {
    debugLog(lmp, "Start of connectClusters() on rank {}\n", me);
    tagint *atomTags = atom->tag;

    for (int currentAtom = 0; currentAtom < atom->nlocal; ++currentAtom) {

      auto atomTag = atomTags[currentAtom];

      // Cluster of current atom
      if (_atomClusterType[currentAtom] == INVALID) { continue; }
      assert(_clusterGraph.containsCluster(_atomClusterType[currentAtom]));
      size_t cluster1Index = _clusterGraph.findCluster(_atomClusterType[currentAtom]);

      // Structure of the current atom
      const auto &crystalStructure = _crystalStructures[_structureType[currentAtom]];
      const auto &permutation =
          crystalStructure.permutations[_atomSymmetryPermutations[currentAtom]].permutation;

      // Visit neighbors of the current atom.
      for (int jj = 0; jj < _neighCount; ++jj) {
        const int neighIdx = _neighborIndices[currentAtom][jj];
        assert(neighIdx != -1);

        // Skip neighbor atoms belonging to the same cluster or to no cluster at all.
        if (_atomClusterType[neighIdx] == INVALID ||
            _atomClusterType[currentAtom] == _atomClusterType[neighIdx]) {
          bool success = addNeighborIndex(neighIdx, currentAtom);
          continue;
        }

        // Skip if there is already a transition between the two clusters.
        if (_clusterGraph.containsTransition(_atomClusterType[currentAtom],
                                             _atomClusterType[neighIdx])) {
          continue;
        }

        // Find common neighbors of central and neighboring atom
        Matrix3d tm1, tm2;
        bool overlap = true;
        for (int i = 0; i < 3; i++) {
          int atomIndex;
          if (i != 2) {
            atomIndex = _neighborIndices[currentAtom][crystalStructure.commonNeighbors[jj][i]];
            assert(atomIndex != -1);
            tm1.column(i) =
                crystalStructure
                    .latticeVectors[permutation[crystalStructure.commonNeighbors[jj][i]]] -
                crystalStructure.latticeVectors[permutation[jj]];
          } else {
            atomIndex = currentAtom;
            tm1.column(i) = -crystalStructure.latticeVectors[permutation[jj]];
          }
          auto pos = std::find(_neighborIndices[neighIdx].begin(),
                               _neighborIndices[neighIdx].begin() + _neighCount, atomIndex);
          if (*pos != atomIndex) {
            overlap = false;
            break;
          }
          auto d = std::distance(_neighborIndices[neighIdx].begin(), pos);
          assert(d < _neighCount);

          // Look up symmetry permutation of neighbor atom.
          const auto &neighCrystalStructure = _crystalStructures[_structureType[neighIdx]];
          const auto &neighPermutation =
              neighCrystalStructure.permutations[_atomSymmetryPermutations[neighIdx]].permutation;

          tm2.column(i) = neighCrystalStructure.latticeVectors[neighPermutation[d]];
        }
        if (!overlap) { continue; }

        assert(tm1.invertible());
        Matrix3d tm1inverse;
        // !TODO! This shoud never be a continue!
        if (!tm1.inverse(tm1inverse)) { continue; }
        Matrix3d transition = tm2 * tm1inverse;

        if (transition.isOrthogonal(EPSILON)) {
          _clusterGraph.addClusterTransition(_atomClusterType[currentAtom],
                                             _atomClusterType[neighIdx], transition);
        }
      }
    }

    debugLog(lmp, "Clusters on rank {}: {}\n", me, _clusterGraph.numClusters());
    debugLog(lmp, "Clusters transitions on rank {}: {}\n", me, _clusterGraph.numTransitions());

    debugLog(lmp, "End of connectClusters() on rank {}\n", me);
  }

  void FixDXA::connectClustersPostTess()
  {
    debugLog(lmp, "Start of connectClustersPostTess() on rank {}\n", me);
    tagint *atomTags = atom->tag;

    int ntotal = atom->nlocal + atom->nghost;
    for (int currentAtom = 0; currentAtom < ntotal; ++currentAtom) {
      if (!_dt.vertexIsRequired(currentAtom)) { continue; }
      // for (size_t cell = 0; cell < _dt.numCells(); ++cell) {
      // if (!_dt.cellIsRequired(cell)) { continue; }
      // for (size_t vert = 0; vert < 4; ++vert) {
      // int currentAtom = _dt.cellVertex(cell, vert);
      auto atomTag = atomTags[currentAtom];

      // Cluster of current atom
      if (_atomClusterType[currentAtom] == INVALID) { continue; }
      // assert(_clusterGraph.containsCluster(_atomClusterType[currentAtom]));

      // Structure of the current atom
      const auto &crystalStructure = _crystalStructures[_structureType[currentAtom]];
      const auto &permutation =
          crystalStructure.permutations[_atomSymmetryPermutations[currentAtom]].permutation;

      // Visit neighbors of the current atom.
      for (int jj = 0; jj < _neighCount; ++jj) {
        const int neighIdx = _neighborIndices[currentAtom][jj];
        if (neighIdx == -1) { continue; }
        // assert(neighIdx != -1);

        // Skip neighbor atoms belonging to the same cluster or to no cluster at all.
        if (_atomClusterType[neighIdx] == INVALID ||
            _atomClusterType[currentAtom] == _atomClusterType[neighIdx]) {
          bool success = addNeighborIndex(neighIdx, currentAtom);
          continue;
        }

        // Skip if there is already a transition between the two clusters.
        if (_clusterGraph.containsTransition(_atomClusterType[currentAtom],
                                             _atomClusterType[neighIdx])) {
          continue;
        }

        // Find common neighbors of central and neighboring atom
        Matrix3d tm1, tm2;
        bool overlap = true;
        for (int i = 0; i < 3; i++) {
          int atomIndex;
          if (i != 2) {
            atomIndex = _neighborIndices[currentAtom][crystalStructure.commonNeighbors[jj][i]];
            // assert(atomIndex != -1);
            if (atomIndex == -1) {
              overlap = false;
              break;
            }
            tm1.column(i) =
                crystalStructure
                    .latticeVectors[permutation[crystalStructure.commonNeighbors[jj][i]]] -
                crystalStructure.latticeVectors[permutation[jj]];
          } else {
            atomIndex = currentAtom;
            tm1.column(i) = -crystalStructure.latticeVectors[permutation[jj]];
          }
          auto pos = std::find(_neighborIndices[neighIdx].begin(),
                               _neighborIndices[neighIdx].begin() + _neighCount, atomIndex);
          if (*pos != atomIndex) {
            overlap = false;
            break;
          }
          auto d = std::distance(_neighborIndices[neighIdx].begin(), pos);
          assert(d < _neighCount);

          // Look up symmetry permutation of neighbor atom.
          const auto &neighCrystalStructure = _crystalStructures[_structureType[neighIdx]];
          const auto &neighPermutation =
              neighCrystalStructure.permutations[_atomSymmetryPermutations[neighIdx]].permutation;

          tm2.column(i) = neighCrystalStructure.latticeVectors[neighPermutation[d]];
        }
        if (!overlap) { continue; }

        assert(tm1.invertible());
        Matrix3d tm1inverse;
        // !TODO! This shoud never be a continue!
        if (!tm1.inverse(tm1inverse)) { continue; }
        Matrix3d transition = tm2 * tm1inverse;

        if (transition.isOrthogonal(EPSILON)) {
          _clusterGraph.addClusterTransition(_atomClusterType[currentAtom],
                                             _atomClusterType[neighIdx], transition);
        }
      }
    }
    // }

    debugLog(lmp, "Clusters on rank {}: {}\n", me, _clusterGraph.numClusters());
    debugLog(lmp, "Clusters transitions on rank {}: {}\n", me, _clusterGraph.numTransitions());

    debugLog(lmp, "End of connectClustersPostTess() on rank {}\n", me);
  }

  void FixDXA::write_cluster_transitions() const
  {
    debugLog(lmp, "Start of write_cluster_transitions() on rank {}\n", me);

    const std::string fname = fmt::format("cluster_transition_rank_{}.data", me);
    std::ofstream outFile(fname);
    if (!outFile) { error->all(FLERR, "Could not open {} for write.", fname); }
    outFile << _clusterGraph.numTransitions() << '\n';
    for (size_t i = 0; i < _clusterGraph.numTransitions(); ++i) {
      const ClusterTransition &t = _clusterGraph.getTransition(i);
      outFile << fmt::format(
          "{} {} {} {} {} {} {} {} {} {} {}\n", t.cluster1, t.cluster2, t.transition.column(0)[0],
          t.transition.column(1)[0], t.transition.column(2)[0], t.transition.column(0)[1],
          t.transition.column(1)[1], t.transition.column(2)[1], t.transition.column(0)[2],
          t.transition.column(1)[2], t.transition.column(2)[2]);
    }
    debugLog(lmp, "End of write_cluster_transitions() on rank {}\n", me);
  }

  void FixDXA::write_cluster_transitions_parallel() const
  {
    debugLog(lmp, "Start of write_cluster_transitions_parallel() on rank {}\n", me);

    int nprocs;
    MPI_Comm_size(world, &nprocs);

    const int sbuf = _clusterGraph.numTransitions();
    std::vector<int> rbuf;
    rbuf.resize(nprocs);
    MPI_Allgather(&sbuf, 1, MPI_INT, rbuf.data(), 1, MPI_INT, world);

    const std::string fname = fmt::format("cluster_transition.bin.data", me);
    MPI_File outFile;
    MPI_File_open(world, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outFile);

    const size_t headerSize = (me == 0) ? 0 : 1;
    const size_t entryCount = std::accumulate(rbuf.begin(), rbuf.begin() + me, (size_t) 0);
    const size_t entriesPerTransition = 2 + 9;
    const MPI_Offset offset = (headerSize + entryCount * entriesPerTransition) * sizeof(double);
    MPI_File_set_view(outFile, offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);

    std::vector<double> outbuf;
    if (me == 0) {
      outbuf.reserve(entriesPerTransition * _clusterGraph.numTransitions() + 1);
      int totalTransitionCount = std::accumulate(rbuf.begin(), rbuf.end(), (int) 0);
      outbuf.push_back(ubuf(totalTransitionCount).d);
    } else {
      outbuf.reserve(entriesPerTransition * _clusterGraph.numTransitions());
    }
    for (size_t i = 0; i < _clusterGraph.numTransitions(); ++i) {
      const ClusterTransition &t = _clusterGraph.getTransition(i);
      outbuf.push_back(ubuf(t.cluster1).d);
      outbuf.push_back(ubuf(t.cluster2).d);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) { outbuf.push_back(t.transition.column(i)[j]); }
      }
    }
    MPI_File_write_all(outFile, outbuf.data(), outbuf.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&outFile);
    debugLog(lmp, "End of write_cluster_transitions_parallel() on rank {}\n", me);
  }

  /*++++++++++++++++++++++++++++++++
    _____
   /  __ \                          
   | /  \/ ___  _ __ ___  _ __ ___
   | |    / _ \| '_ ` _ \| '_ ` _ \ 
   | \__/\ (_) | | | | | | | | | | |
    \____/\___/|_| |_| |_|_| |_| |_|
  ++++++++++++++++++++++++++++++++++*/

  void FixDXA::pack_neighborIndices_forward_comm()
  {
    debugLog(lmp, "Start of pack_neighborIndices_forward_comm() on rank {}\n", me);

    _neighborTags.resize(atom->nlocal);
    tagint *atomTags = atom->tag;
    for (size_t i = 0; i < atom->nlocal; ++i) {
      for (size_t j = 0; j < _neighCount; ++j) {
        _neighborTags[i][j] = (_neighborIndices[i][j] != -1) ? atomTags[_neighborIndices[i][j]]
                                                             : -1;
      }
    }
    debugLog(lmp, "End of pack_neighborIndices_forward_comm() on rank {}\n", me);
  }

  void FixDXA::unpack_neighborIndices_forward_comm()
  {
    debugLog(lmp, "Start of unpack_neighborIndices_forward_comm() on rank {}\n", me);

    size_t ntotal = atom->nlocal + atom->nghost;
    assert(_neighborTags.size() == ntotal);

    tagint *atomTags = atom->tag;
    for (size_t i = atom->nlocal; i < ntotal; ++i) {
      const int ni = _neighList->ilist[i];
      const int *jlist = _neighList->firstneigh[ni];
      const int jnum = _neighList->numneigh[ni];

      const Vector3d vstart = xToVector(atom->x[i]);

      for (size_t j = 0; j < _neighCount; ++j) {
        if (_neighborTags[i][j] == -1) {
          assert((j < _neighCount) ? _structureType[i] == OTHER : true);
          continue;
        }

        int bestIndex = -1;
        double minDist = std::numeric_limits<double>::max();
        for (int k = 0; k < jnum; ++k) {
          int jj = jlist[k];
          jj &= NEIGHMASK;
          if (atomTags[jj] == _neighborTags[i][j]) {
            double dist = (vstart - xToVector(atom->x[jj])).lengthSquared();
            if (dist < minDist) {
              minDist = dist;
              bestIndex = jj;
            }
          }
        }
        _neighborIndices[i][j] = bestIndex;
      }
    }
    debugLog(lmp, "End of unpack_neighborIndices_forward_comm() on rank {}\n", me);
  }

  int FixDXA::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
  {
    int m = 0;
    int j;
    switch (_commStep) {
      case STRUCTURE:
        for (int i = 0; i < n; ++i) {
          j = list[i];
          assert(j < _structureType.size());
          buf[m++] = ubuf(static_cast<int>(_structureType[j])).d;
        }
        break;
      case STRUCTURE_NEIGHS:
        for (int i = 0; i < n; ++i) {
          j = list[i];
          assert(j < _structureType.size());
          buf[m++] = ubuf(static_cast<int>(_structureType[j])).d;
        }
        for (int i = 0; i < n; ++i) {
          j = list[i];
          assert(j < _neighborTags.size());
          for (size_t k = 0; k < _neighCount; ++k) { buf[m++] = ubuf(_neighborTags[j][k]).d; }
        }
        break;
      case CLUSTER:
        for (int i = 0; i < n; ++i) {
          j = list[i];
          assert(j < _atomClusterType.size());
          assert(j < _atomSymmetryPermutations.size());
          buf[m++] = ubuf(static_cast<int>(_atomClusterType[j])).d;
          buf[m++] = ubuf(_atomSymmetryPermutations[j]).d;
        }
        break;
      case DISPLACEMENT:
        for (int i = 0; i < n; ++i) {
          j = list[i];
          assert(j < _displacedAtoms.size());
          buf[m++] = _displacedAtoms[j][0];
          buf[m++] = _displacedAtoms[j][1];
          buf[m++] = _displacedAtoms[j][2];
        }
        break;
      default:
        unreachable(lmp);
        break;
        return 0;
    }
    return m;
  }
  void FixDXA::unpack_forward_comm(int n, int first, double *buf)
  {
    int m = 0;
    assert(first >= atom->nlocal);
    switch (_commStep) {
      case STRUCTURE:
        for (int i = first, last = first + n; i < last; ++i) {
          assert(i < _structureType.size());
          _structureType[i] = static_cast<StructureType>(ubuf(buf[m++]).i);
        }
        break;
      case STRUCTURE_NEIGHS:
        for (int i = first, last = first + n; i < last; ++i) {
          assert(i < _structureType.size());
          _structureType[i] = static_cast<StructureType>(ubuf(buf[m++]).i);
        }
        // TODO add offset of atoms->nlocal
        _neighborTags.resize(first + n);
        for (int i = first, last = first + n; i < last; ++i) {
          assert(i < _neighborTags.size());
          for (size_t j = 0; j < _neighCount; ++j) {
            assert(_neighborTags[i][j] == 0);
            _neighborTags[i][j] = static_cast<tagint>(ubuf(buf[m++]).i);
          }
        }
        break;
      case CLUSTER:
        for (int i = first, last = first + n; i < last; ++i) {
          assert(i < _atomClusterType.size());
          assert(i < _atomSymmetryPermutations.size());
          _atomClusterType[i] = static_cast<tagint>(ubuf(buf[m++]).i);
          _atomSymmetryPermutations[i] = ubuf(buf[m++]).i;
        }
        break;
      case DISPLACEMENT:
        for (int i = first, last = first + n; i < last; ++i) {
          assert(i < _displacedAtoms.size());
          _displacedAtoms[i][0] = buf[m++];
          _displacedAtoms[i][1] = buf[m++];
          _displacedAtoms[i][2] = buf[m++];
        }
        break;
      default:
        unreachable(lmp);
        break;
    }
  }

  /*++++++++++++++++++++++++++++++++++++++++++++++++++++
   _____                  _ _       _   _             
  |_   _|                | | |     | | (_)            
    | | ___  ___ ___  ___| | | __ _| |_ _  ___  _ __  
    | |/ _ \/ __/ __|/ _ \ | |/ _` | __| |/ _ \| '_ \ 
    | |  __/\__ \__ \  __/ | | (_| | |_| | (_) | | | |
    \_/\___||___/___/\___|_|_|\__,_|\__|_|\___/|_| |_|
++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

  void FixDXA::write_tessellation_parallel() const
  {
    debugLog(lmp, "Start of write_tessellation_parallel() on rank {}\n", me);

    int nprocs;
    MPI_Comm_size(world, &nprocs);

    assert(_dt.isValid());
    const int sbuf = (int) _dt.numOwnedFacets();
    std::vector<int> rbuf;
    rbuf.resize(nprocs);
    MPI_Allgather(&sbuf, 1, MPI_INT, rbuf.data(), 1, MPI_INT, world);

    const std::string fname = fmt::format("facets.bin.data", me);
    MPI_File outFile;
    MPI_File_open(world, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outFile);

    const size_t headerSize = (me == 0) ? 0 : 1;
    const size_t entryCount = std::accumulate(rbuf.begin(), rbuf.begin() + me, (size_t) 0);
    const size_t entriesPerTransition = 4;
    const MPI_Offset offset = (headerSize + entryCount * entriesPerTransition) * sizeof(double);
    MPI_File_set_view(outFile, offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);

    std::vector<double> outbuf;
    if (me == 0) {
      outbuf.reserve(entriesPerTransition * _dt.numOwnedFacets() + 1);
      int totalFacetCount = std::accumulate(rbuf.begin(), rbuf.end(), (int) 0);
      outbuf.push_back(ubuf(totalFacetCount).d);
    } else {
      outbuf.reserve(entriesPerTransition * _dt.numOwnedFacets());
    }

    for (size_t cell = 0; cell < _dt.numCells(); ++cell) {
      for (size_t facet = 0; facet < 4; ++facet) {
        if (!_dt.facetIsOwned(cell, facet)) { continue; }
        outbuf.push_back(ubuf((tagint) (headerSize + entryCount + cell)).d);
        outbuf.push_back(ubuf(atom->tag[_dt.facetVertex(cell, facet, 0)]).d);
        outbuf.push_back(ubuf(atom->tag[_dt.facetVertex(cell, facet, 1)]).d);
        outbuf.push_back(ubuf(atom->tag[_dt.facetVertex(cell, facet, 2)]).d);
      }
    }
    assert(outbuf.size() == entriesPerTransition * _dt.numOwnedFacets() + (me == 0));
    MPI_File_write_all(outFile, outbuf.data(), outbuf.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&outFile);

    debugLog(lmp, "End of write_tessellation_parallel() on rank {}\n", me);
  }

  void FixDXA::write_per_rank_tessellation() const
  {
    debugLog(lmp, "Start of write_tessellation() on rank {}\n", me);

    const std::string fname = fmt::format("tessellation_on_rank_{}.xyz", me);
    std::ofstream outFile(fname);
    if (!outFile) { error->all(FLERR, "Could not open {} for write.", fname); }
    outFile << "DXA debug topology file\n" << atom->nlocal + atom->nghost << '\n';
    for (size_t i = 0; i < atom->nlocal + atom->nghost; ++i) {
      outFile << fmt::format("{} {} {} {} {}\n", atom->tag[i], (i < atom->nlocal) ? 1 : 2,
                             _displacedAtoms[i][0], _displacedAtoms[i][1], _displacedAtoms[i][2]);
    }
    outFile << _dt.numOwnedFacets() << " triangles\n";
    for (size_t i = 0; i < _dt.numCells(); ++i) {
      for (size_t j = 0; j < 4; ++j) {
        if (!_dt.facetIsOwned(i, j)) { continue; }
        outFile << fmt::format("{} {} {} {}\n", i, _dt.facetVertex(i, j, 0),
                               _dt.facetVertex(i, j, 1), _dt.facetVertex(i, j, 2));
      }
    }

    debugLog(lmp, "End of write_tessellation() on rank {}\n", me);
  }

  void FixDXA::write_per_rank_atoms() const
  {
    debugLog(lmp, "Start of write_atoms() on rank {}\n", me);

    const std::string fname = fmt::format("atoms_on_rank_{}.xyz", me);
    std::ofstream outFile(fname);
    if (!outFile) { error->all(FLERR, "Could not open {} for write.", fname); }
    outFile << atom->nlocal + atom->nghost
            << "\nProperties=id:I:1:atom_types:I:1:cluster:I:1:pos:R:3\n";
    for (size_t i = 0; i < atom->nlocal + atom->nghost; ++i) {
      outFile << fmt::format("{} {} {} {} {} {}\n", atom->tag[i], (i < atom->nlocal) ? 1 : 2,
                             (int) _structureType[i], atom->x[i][0], atom->x[i][1], atom->x[i][2]);
    }
    debugLog(lmp, "End of write_atoms() on rank {}\n", me);
  }

  CellValidity FixDXA::validateSliverCell(int cell)
  {
    CellValidity cellValid = CellValidity::VALID;

    for (size_t facet = 0; facet < 4; ++facet) {
      int oppCell = _dt.oppositeCell(cell, facet);
      if (oppCell == -1) {
        cellValid = CellValidity::INVALID;
        break;
      }
      cellValid = _dt.cellIsValid(oppCell);
      assert(cellValid != CellValidity::SLIVER);
      if (cellValid == CellValidity::INVALID) { break; }
    }
    if (cellValid != CellValidity::INVALID) { cellValid = CellValidity::VALID; }
    _dt.setCellIsValid(cell, cellValid);
    return cellValid;
  }

  CellValidity FixDXA::validateCell(int cell, const std::array<Plane<double>, 6> &bbox)
  {
    CellValidity cellValid = _dt.cellIsValid(cell);
    if (cellValid != CellValidity::UNPROCESSED) { return cellValid; }

    std::array<Vector3d, 4> cellVerts;
    int otherCount = 0;
    for (size_t vert = 0; vert < 4; ++vert) {
      int idx = _dt.cellVertex(cell, vert);
      if (idx == -1) { continue; }
      cellVerts[vert] = _dt.getVertexPos(idx);
      otherCount += _structureType[idx] == OTHER;
    }

    // Deal with defective cells
    if (cell < _dt.numFiniteCells() && otherCount == 4) {
      cellValid = CellValidity::OTHER;
      _dt.setCellIsValid(cell, cellValid);
      return cellValid;
    }
    // Deal with surface cells
    if (cell >= _dt.numFiniteCells() && otherCount == 3) {
      cellValid = CellValidity::SURFACE;
      _dt.setCellIsValid(cell, cellValid);
      return cellValid;
    }
    if (cell > _dt.numFiniteCells()) {
      debugLog(lmp, "vert idx: {} {} {} {}; vert: {} {} {} {}; other cts: {} on rank {}",
               _dt.cellVertex(cell, 0), _dt.cellVertex(cell, 1), _dt.cellVertex(cell, 2),
               _dt.cellVertex(cell, 3), atom->tag[_dt.cellVertex(cell, 0)],
               atom->tag[_dt.cellVertex(cell, 1)], atom->tag[_dt.cellVertex(cell, 2)],
               atom->tag[_dt.cellVertex(cell, 3)], otherCount, me);
    }
    assert(cell < _dt.numFiniteCells());
    Sphere<double> s{cellVerts[0], cellVerts[1], cellVerts[2], cellVerts[3]};
    assert(s.valid() || s.unreliable());

    if (s.unreliable()) {
      cellValid = CellValidity::SLIVER;
      _dt.setCellIsValid(cell, cellValid);
      return cellValid;
    }

    bool isValid = true;
    for (const auto &b : bbox) {
      if (b.isOpenBoundary()) { continue; }
      double distance = b.getSignedPointDistance(s.origin());
      isValid = distance < 0 && std::abs(distance) > s.radius();
      if (!isValid) { break; }
    }
    cellValid = (isValid) ? CellValidity::VALID : CellValidity::INVALID;
    _dt.setCellIsValid(cell, cellValid);
    return cellValid;
  }

  bool FixDXA::validateTessllation()
  {

    // planes of bounding box
    std::array<Plane<double>, 6> bbox;
    // 100, -100, 010, 0-10, 001, 00-1
    // Bounding box of local and ghost atom
    // std::array<double, 3> bboxMin{std::numeric_limits<double>::max(),
    //                               std::numeric_limits<double>::max(),
    //                               std::numeric_limits<double>::max()};
    // std::array<double, 3> bboxMax{std::numeric_limits<double>::min(),
    //                               std::numeric_limits<double>::min(),
    //                               std::numeric_limits<double>::min()};
    // for (size_t i = 0, end = atom->nlocal + atom->nghost; i < end; ++i) {
    //   for (size_t j = 0; j < 3; ++j) {
    //     if (atom->x[i][j] < bboxMin[j]) { bboxMin[j] = atom->x[i][j]; }
    //     if (atom->x[i][j] > bboxMax[j]) { bboxMax[j] = atom->x[i][j]; }
    //   }
    // }
    std::array<double, 3> bboxMin;
    std::array<double, 3> bboxMax;
    {
      double *cutghost = comm->cutghost;
      if (domain->triclinic == 0) {
        bboxMin[0] = domain->sublo[0] - cutghost[0];
        bboxMin[1] = domain->sublo[1] - cutghost[1];
        bboxMin[2] = domain->sublo[2] - cutghost[2];
        bboxMax[0] = domain->subhi[0] + cutghost[0];
        bboxMax[1] = domain->subhi[1] + cutghost[1];
        bboxMax[2] = domain->subhi[2] + cutghost[2];
      } else {
        std::array<double, 3> lo;
        std::array<double, 3> hi;
        lo[0] = domain->sublo_lamda[0] - cutghost[0];
        lo[1] = domain->sublo_lamda[1] - cutghost[1];
        lo[2] = domain->sublo_lamda[2] - cutghost[2];
        hi[0] = domain->subhi_lamda[0] + cutghost[0];
        hi[1] = domain->subhi_lamda[1] + cutghost[1];
        hi[2] = domain->subhi_lamda[2] + cutghost[2];
        domain->bbox(lo.data(), hi.data(), bboxMin.data(), bboxMax.data());
      }
    }

    {    // Subdomain boundaries (0,1) coordiantes
      std::array<double, 3> sublo = {domain->sublo_lamda[0], domain->sublo_lamda[1],
                                     domain->sublo_lamda[2]};
      std::array<double, 3> subhi = {domain->subhi_lamda[0], domain->subhi_lamda[1],
                                     domain->subhi_lamda[2]};
      // pbc flags x,y,z
      std::array<int, 3> periodic = {domain->periodicity[0], domain->periodicity[1],
                                     domain->periodicity[2]};
      // 100, -100, 010, 0-10, 001, 00-1
      // Bounding box of local and ghost atom
      bbox[0].replaceData({bboxMax[0], bboxMin[1], bboxMin[2]},
                          {bboxMax[0], bboxMax[1], bboxMax[2]},
                          {bboxMax[0], bboxMin[1], bboxMax[2]});
      bbox[0].setOpenBoundary(almostEqual(subhi[0], 1.0) && !periodic[0]);
      bbox[1].replaceData({bboxMin[0], bboxMin[1], bboxMin[2]},
                          {bboxMin[0], bboxMin[1], bboxMax[2]},
                          {bboxMin[0], bboxMax[1], bboxMax[2]});
      bbox[1].setOpenBoundary(almostEqual(sublo[0], 0.0) && !periodic[0]);
      bbox[2].replaceData({bboxMax[0], bboxMax[1], bboxMin[2]},
                          {bboxMin[0], bboxMax[1], bboxMin[2]},
                          {bboxMin[0], bboxMax[1], bboxMax[2]});
      bbox[2].setOpenBoundary(almostEqual(subhi[1], 1.0) && !periodic[1]);
      bbox[3].replaceData({bboxMin[0], bboxMin[1], bboxMin[2]},
                          {bboxMax[0], bboxMin[1], bboxMin[2]},
                          {bboxMin[0], bboxMin[1], bboxMax[2]});
      bbox[3].setOpenBoundary(almostEqual(sublo[1], 0.0) && !periodic[1]);
      bbox[4].replaceData({bboxMin[0], bboxMin[1], bboxMax[2]},
                          {bboxMax[0], bboxMin[1], bboxMax[2]},
                          {bboxMax[0], bboxMax[1], bboxMax[2]});
      bbox[4].setOpenBoundary(almostEqual(subhi[2], 1.0) && !periodic[2]);
      bbox[5].replaceData({bboxMin[0], bboxMin[1], bboxMin[2]},
                          {bboxMax[0], bboxMax[1], bboxMin[2]},
                          {bboxMax[0], bboxMin[1], bboxMin[2]});
      bbox[5].setOpenBoundary(almostEqual(sublo[2], 0.0) && !periodic[2]);
    }

    // Cells are crystalline have to be consistent (valid) across domains:
    // - Their circumsphere does not touch the wall of the domain
    // However, cells that are on the surface of the domain do not need to be valid
    // - They have 3 structure type other and 1 infinite vertex (if they are at an open boundary or the opposite face is far away)
    // - They have 4 structure type other corners if they the other side of the surface is still inside the domain
    // -> This should / might handle open boundaries

    for (size_t cell = 0; cell < _dt.numCells(); ++cell) {
      if (!_dt.cellIsRequired(cell)) { continue; }

      CellValidity cellValid = validateCell(cell, bbox);

      if (cellValid == CellValidity::INVALID) {
        lmp->error->all(FLERR, "Delaunay tessellation not correct!\n");
        return false;
      }
    }

    // fix sliver cells
    for (size_t cell = 0; cell < _dt.numCells(); ++cell) {
      if (!_dt.cellIsRequired(cell)) { continue; }
      if (_dt.cellIsValid(cell) != CellValidity::SLIVER) { continue; }

      CellValidity cellValid = validateSliverCell(cell);

      if (cellValid == CellValidity::INVALID) {
        lmp->error->all(FLERR, "Delaunay tessellation not correct!\n");
        return false;
      }
    }

#if 0
    // Infinite cells
    for (size_t cell = _dt.numFiniteCells(); cell < _dt.numCells(); ++cell) {
      if (!_dt.cellIsRequired(cell)) {
        // _dt.setCellIsValid(cell, false);
        continue;
      }
      int otherCount = 0;
      for (size_t vert = 0; vert < 4; ++vert) {
        int idx = _dt.cellVertex(cell, vert);
        if (idx == -1) { continue; }
        // cellVerts[vert] = _dt.getVertexPos(idx);
        otherCount += _structureType[idx] == OTHER;
      }
      if (otherCount == 3) {
        // _dt.setCellIsValid(cell, true);
        continue;
      }
      lmp->error->all(FLERR, "Delaunay tessellation not correct!\n");
    }
#endif
    return true;
  }

  bool FixDXA::firstTessllation()
  {
    debugLog(lmp, "Start of firstTessllation() on rank {}\n", me);
    size_t ntotal = atom->nlocal + atom->nghost;
    _displacedAtoms.resize(ntotal);
    auto rng = std::unique_ptr<RanPark>(new RanPark(lmp, 1323 + me));
    for (size_t i = 0; i < 50; ++i) { rng->uniform(); };

    const double scale = 1e-4 / _maxNeighDistance;
    for (size_t i = 0; i < atom->nlocal; ++i) {
      // TODO: This dispalcement is very large but guarantees succesful comparison with scipy delaunay
      _displacedAtoms[i][0] = scale * (2 * rng->uniform() - 1);
      _displacedAtoms[i][1] = scale * (2 * rng->uniform() - 1);
      _displacedAtoms[i][2] = scale * (2 * rng->uniform() - 1);
    }

    _commStep = DISPLACEMENT;
    comm->forward_comm(this, 3);
    _commStep = NOCOM;

    for (size_t i = 0; i < ntotal; ++i) {
      _displacedAtoms[i][0] += atom->x[i][0];
      _displacedAtoms[i][1] += atom->x[i][1];
      _displacedAtoms[i][2] += atom->x[i][2];
    }

    _dt.generateTessellation(atom->nlocal, atom->nghost, &_displacedAtoms[0][0], atom->tag);

    debugLog(lmp, "End of firstTessllation() on rank {}\n", me);

    return true;
  }

  void FixDXA::buildEdges()
  {
    debugLog(lmp, "Start of buildEdges() on rank {}\n", me);
    size_t a, b;
    size_t idx = 0;
    Edge newEdge;
    // TODO: Find something cache friendly
    std::set<Edge> edgesSet;

    for (size_t cell = 0; cell < _dt.numCells(); ++cell) {
      if (!_dt.cellIsRequired(cell)) { continue; }
      if (_dt.cellIsValid(cell) == CellValidity::SURFACE) { continue; }
      assert(_dt.cellIsValid(cell) != CellValidity::INVALID ||
             _dt.cellIsValid(cell) != CellValidity::UNPROCESSED);
      for (size_t facet = 0; facet < 4; ++facet) {
        // if (!_dt.facetIsOwned(cell, facet)) { continue; }
        for (size_t e = 0; e < 3; ++e) {
          // edge from a -> b
          newEdge.a = _dt.facetVertex(cell, facet, e);
          newEdge.b = _dt.facetVertex(cell, facet, (e + 1) % 3);

          assert(newEdge.a != newEdge.b);
          if (newEdge.a > newEdge.b) { std::swap(newEdge.a, newEdge.b); }
          // TODO: test alternatives to linear search
          // auto pos = std::find(_edges.begin(), _edges.end(), newEdge);
          // if (pos == _edges.end()) { _edges.push_back({newEdge.a, newEdge.b}); }
          edgesSet.insert({newEdge.a, newEdge.b});
        }
      }
    }
    // TODO: is this necessary
    _edges.clear();
    _edges.reserve(edgesSet.size());
    _edges.insert(_edges.begin(), std::make_move_iterator(edgesSet.begin()),
                  std::make_move_iterator(edgesSet.end()));
    assert(std::is_sorted(_edges.begin(), _edges.end()));
    // std::sort(_edges.begin(), _edges.end())
    debugLog(lmp, "End of buildEdges() on rank {}\n", me);
  }

  // This can maybe probably go
  void FixDXA::updateClustersFromNeighbors()
  {
    debugLog(lmp, "Start of updateClustersFromNeighbors() on rank {}\n", me);
    bool done = false;
    do {
      done = true;
      for (const auto &e : _edges) {
        if (e.a < atom->nlocal && _atomClusterType[e.a] == INVALID &&
            _atomClusterType[e.b] != INVALID) {
          _atomClusterType[e.a] = _atomClusterType[e.b];
          // TODO: is this necessary?
          _atomSymmetryPermutations[e.a] = -1;
          // _atomSymmetryPermutations[e.a] = _atomSymmetryPermutations[e.b];
          done = false;
        } else if (e.b < atom->nlocal && _atomClusterType[e.a] != INVALID &&
                   _atomClusterType[e.b] == INVALID) {
          _atomClusterType[e.b] = _atomClusterType[e.a];
          // TODO: is this necessary?
          _atomSymmetryPermutations[e.b] = -1;
          // _atomSymmetryPermutations[e.b] = _atomSymmetryPermutations[e.a];
          done = false;
        }
      }
    } while (!done);

    // TODO: is this necessary?
    _commStep = CLUSTER;
    comm->forward_comm(this, 2);
    _commStep = NOCOM;

    debugLog(lmp, "End of updateClustersFromNeighbors() on rank {}\n", me);
  }

  int FixDXA::findNeighborIndex(size_t centralAtom, size_t neighborAtom) const
  {
    const std::array<int, _maxNeighCount> &nnlist = _neighborIndices[centralAtom];
    for (size_t jj = 0; jj < _neighCount; ++jj) {
      if (nnlist[jj] == neighborAtom) {
        return jj;
      } else if (nnlist[jj] == -1) {
        return -1;
      }
    }
    return -1;
  }

  const Vector3d &FixDXA::neighborVector(size_t centralAtom, size_t neighborIndex) const
  {
    StructureType structureType = _structureType[centralAtom];
    const auto &crystalStructure = _crystalStructures[structureType];
    assert(neighborIndex >= 0 && neighborIndex < crystalStructure.numNeighbors);

    const int symmetryPermutation = _atomSymmetryPermutations[centralAtom];
    assert(symmetryPermutation >= 0 && symmetryPermutation < crystalStructure.permutations.size());

    const auto &permutation = crystalStructure.permutations[symmetryPermutation].permutation;
    return crystalStructure.latticeVectors[permutation[neighborIndex]];
  }

  struct Node {
    size_t atom = 0;
    size_t parent = 0;
    int length = 0;
    Node(size_t atom, size_t parent, int length) : atom{atom}, parent{parent}, length{length} {}
    bool operator==(const Node &other) const { return other.atom == atom; }
    bool operator==(const size_t other) const { return other == atom; }
  };

  // struct NodeHash {
  //   size_t operator()(const Node &node) const
  //   {
  //     return hashCombine(node.atom, hashCombine(node.parent, node.length));
  //   }

  //   // Boost hash combine https://stackoverflow.com/a/2595226
  //   size_t hashCombine(size_t hash, size_t newValue) const
  //   {
  //     return hash ^= std::hash<size_t>{}(newValue) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  //   }
  // };

  // TODO check whether this can be const
  std::pair<Vector3d, int> FixDXA::findPath(const size_t atom1, const size_t atom2,
                                            const char numSteps)
  {
    // debugLog(lmp, "findPath() atoms {} {}\n", atom1, atom2);
    assert(atom1 != atom2);

    bool debugThis = (atom->tag[atom1] == 7008 && atom->tag[atom2] == 13110) ||
        (atom->tag[atom2] == 7008 && atom->tag[atom1] == 13110);

    if (debugThis) {
      debugLog(lmp, "\n{}: {} {}: {} {}\n", me, atom->tag[atom1], atom->tag[atom2],
               (int) _structureType[atom1], (int) _structureType[atom2]);
    }
    const bool validCluster1 = _atomSymmetryPermutations[atom1] != -1;
    const bool validCluster2 = _atomSymmetryPermutations[atom2] != -1;
    if (validCluster1) {
      const int neighborIndex = findNeighborIndex(atom1, atom2);
      if (neighborIndex != -1) {
        const Vector3d &neighVec = neighborVector(atom1, neighborIndex);
        return {neighVec, _atomClusterType[atom1]};
      }
    }
    if (validCluster2) {
      const int neighborIndex = findNeighborIndex(atom2, atom1);
      if (neighborIndex != -1) {
        const Vector3d &neighVec = neighborVector(atom2, neighborIndex);
        return {-1 * neighVec, _atomClusterType[atom2]};
      };
    }

    if (numSteps == 1) { return {{0, 0, 0}, -1}; }

    std::deque<Node> queue;
    std::vector<Node> visited;
    visited.reserve(100);

    // Breadth first path search
    queue.push_front({atom1, (size_t) 0, 0});
    visited.push_back(queue.front());
    // visitedFlag[atom1] = true;
    while (!queue.empty()) {
      const Node &currentNode = queue.front();

      int neighAtom = -1;
      bool transition = false;
      int neighCount = _crystalStructures[_structureType[currentNode.atom]].numNeighbors;
      assert(neighCount <= _neighCount);
      const std::array<int, _maxNeighCount> &nnlist = _neighborIndices[currentNode.atom];

      for (size_t jj = 0; jj < neighCount; ++jj) {
        neighAtom = nnlist[jj];
        if (neighAtom == -1) { continue; }
        // Path should only traverse valid cells
        if (!_dt.vertexIsRequired(neighAtom)) { continue; }

        // if (_atomClusterType[currentNode.atom] == 4412 && _atomClusterType[neighAtom] == 27860) {
        //   auto x = 5;
        //   ;
        // }
        // if (_atomClusterType[currentNode.atom] == 27860 && _atomClusterType[neighAtom] == 4412) {
        //   auto x = 5;
        //   ;
        // }

        if (_atomClusterType[currentNode.atom] == _atomClusterType[neighAtom]) {
          transition = true;
        } else {
          // TODO is the cluster graph bi-directional / do I need to check the inverse transition?
          transition = _clusterGraph.containsTransition(_atomClusterType[currentNode.atom],
                                                        _atomClusterType[neighAtom]);
        }

        // we have found a path
        if (transition && neighAtom == atom2) { break; }

        // if the atomCluster only got filled by the previous propagation of cluster -> skip
        if (_atomSymmetryPermutations[neighAtom] == -1) { continue; }

        Node neighNode{(size_t) neighAtom, currentNode.atom, currentNode.length + 1};
        // neighbor has not been visited
        if (transition && neighNode.length < numSteps &&
            std::find(visited.begin(), visited.end(), neighNode) == visited.end()) {
          // if (transition && neighNode.length < numSteps && !visitedFlag[neighNode.atom]) {
          queue.push_back(neighNode);
          visited.push_back(neighNode);
          // visitedFlag[neighNode.atom] = true;
        }
      }

      // we have found a path
      if (transition && neighAtom == atom2) {
        // neighIdx -> child
        // current -> parent

        // QUESTION: neighVec currentNode cluster?
        size_t neighIndex = findNeighborIndex(currentNode.atom, neighAtom);
        Vector3d neighVec = neighborVector(currentNode.atom, neighIndex);

        // debugLog(lmp, "Path from {} ({}) -> {} ({})\n", atom1, _atomClusterType[atom1], atom2,
        //          _atomClusterType[atom2]);
        // debugLog(lmp, "{} ({}) <- {} ({}) <- ", neighAtom, _atomClusterType[neighAtom],
        //          currentNode.atom, _atomClusterType[currentNode.atom]);

        const Node *child = &currentNode;
        auto pos = std::find(visited.begin(), visited.end(), child->parent);
        assert(pos != visited.end());
        const Node *parent = &(*pos);

        // debugLog(lmp, "{} ({}) <- {} ({}) <- ", child->atom, _atomClusterType[child->atom],
        //          parent->atom, _atomClusterType[parent->atom]);

        size_t exitVar = 0;
        while (true) {
          if (_atomClusterType[child->atom] == _atomClusterType[parent->atom]) {
            neighIndex = findNeighborIndex(parent->atom, child->atom);
            neighVec += neighborVector(parent->atom, neighIndex);
          } else {
            neighIndex = findNeighborIndex(parent->atom, child->atom);
            const Vector3d &step = neighborVector(parent->atom, neighIndex);
            neighVec = _clusterGraph.applyReverseTransition(
                _atomClusterType[parent->atom], _atomClusterType[child->atom], neighVec);
            neighVec += step;
          }
          child = parent;
          if (child->atom == atom1) { break; }
          pos = std::find(visited.begin(), visited.end(), child->parent);
          assert(pos != visited.end());
          parent = &(*pos);

          // debugLog(lmp, "{} ({}) <- {} ({}) <- ", child->atom, _atomClusterType[child->atom],
          //          parent->atom, _atomClusterType[parent->atom]);
          if (exitVar++ == 2 * numSteps) { unreachable(lmp); }
        };
        // debugLog(lmp, "\n");

        return {neighVec, _atomClusterType[atom1]};
      }

      queue.pop_front();
    }

    // no path was found
    if (debugThis) { debugLog(lmp, "\n{}: Path not found\n", me); }
    return {{0, 0, 0}, -1};
  }

  void FixDXA::assignIdealLatticeVectorsToEdges()
  {
    debugLog(lmp, "Start of assignIdealLatticeVectorsToEdges() on rank {}\n", me);
    constexpr char numSteps = 4;
    _edgeVectors.clear();
    _edgeVectors.resize(_edges.size());

    for (size_t edgeIdx = 0; edgeIdx < _edges.size(); ++edgeIdx) {
      const Edge &edge = _edges[edgeIdx];

      bool debugThis = (atom->tag[edge.a] == 13113 && atom->tag[edge.b] == 13114) ||
          (atom->tag[edge.b] == 13114 && atom->tag[edge.a] == 13113) ||
          (atom->tag[edge.a] == 13112 && atom->tag[edge.b] == 13113) ||
          (atom->tag[edge.b] == 13113 && atom->tag[edge.a] == 13112) ||
          (atom->tag[edge.a] == 13112 && atom->tag[edge.b] == 13114) ||
          (atom->tag[edge.b] == 13114 && atom->tag[edge.a] == 13112);
      if (debugThis) {
        auto a = 1;
        auto b = 2;
      }

      size_t cluster1Id = _atomClusterType[edge.a];
      size_t cluster2Id = _atomClusterType[edge.b];
      assert(cluster1Id != INVALID && cluster2Id != INVALID);

      // TODO: at the moment cluster information is not! trasnferred to other processors
      // therefore cluster.orientation might contain only zeros
      if (cluster1Id == INVALID || cluster2Id == INVALID) { continue; }

      int idealCluster;
      Vector3d idealVector;
      if (_structureType[edge.a] == OTHER) {
        std::tie(idealVector, idealCluster) = findPath(edge.b, edge.a, 4);
        idealVector = -1 * idealVector;
      } else {
        std::tie(idealVector, idealCluster) = findPath(edge.a, edge.b, 4);
      }
      if (debugThis) {
        debugLog(lmp, "\n{}: {}->{}: {} {}: {} {}: {}: {} {} {}\n", me, atom->tag[edge.a],
                 atom->tag[edge.b], (int) _structureType[edge.a], (int) _structureType[edge.b],
                 cluster1Id, cluster2Id, idealCluster, idealVector[0], idealVector[1],
                 idealVector[2]);
      }
      if (idealCluster == -1) { continue; }

      if (cluster1Id == cluster2Id && cluster1Id == idealCluster) {
        _edgeVectors[edgeIdx].vector = idealVector;
        _edgeVectors[edgeIdx].transition1 = cluster1Id;
        _edgeVectors[edgeIdx].transition2 = cluster2Id;
        continue;
      }

      // todo simplify control flow
      if (idealCluster == cluster2Id) {
        // if they belong to different clusters we have to check whether there is a transition
        size_t transitionIndex = _clusterGraph.determineClusterTransition(idealCluster, cluster1Id);
        // no transition
        if (transitionIndex == std::numeric_limits<size_t>::max()) {
          continue;
          ;
        }
        idealVector = _clusterGraph.applyTransition(idealCluster, cluster1Id, idealVector);
      }

      size_t transitionIndex = _clusterGraph.determineClusterTransition(cluster1Id, cluster2Id);
      if (transitionIndex == std::numeric_limits<size_t>::max()) { continue; }

      _edgeVectors[edgeIdx].vector = idealVector;
      _edgeVectors[edgeIdx].transition1 = cluster1Id;
      _edgeVectors[edgeIdx].transition2 = cluster2Id;
    }
    debugLog(lmp, "End of assignIdealLatticeVectorsToEdges() on rank {}\n", me);
  }

  void FixDXA::write_per_rank_edges() const
  {
    debugLog(lmp, "Start of write_per_rank_edges() on rank {}\n", me);

    const std::string fname = fmt::format("edges_on_rank_{}.xyz", me);
    std::ofstream outFile(fname);
    if (!outFile) { error->all(FLERR, "Could not open {} for write.", fname); }
    outFile << "DXA debug topology file\n" << atom->nlocal + atom->nghost << '\n';
    for (size_t i = 0; i < atom->nlocal + atom->nghost; ++i) {
      outFile << fmt::format("{} {} {} {} {}\n", atom->tag[i], (i < atom->nlocal) ? 1 : 2,
                             _displacedAtoms[i][0], _displacedAtoms[i][1], _displacedAtoms[i][2]);
    }
    outFile << _edges.size() << " edges\n";
    for (size_t i = 0; i < _edges.size(); ++i) {
      outFile << fmt::format("{} {} {} {} {} {} {}\n", _edges[i].a, _edges[i].b,
                             _edgeVectors[i].vector[0], _edgeVectors[i].vector[1],
                             _edgeVectors[i].vector[2], _edgeVectors[i].transition1,
                             _edgeVectors[i].transition2);
    }
    outFile << _clusterGraph.numClusters() << " clusters\n";
    for (size_t i = 0; i < _clusterGraph.numClusters(); ++i) {
      const Cluster &cluster = _clusterGraph.getCluster(i);
      outFile << fmt::format("{} {} {} {} {} {} {} {} {} {}\n", cluster.id,
                             cluster.orientation.column(0)[0], cluster.orientation.column(0)[1],
                             cluster.orientation.column(0)[2], cluster.orientation.column(1)[0],
                             cluster.orientation.column(1)[1], cluster.orientation.column(1)[2],
                             cluster.orientation.column(2)[0], cluster.orientation.column(2)[1],
                             cluster.orientation.column(2)[2]);
    }
    debugLog(lmp, "End of write_per_rank_edges() on rank {}\n", me);
  }

  bool FixDXA::classifyElasticCompatible(size_t cell) const
  {
    if (!_dt.cellIsFinite(cell)) { return false; }

    // store the 6 edges of the tetrahedron
    static constexpr std::array<std::array<int, 2>, 6> edgeVertexOrder{
        {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    static constexpr std::array<std::array<int, 3>, 4> facetLoops{
        {{0, 4, 2}, {1, 5, 2}, {0, 3, 1}, {3, 5, 4}}};

    std::array<EdgeVector, 6> edgeVectors;
    for (int edgeIndex = 0; edgeIndex < 6; edgeIndex++) {
      size_t v1 = _dt.cellVertex(cell, edgeVertexOrder[edgeIndex][0]);
      size_t v2 = _dt.cellVertex(cell, edgeVertexOrder[edgeIndex][1]);

      Edge edge{v1, v2};
      auto pos = std::lower_bound(_edges.begin(), _edges.end(), edge);
      if (*pos == edge) {
        size_t idx = std::distance(_edges.begin(), pos);
        const EdgeVector &edgeVec = _edgeVectors[idx];
        if (edgeVec.transition1 == 0 || edgeVec.transition2 == 0) { return false; }
        edgeVectors[edgeIndex] = edgeVec;
        continue;
      }

      Edge reverseEdge{v2, v1};
      pos = std::lower_bound(_edges.begin(), _edges.end(), reverseEdge);
      if (*pos == reverseEdge) {
        size_t idx = std::distance(_edges.begin(), pos);
        const EdgeVector &edgeVec = _edgeVectors[idx];
        if (edgeVec.transition1 == 0 || edgeVec.transition2 == 0) { return false; }
        if (edgeVec.transition1 == edgeVec.transition2) {
          edgeVectors[edgeIndex].vector = -1 * edgeVec.vector;
          edgeVectors[edgeIndex].transition1 = edgeVec.transition1;
          edgeVectors[edgeIndex].transition2 = edgeVec.transition2;
        } else {
          edgeVectors[edgeIndex].vector = _clusterGraph.applyTransition(
              edgeVec.transition1, edgeVec.transition2, -1 * edgeVec.vector);
          edgeVectors[edgeIndex].transition1 = edgeVec.transition2;
          edgeVectors[edgeIndex].transition2 = edgeVec.transition1;
        }
        continue;
      }
      // edge was not found (neither fwd or bwd)
      unreachable(lmp);
      return false;
    }    // end of for loop

    for (size_t facet = 0; facet < 4; ++facet) {
      const EdgeVector &v1 = edgeVectors[facetLoops[facet][0]];
      Vector3d bVector = v1.vector;

      const EdgeVector &v2 = edgeVectors[facetLoops[facet][1]];
      bVector += _clusterGraph.applyReverseTransition(v1.transition1, v1.transition2, v2.vector);
      // bVector += _clusterGraph.applyReverseTransition(v2.transition1, v2.transition2, v2.vector);

      const EdgeVector &v3 = edgeVectors[facetLoops[facet][2]];
      bVector -= v3.vector;
      if (!bVector.equals(0, LATTICE_VECTOR_EPSILON)) { return false; }
    }

    // Disclination test
    for (size_t facet = 0; facet < 4; ++facet) {
      const EdgeVector &e1 = edgeVectors[facetLoops[facet][0]];
      const EdgeVector &e2 = edgeVectors[facetLoops[facet][1]];
      const EdgeVector &e3 = edgeVectors[facetLoops[facet][2]];
      if ((e1.transition1 != e1.transition2) || (e2.transition1 != e2.transition2) ||
          (e3.transition1 != e3.transition2)) {

        Matrix3d frankRotation =
            _clusterGraph.getReverseTransitionMatrix(e3.transition1, e3.transition2) *
            _clusterGraph.getTransitionMatrix(e2.transition1, e2.transition2) *
            _clusterGraph.getTransitionMatrix(e1.transition1, e1.transition2);
        if (!frankRotation.equals(Matrix3d::Identity(), TRANSITION_MATRIX_EPSILON)) {
          return false;
        }
      }
    }
    return true;
  }

  template <class ForwardIterator>
  static ForwardIterator mostCommonElement(ForwardIterator begin, ForwardIterator end)
  {

    ForwardIterator it = begin;
    ForwardIterator mostCommon = begin;
    size_t count = 0;
    size_t mostCommonCount = 0;

    for (; begin < end; ++begin) {
      if (*it == *begin) {
        count++;
      } else {
        it = begin;
        count = 1;
      }
      if (count > mostCommonCount) {
        mostCommonCount = count;
        mostCommon = it;
      }
    }
    return mostCommon;
  };

  int FixDXA::classifyCell(size_t cell) const
  {
    if (classifyElasticCompatible(cell)) {
      std::array<int, 4> clusters;
      for (int i = 0; i < 4; i++) { clusters[i] = _atomClusterType[_dt.cellVertex(cell, i)]; }
      std::sort(clusters.begin(), clusters.end());
      return *mostCommonElement(clusters.cbegin(), clusters.cend());
    }
    return -1;
  };

  void FixDXA::classifyRegions()
  {
    _regions.clear();
    _regions.resize(_dt.numCells(), -3);
    double alpha = 5 * _maxNeighDistance;
    for (size_t cell = 0; cell < _dt.numCells(); ++cell) {
      if (!_dt.cellIsRequired(cell)) { continue; }
      assert(_dt.cellIsValid(cell) != CellValidity::INVALID);

      bool isFilled = false;
      if (_dt.cellIsFinite(cell)) {
        Delaunay::AlphaTestResult alphaTestResult = _dt.alphaTest(cell, alpha);
        switch (alphaTestResult) {
          case Delaunay::AlphaTestResult::INSIDE: {
            isFilled = true;
            break;
          }
          case Delaunay::AlphaTestResult::OUTSIDE: {
            break;
          }
          case Delaunay::AlphaTestResult::UNRELIABLE: {
            // for unrealiable alpha test results we compare against our four neighbors. If any neighbor is not filled
            // the current cell is set to not filled.
            size_t facet = 0;
            for (; facet < 4; ++facet) {
              int oppositeCell = _dt.oppositeCell(cell, facet);
              assert(oppositeCell != -1);
              if (!_dt.cellIsFinite(oppositeCell)) { break; }
              Delaunay::AlphaTestResult oppAlphaTestResult = _dt.alphaTest(oppositeCell, alpha);
              if (oppAlphaTestResult == Delaunay::AlphaTestResult::OUTSIDE) { break; }
            }
            isFilled = facet == 4;
            break;
          }
          default: {
            unreachable(lmp);
            break;
          }
        }    // end of switch
      }      // end of if

      if (isFilled) {
        _regions[cell] = classifyCell(cell);
      } else {
        _regions[cell] = -2;
      }
    }
  }

  template <typename T> static std::vector<size_t> argsort(const std::vector<T> &arr)
  {
    std::vector<size_t> idx(arr.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&arr](size_t left, size_t right) {
      return arr[left] < arr[right];
    });
    return idx;
  }

  void FixDXA::constructMesh() const
  {
#if 1
    assert(_regions.size() == _dt.numCells());
    std::vector<int> remappedIdx;
    remappedIdx.resize(atom->nlocal + atom->nghost, -1);
    std::vector<int> triangles;
    std::vector<int> triangleRegions;
    // Todo: good heuristic
    size_t prealloc = std::count_if(_structureType.begin(), _structureType.begin() + atom->nlocal,
                                    [](StructureType s) {
                                      return s == OTHER;
                                    });
    triangles.reserve(prealloc);

    // DynamicDisjointSet<tagint> transitionsDS = _clusterGraph.getDynamicDisjointSet();
    DisjointSet<tagint> transitionsDS = _clusterGraph.getDisjointSet();

    int vertexCount = 0;
    for (size_t cell = 0; cell < _dt.numCells(); ++cell) {
      if (!_dt.cellIsRequired(cell)) { continue; }
      int rcell = _regions[cell];
      for (int facet = 0; facet < 4; ++facet) {
        if (!_dt.facetIsOwned(cell, facet)) { continue; }
        assert(rcell != -3);
        int oppositeCell = _dt.oppositeCell(cell, facet);
        assert(oppositeCell >= 0);
        assert(oppositeCell < _regions.size());
        int rocell = _regions[oppositeCell];
        assert(rocell != -3);

        if (rcell == rocell) { continue; }
        if ((rcell == -1 && rocell == -2) || (rcell == -2 && rocell == -1)) { continue; }
        if (transitionsDS.find(rcell) == transitionsDS.find(rocell)) { continue; }

        // triangles.push_back(transitionsDS.find(rcell));
        triangles.push_back(transitionsDS.find(rcell));
        triangles.push_back(transitionsDS.find(rocell));

        for (int vert = 0; vert < 3; ++vert) {
          int vertexIndex = _dt.facetVertex(cell, facet, vert);
          if (remappedIdx[vertexIndex] == -1) { remappedIdx[vertexIndex] = vertexCount++; }
          triangles.push_back(remappedIdx[vertexIndex]);
        }
      }
    }
    assert(triangles.size() % 5 == 0);

    const std::vector<size_t> order = argsort(remappedIdx);

    // assert(remappedIdx[order[order.size() - vertexCount - 1]] == -1);
    // assert(vertexCount > 0 && remappedIdx[order[order.size() - vertexCount]] != -1);
    const std::string fname = fmt::format("triangles_on_rank_{}.xyz", me);
    std::ofstream outFile(fname);
    if (!outFile) { error->all(FLERR, "Could not open {} for write.", fname); }
    outFile << "DXA debug triangle file\n" << vertexCount << '\n';
    for (size_t i = order.size() - vertexCount; i < order.size(); ++i) {
      int idx = order[i];    // remappedIdx[order[i]];
      outFile << fmt::format("{} {} {}\n", atom->x[idx][0], atom->x[idx][1], atom->x[idx][2]);
    }
    outFile << triangles.size() / 5 << " triangles\n";
    for (size_t i = 0; i < triangles.size(); i += 5) {
      outFile << fmt::format("{} {} {} {} {}\n", triangles[i], triangles[i + 1], triangles[i + 2],
                             triangles[i + 3], triangles[i + 4]);
    }
#else
    DynamicDisjointSet<tagint> transitionsDS = _clusterGraph.getDynamicDisjointSet();

    const std::string fname = fmt::format("triangles_on_rank_{}.xyz", me);
    std::ofstream outFile(fname);
    if (!outFile) { error->all(FLERR, "Could not open {} for write.", fname); }
    outFile << "DXA debug triangle file\n" << atom->nlocal + atom->nghost << '\n';
    for (size_t i = 0; i < atom->nlocal + atom->nghost; ++i) {
      outFile << fmt::format("{} {} {} {} {}\n", atom->tag[i], (i < atom->nlocal) ? 1 : 2,
                             atom->x[i][0], atom->x[i][1], atom->x[i][2]);
    }
    outFile << _dt.numOwnedCells() << " cells\n";

    for (size_t cell = 0; cell < _dt.numCells(); ++cell) {
      if (!_dt.cellIsOwned(cell)) { continue; }
      outFile << transitionsDS.find(_regions[cell]) << ' ';
      for (size_t facet = 0; facet < 4; ++facet) {
        outFile << fmt::format("{} {} {} ", _dt.facetVertex(cell, facet, 0),
                               _dt.facetVertex(cell, facet, 1), _dt.facetVertex(cell, facet, 2));
      }
      outFile << '\n';
    }

#endif
  }

}    // namespace FIXDXA_NS
}    // namespace LAMMPS_NS