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

// todo remove
#include <fmt/ranges.h>

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

    utils::logmesg(lmp, "End of FixDXA() on rank {}\n", me);
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
    buildClusters();
    for (int i = 0; i < atom->nlocal; ++i) {
      _output[i][0] = static_cast<double>(static_cast<int>(_structureType[i]));
      _output[i][1] = static_cast<double>(_atomClusterType[i]);
    }
    array_atom = _output;
    connectClusters();
#ifndef NDEBUG
    write_cluster_transitions();
#endif
    write_cluster_transitions_parallel();

    // Tessellation
    firstTessllation();
    write_tessellation_parallel();
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
    utils::logmesg(lmp, "Fix DXA version {}\n", VERSION);
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
    double **x = atom->x;
    const int inum = _neighList->inum;
    assert(_neighList->inum == atom->nlocal);
    const int gnum = _neighList->gnum;
    assert(_neighList->gnum == atom->nghost);
    const int nmax = inum + gnum;

    _nnList.resize(numNeigh);

    std::vector<std::pair<int, double>> _nnListBuffer;

    const int i = _neighList->ilist[ii];
    const int *jlist = _neighList->firstneigh[i];
    const int jnum = _neighList->numneigh[i];
    assert((ii < inum) ? jnum > numNeigh : true);

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
  bool FixDXA::getCNANeighbors(std::vector<CNANeighbor> &neighborVectors, const int index,
                               const int nn) const
  {
    double **x = atom->x;
    assert(nn <= _nnList.size());
    neighborVectors.resize(nn);
    for (int i = 0; i < nn; ++i) {
      CNANeighbor &neigh = neighborVectors[i];
      assert(_nnList[i] != -1);
      for (int k = 0; k < 3; ++k) { neigh.xyz[k] = x[_nnList[i]][k] - x[index][k]; }
      neigh.lengthSq = neigh.xyz.lengthSquared();
      neigh.idx = index;
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
    utils::logmesg(lmp, "Start of identifyCrystalStructure() on rank {}\n", me);
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
    utils::logmesg(lmp, "Rank {}:\n", me);
    for (int i = 0; i < summary.size(); ++i) {
      utils::logmesg(lmp, "\nstructure {}: {} / {}", i, summary[i], atom->nlocal);
    }
    utils::logmesg(lmp, "\n");

    pack_neighborIndices_forward_comm();
    _commStep = STRUCTURE_NEIGHS;
    comm->forward_comm(this, _neighCount + 1);
    _commStep = NOCOM;
    unpack_neighborIndices_forward_comm();

    utils::logmesg(lmp, "End of identifyCrystalStructure() on rank {}\n", me);
  }

  static inline Vector3d xToVector(double *x)
  {
    return {x[0], x[1], x[2]};
  }

  void FixDXA::buildClusters()
  {
    utils::logmesg(lmp, "Start of buildClusters() on rank {}\n", me);

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

      _atomClusterType[ii] = atomTags[ii];
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
      }
      assert(std::abs(orientationV.determinant()) > EPSILON);
      _clusterGraph.clusters[clusterIndex].orientation = orientationW * orientationV.inverse();
    }

    // Reorient atoms to align clusters with global coordinate system.
    // for (size_t ii = 0; ii < atom->nlocal; ++ii) {
    //   int clusterId = _atomClusterType[ii];
    //   if (clusterId == INVALID) continue;
    //   int clusterIndex = _clusterGraph.findCluster(clusterId);
    //   assert(clusterIndex < _clusterGraph.clusters.size());
    //   Cluster &cluster = _clusterGraph.clusters[clusterIndex];

    //   const auto &crystalStructure = _crystalStructures[cluster.structure];
    //   int oldSymmetryPermutation = _atomSymmetryPermutations[ii];
    //   int newSymmetryPermutation = crystalStructure.permutations[oldSymmetryPermutation]
    //                                    .inverseProduct[cluster->symmetryTransformation];
    //   _atomSymmetryPermutations[atomIndex] = newSymmetryPermutation;
    // }
    // !TODO! -> ASK ALEX ABOUT THE REORIENT -> StructureAnalysis.cpp :909-947:

    _commStep = CLUSTER;
    comm->forward_comm(this, 2);
    _commStep = NOCOM;

    utils::logmesg(lmp, "End of buildClusters() on rank {}\n", me);
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
    utils::logmesg(lmp, "Start of connectClusters() on rank {}\n", me);
    tagint *atomTags = atom->tag;

    for (int currentAtom = 0; currentAtom < atom->nlocal; ++currentAtom) {

      auto atomTag = atomTags[currentAtom];

      // Cluster of current atom
      if (_atomClusterType[currentAtom] == INVALID) { continue; }
      size_t cluster1Index = _clusterGraph.findCluster(_atomClusterType[currentAtom]);
      assert(cluster1Index < _clusterGraph.clusters.size());

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
        {
          size_t clusterTransition = _clusterGraph.findClusterTransition(
              _atomClusterType[currentAtom], _atomClusterType[neighIdx]);
          if (clusterTransition != _clusterGraph.clusterTransitions.size()) { continue; }
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

    utils::logmesg(lmp, "Clusters on rank {}: {}\n", me, _clusterGraph.clusters.size());
    utils::logmesg(lmp, "Clusters transitions on rank {}: {}\n", me,
                   _clusterGraph.clusterTransitions.size());

    utils::logmesg(lmp, "End of connectClusters() on rank {}\n", me);
  }

  void FixDXA::write_cluster_transitions() const
  {
    utils::logmesg(lmp, "Start of write_cluster_transitions() on rank {}\n", me);

    const std::string fname = fmt::format("cluster_transition_rank_{}.data", me);
    std::ofstream outFile(fname);
    if (!outFile) { error->all(FLERR, "Could not open {} for write.", fname); }
    outFile << _clusterGraph.clusterTransitions.size() << '\n';
    for (const auto &t : _clusterGraph.clusterTransitions) {
      outFile << fmt::format(
          "{} {} {} {} {} {} {} {} {} {} {}\n", t.cluster1, t.cluster2, t.transition.column(0)[0],
          t.transition.column(1)[0], t.transition.column(2)[0], t.transition.column(0)[1],
          t.transition.column(1)[1], t.transition.column(2)[1], t.transition.column(0)[2],
          t.transition.column(1)[2], t.transition.column(2)[2]);
    }
    utils::logmesg(lmp, "End of write_cluster_transitions() on rank {}\n", me);
  }

  void FixDXA::write_cluster_transitions_parallel() const
  {
    utils::logmesg(lmp, "Start of write_cluster_transitions_parallel() on rank {}\n", me);

    int nprocs;
    MPI_Comm_size(world, &nprocs);

    const int sbuf = _clusterGraph.clusterTransitions.size();
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
      outbuf.reserve(entriesPerTransition * _clusterGraph.clusterTransitions.size() + 1);
      int totalTransitionCount = std::accumulate(rbuf.begin(), rbuf.end(), (int) 0);
      outbuf.push_back(ubuf(totalTransitionCount).d);
    } else {
      outbuf.reserve(entriesPerTransition * _clusterGraph.clusterTransitions.size());
    }
    for (const auto &t : _clusterGraph.clusterTransitions) {
      outbuf.push_back(ubuf(t.cluster1).d);
      outbuf.push_back(ubuf(t.cluster2).d);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) { outbuf.push_back(t.transition.column(i)[j]); }
      }
    }
    MPI_File_write_all(outFile, outbuf.data(), outbuf.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&outFile);
    utils::logmesg(lmp, "End of write_cluster_transitions_parallel() on rank {}\n", me);
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
    utils::logmesg(lmp, "Start of pack_neighborIndices_forward_comm() on rank {}\n", me);

    _neighborTags.resize(atom->nlocal);
    tagint *atomTags = atom->tag;
    for (size_t i = 0; i < atom->nlocal; ++i) {
      for (size_t j = 0; j < _neighCount; ++j) {
        _neighborTags[i][j] = (_neighborIndices[i][j] != -1) ? atomTags[_neighborIndices[i][j]]
                                                             : -1;
      }
    }
    utils::logmesg(lmp, "End of pack_neighborIndices_forward_comm() on rank {}\n", me);
  }

  void FixDXA::unpack_neighborIndices_forward_comm()
  {
    utils::logmesg(lmp, "Start of unpack_neighborIndices_forward_comm() on rank {}\n", me);

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
    utils::logmesg(lmp, "End of unpack_neighborIndices_forward_comm() on rank {}\n", me);
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
    utils::logmesg(lmp, "Start of write_tessellation_parallel() on rank {}\n", me);

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

    for (size_t cell = 0; cell < _dt.numFiniteCells(); ++cell) {
      // if (!_dt.cellIsOwned(cell)) { continue; }
      for (size_t facet = 0; facet < 4; ++facet) {
        if (!_dt.facetIsOwned(cell, facet)) { continue; }
        auto i1 = _dt.facetVertex(cell, facet, 0);
        auto i2 = _dt.facetVertex(cell, facet, 1);
        auto i3 = _dt.facetVertex(cell, facet, 2);
        auto c1 = atom->tag[_dt.facetVertex(cell, facet, 0)];
        auto c2 = atom->tag[_dt.facetVertex(cell, facet, 1)];
        auto c3 = atom->tag[_dt.facetVertex(cell, facet, 2)];
        if ((i1 == i2) || (i1 == i3) || (i2 == i3)) {
          auto x = 10;
          ;
        }
        if ((c1 == c2) || (c1 == c3) || (c2 == c3)) {
          auto x = 10;
          ;
        }
        // assert(std::abs(atom->x[_dt.facetVertex(cell, facet, 2)][0] -
        //                 _dt.getVertexPos(_dt.facetVertex(cell, facet, 2))[0]) < 1e-4);
        // assert(std::abs(atom->x[_dt.facetVertex(cell, facet, 2)][1] -
        //                 _dt.getVertexPos(_dt.facetVertex(cell, facet, 2))[1]) < 1e-4);
        // assert(std::abs(atom->x[_dt.facetVertex(cell, facet, 2)][2] -
        //                 _dt.getVertexPos(_dt.facetVertex(cell, facet, 2))[2]) < 1e-4);
        // if ((c1 == 170 || c1 == 176 || c1 == 174) && (c2 == 170 || c2 == 176 || c2 == 174) &&
        //     (c3 == 170 || c3 == 176 || c3 == 174)) {
        //   auto x = 1;
        // }
        // if ((c1 == 170 || c1 == 172 || c1 == 174) && (c2 == 170 || c2 == 172 || c2 == 174) &&
        //     (c3 == 170 || c3 == 172 || c3 == 174)) {
        //   auto x = 1;
        // }
        if (headerSize + entryCount + cell == 78) {
          auto x = 1;
          ;
        }
        outbuf.push_back(ubuf((tagint) (headerSize + entryCount + cell)).d);
        outbuf.push_back(ubuf(atom->tag[_dt.facetVertex(cell, facet, 0)]).d);
        outbuf.push_back(ubuf(atom->tag[_dt.facetVertex(cell, facet, 1)]).d);
        outbuf.push_back(ubuf(atom->tag[_dt.facetVertex(cell, facet, 2)]).d);
      }
    }
    MPI_File_write_all(outFile, outbuf.data(), outbuf.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&outFile);

    utils::logmesg(lmp, "End of write_tessellation_parallel() on rank {}\n", me);
  }

  // void FixDXA::write_tessellation_parallel() const
  // {
  //   utils::logmesg(lmp, "Start of write_tessellation_parallel() on rank {}\n", me);

  //   int nprocs;
  //   MPI_Comm_size(world, &nprocs);

  //   assert(_dt.isValid());
  //   const int sbuf = (int) _dt.numOwnedFacets();
  //   std::vector<int> rbuf;
  //   rbuf.resize(nprocs);
  //   MPI_Allgather(&sbuf, 1, MPI_INT, rbuf.data(), 1, MPI_INT, world);

  //   const std::string fname = fmt::format("facets.bin.data", me);
  //   MPI_File outFile;
  //   MPI_File_open(world, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outFile);

  //   const size_t headerSize = (me == 0) ? 0 : 1;
  //   const size_t entryCount = std::accumulate(rbuf.begin(), rbuf.begin() + me, (size_t) 0);
  //   const size_t entriesPerTransition = 4;
  //   const MPI_Offset offset = (headerSize + entryCount * entriesPerTransition) * sizeof(double);
  //   MPI_File_set_view(outFile, offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);

  //   std::vector<double> outbuf;
  //   if (me == 0) {
  //     outbuf.reserve(entriesPerTransition * _dt.numOwnedFacets() + 1);
  //     int totalFacetCount = std::accumulate(rbuf.begin(), rbuf.end(), (int) 0);
  //     outbuf.push_back(ubuf(totalFacetCount).d);
  //   } else {
  //     outbuf.reserve(entriesPerTransition * _dt.numOwnedFacets());
  //   }

  //   for (size_t cell = 0; cell < _dt.numFiniteCells(); ++cell) {
  //     // if (!_dt.cellIsOwned(cell)) { continue; }
  //     for (size_t facet = 0; facet < 4; ++facet) {

  //       outbuf.push_back(ubuf((tagint) (headerSize + entryCount + cell)).d);
  //       outbuf.push_back(ubuf(atom->tag[_dt.facetVertex(cell, facet, 0)]).d);
  //       outbuf.push_back(ubuf(atom->tag[_dt.facetVertex(cell, facet, 1)]).d);
  //       outbuf.push_back(ubuf(atom->tag[_dt.facetVertex(cell, facet, 2)]).d);
  //     }
  //   }
  //   MPI_File_write_all(outFile, outbuf.data(), outbuf.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
  //   MPI_File_close(&outFile);

  //   utils::logmesg(lmp, "End of write_tessellation_parallel() on rank {}\n", me);
  // }

  bool FixDXA::validateTessllation() const
  {

    // planes of bounding box
    std::array<Plane<double>, 6> bbox;
    // 100, -100, 010, 0-10, 001, 00-1
    // Bounding box of local and ghost atom
    std::array<double, 3> bboxMin{std::numeric_limits<double>::max(),
                                  std::numeric_limits<double>::max(),
                                  std::numeric_limits<double>::max()};
    std::array<double, 3> bboxMax{std::numeric_limits<double>::min(),
                                  std::numeric_limits<double>::min(),
                                  std::numeric_limits<double>::min()};
    for (size_t i = 0, end = atom->nlocal + atom->nghost; i < end; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        if (atom->x[i][j] < bboxMin[j]) { bboxMin[j] = atom->x[i][j]; }
        if (atom->x[i][j] > bboxMax[j]) { bboxMax[j] = atom->x[i][j]; }
      }
    }
    bbox[0].replaceData({bboxMax[0], bboxMin[1], bboxMin[2]}, {bboxMax[0], bboxMax[1], bboxMax[2]},
                        {bboxMax[0], bboxMin[1], bboxMax[2]});
    bbox[1].replaceData({bboxMin[0], bboxMin[1], bboxMin[2]}, {bboxMin[0], bboxMin[1], bboxMax[2]},
                        {bboxMin[0], bboxMax[1], bboxMax[2]});
    bbox[2].replaceData({bboxMax[0], bboxMax[1], bboxMin[2]}, {bboxMin[0], bboxMax[1], bboxMin[2]},
                        {bboxMin[0], bboxMax[1], bboxMax[2]});
    bbox[3].replaceData({bboxMin[0], bboxMin[1], bboxMin[2]}, {bboxMax[0], bboxMin[1], bboxMin[2]},
                        {bboxMin[0], bboxMin[1], bboxMax[2]});
    bbox[4].replaceData({bboxMin[0], bboxMin[1], bboxMax[2]}, {bboxMax[0], bboxMin[1], bboxMax[2]},
                        {bboxMax[0], bboxMax[1], bboxMax[2]});
    bbox[5].replaceData({bboxMin[0], bboxMin[1], bboxMin[2]}, {bboxMax[0], bboxMax[1], bboxMin[2]},
                        {bboxMax[0], bboxMin[1], bboxMin[2]});
    std::array<Vector3d, 4> cellVerts;
    for (size_t cell = 0; cell < _dt.numFiniteCells(); ++cell) {
      if (!_dt.cellIsOwned(cell)) { continue; }
      for (size_t vert = 0; vert < 4; ++vert) {
        int idx = _dt.cellVertex(cell, vert);
        cellVerts[vert] = _dt.getVertexPos(idx);
        assert(almostEqual(cellVerts[vert][0], _displacedAtoms[idx][0]));
        assert(almostEqual(cellVerts[vert][1], _displacedAtoms[idx][1]));
        assert(almostEqual(cellVerts[vert][2], _displacedAtoms[idx][2]));
      }
      Sphere<double> s{cellVerts[0], cellVerts[1], cellVerts[2], cellVerts[3]};
      assert(s.valid());
      for (const auto &b : bbox) {
        double distance = b.getSignedPointDistance(s.origin());
        if (distance > 0 || std::abs(distance) <= s.radius()) {
          lmp->error->all(FLERR, "Delaunay tessellation not correct!\n");
          return false;
        }
      }
    }
    return true;
  }

  bool FixDXA::firstTessllation()
  {
    utils::logmesg(lmp, "Start of firstTessllation() on rank {}\n", me);
    size_t ntotal = atom->nlocal + atom->nghost;
    _displacedAtoms.resize(ntotal);
    auto rng = std::unique_ptr<RanPark>(new RanPark(lmp, 1323 + me));
    for (size_t i = 0; i < 50; ++i) { rng->uniform(); };
    for (size_t i = 0; i < atom->nlocal; ++i) {
      _displacedAtoms[i][0] = 1e-6 * (2 * rng->uniform() - 1);
      _displacedAtoms[i][1] = 1e-6 * (2 * rng->uniform() - 1);
      _displacedAtoms[i][2] = 1e-6 * (2 * rng->uniform() - 1);
    }

    _commStep = DISPLACEMENT;
    comm->forward_comm(this, 3);
    _commStep = NOCOM;

    for (size_t i = 0; i < ntotal; ++i) {
      _displacedAtoms[i][0] += atom->x[i][0];
      _displacedAtoms[i][1] += atom->x[i][1];
      _displacedAtoms[i][2] += atom->x[i][2];
    }

    _dt.generateTessellation(atom->nlocal, atom->nghost, &_displacedAtoms[0][0]);

    // validateTessllation();

    utils::logmesg(lmp, "End of firstTessllation() on rank {}\n", me);

    return true;
  }

}    // namespace FIXDXA_NS
}    // namespace LAMMPS_NS