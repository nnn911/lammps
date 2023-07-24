////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright 2022 OVITO GmbH, Germany
//
//  This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY KIND,
//  either express or implied. See the GPL or the MIT License for the specific language
//  governing rights and limitations.
//
////////////////////////////////////////////////////////////////////////////////////////

#ifdef FIX_CLASS
// clang-format off
FixStyle(dxa,FIXDXA_NS::FixDXA);
// clang-format on
#else
#ifndef LMP_FIX_DXA_H
#define LMP_FIX_DXA_H

#include "fix.h"
#include "fix_dxa_delaunay.h"
#include "fix_dxa_math.h"
#include <memory>

namespace LAMMPS_NS {
namespace FIXDXA_NS {

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // NEIGHBORBONDARRAY
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  template <size_t size> class NeighborBondArray {
   public:
    NeighborBondArray() { reset(); };

    void reset() { _data.fill(0); };

    void setNeighborBond(int n1, int n2)
    {
      _data[n1] |= 1 << n2;
      _data[n2] |= 1 << n1;
    };

    void setNeighborBond(int n1, int n2, bool set)
    {
      if (set)
        setNeighborBond(n1, n2);
      else {
        _data[n1] &= ~(1 << n2);
        _data[n2] &= ~(1 << n1);
      }
    };

    bool areNeighbors(int n1, int n2) const { return _data[n1] & 1 << n2; }

    int countCommonNeighbors(int n, uint32_t &commonNeighbors) const
    {
      commonNeighbors = _data[n];
      return countCommonNeighbors(n);
    }

    int countCommonNeighbors(int n) const
    {
#if defined(__has_builtin) && __has_builtin(__builtin_popcount)
      return __builtin_popcount(_data[n]);
#elif __clang__ || __GNUC__
      return __builtin_popcount(_data[n]);
#elif _MSC_VER
      return __popcnt(_data[n]);
#else
      // Count the number of bits set in neighbor bit-field.
      // https://graphics.stanford.edu/%7Eseander/bithacks.html#CountBitsSetParallel
      unsigned int v = _data[n] - ((_data[n] >> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
      return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif
    }

    int findNeighborBonds(int n, std::array<uint32_t, size * size> &neighborBonds, int nn) const
    {
      int numBonds = 0;
      unsigned int nib[32];
      int nibn = 0;
      unsigned int ni1b = 1;
      for (int ni1 = 0; ni1 < nn; ni1++, ni1b <<= 1) {
        if (_data[n] & ni1b) {
          unsigned int b = _data[n] & _data[ni1];
          for (int n = 0; n < nibn; n++) {
            if (b & nib[n]) { neighborBonds[numBonds++] = ni1b | nib[n]; }
          }
          nib[nibn++] = ni1b;
        }
      }
      return numBonds;
    }

    // refactor this to work with arrays!
    static int getAdjacentBonds(unsigned int atom, unsigned int *bondsToProcess, int &numBonds,
                                unsigned int &atomsToProcess, unsigned int &atomsProcessed)
    {
      int adjacentBonds = 0;
      for (int b = numBonds - 1; b >= 0; b--) {
        if (atom & *bondsToProcess) {
          ++adjacentBonds;
          atomsToProcess |= *bondsToProcess & (~atomsProcessed);
          memmove(bondsToProcess, bondsToProcess + 1, sizeof(bondsToProcess[0]) * b);
          numBonds--;
        } else
          ++bondsToProcess;
      }
      return adjacentBonds;
    }
    static int calcMaxChainLength(int numBonds, std::array<uint32_t, size * size> &neighborBonds)
    {
      // Group the common bonds into clusters.
      int maxChainLength = 0;
      while (numBonds--) {
        // Make a new cluster starting with the first remaining bond to be processed.
        unsigned int atomsToProcess = neighborBonds[numBonds];
        unsigned int atomsProcessed = 0;
        int clusterSize = 1;
        do {
          // Determine the number of trailing 0-bits in atomsToProcess, starting at the least significant bit position.
#if defined(__has_builtin) && __has_builtin(__builtin_ctz)
          int nextAtomIndex = __builtin_ctz(atomsToProcess);
#elif __clang__ || __GNUC__
          int nextAtomIndex = __builtin_ctz(atomsToProcess);
#elif _MSC_VER
          unsigned long nextAtomIndex;
          _BitScanForward(&nextAtomIndex, atomsToProcess);
#else
#error "__builtin_ctz or _BitScanForward" required!
#endif
          unsigned int nextAtom = 1 << nextAtomIndex;
          atomsProcessed |= nextAtom;
          atomsToProcess &= ~nextAtom;
          clusterSize += getAdjacentBonds(nextAtom, neighborBonds.data(), numBonds, atomsToProcess,
                                          atomsProcessed);
        } while (atomsToProcess);
        if (clusterSize > maxChainLength) maxChainLength = clusterSize;
      }
      return maxChainLength;
    }

   private:
    std::array<uint32_t, size> _data;
  };

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ENUMS
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // TODO -> MAKE THESE ENUM CLASSES
  enum StructureType : int { BCC = 0, CUBIC_DIA, FCC, HCP, HEX_DIA, OTHER, MAXSTRUCTURECOUNT };
  enum ClusterStatus : tagint { INVALID = -1 };
  enum CommSteps { STRUCTURE, STRUCTURE_NEIGHS, CLUSTER, DISPLACEMENT, NOCOM };

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // SYMMETRYPERMUTATION
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  template <size_t size> struct SymmetryPermutation {
    Matrix3d transformation;
    std::array<int, size> permutation;
    std::vector<int> product;
    std::vector<int> inverseProduct;
  };

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // CRYSTALSTRUCTURE
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  template <size_t size> struct CrystalStructure {
    //Coordination
    int numNeighbors;
    NeighborBondArray<size> neighborArray;
    std::array<int, size> cnaSignatures;
    std::array<std::array<int, 2>, size> commonNeighbors;

    // Lattice
    std::vector<Vector3<double>> latticeVectors;
    Matrix3d primitiveCell;
    Matrix3d primitiveCellInverse;

    // Symmetry
    std::vector<SymmetryPermutation<size>> permutations;
  };

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // CNANeighbor
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  struct CNANeighbor {
    Vector3<double> xyz = {0, 0, 0};
    double lengthSq = 0;
    int idx = -1;
    int neighIdx = -1;

    bool operator<(const CNANeighbor &o) const { return lengthSq < o.lengthSq; }
    bool operator>(const CNANeighbor &o) const { return lengthSq > o.lengthSq; }
  };

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // CLUSTER
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  struct Cluster {
    tagint id;
    StructureType structure;
    size_t symmetryPermutationIndex = 0;
    Matrix3d orientation = Matrix3d::Identity();
    Cluster(tagint id, StructureType structure) : id{id}, structure{structure} {}
  };

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // CLUSTERTRANSITION
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  struct ClusterTransition {
    tagint cluster1;
    tagint cluster2;
    Matrix3d transition;

    ClusterTransition(tagint cluster1, tagint cluster2, const Matrix3d &transition) :
        cluster1{cluster1}, cluster2{cluster2}, transition{transition}
    {
    }

    bool operator==(const ClusterTransition &rhs) const
    {
      return (cluster1 == rhs.cluster1) && (cluster2 == rhs.cluster2) &&
          (transition.equals(rhs.transition, TRANSITION_MATRIX_EPSILON));
    }
  };

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // CLUSTERGRAPH
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  class ClusterGraph {
   public:
    ClusterGraph() = default;
    explicit ClusterGraph(size_t n) { reserve(n); }

    void reserve(size_t n)
    {
      _clusters.reserve(n);
      _transitions.reserve(2 * n);
    }

    int addCluster(tagint id, StructureType structureType)
    {
      _clusters.emplace_back(id, structureType);
      return _clusters.size() - 1;
    }

    int addClusterTransition(tagint cluster1, tagint cluster2,
                             const Matrix3d &transition = Matrix3d::Identity())
    {
      _transitions.emplace_back(cluster1, cluster2, transition);
      _transitions.emplace_back(cluster2, cluster1, transition.inverse());
      _disconnected.clear();
      return _clusters.size() - 2;
    }

    bool clustersAreDisconnected(tagint cluster1Id, tagint cluster2Id) const
    {
      return std::find(_disconnected.begin(), _disconnected.end(),
                       std::pair<tagint, tagint>{cluster1Id, cluster2Id}) != _disconnected.end();
    }

    struct Node {
      tagint cluster;
      tagint parent;
      int length;
      Node(tagint cluster, tagint parent, int length) :
          cluster{cluster}, parent{parent}, length{length}
      {
      }
      bool operator==(const Node &other) const { return other.cluster == cluster; }
      bool operator==(const tagint &other) const { return other == cluster; }
    };
    // generates a cluster transition from 2 distant clusters in the cluster graph
    // breadth first search of the cluster graph
    size_t determineClusterTransition(tagint cluster1Id, tagint cluster2Id)
    {
      // if (!containsCluster(cluster1Id)) {
      //   std::cerr << fmt::format("Requested cluster {} \n", cluster1Id);
      // }
      // if (!containsCluster(cluster2Id)) {
      //   std::cerr << fmt::format("Requested cluster {} \n", cluster2Id);
      // }

      // assert(containsCluster(cluster1Id));
      // assert(containsCluster(cluster2Id));

      // Check if transition already exists
      {
        int idx = findTransition(cluster1Id, cluster2Id);
        if (idx < _transitions.size()) { return idx; }
      }
      // Check if clusters are already labeled as disconnected
      {
        if (clustersAreDisconnected(cluster1Id, cluster2Id)) {
          return std::numeric_limits<size_t>::max();
        }
      }

      // try connecting the clusters
      constexpr int numSteps = 4;

      std::deque<Node> queue;
      std::vector<Node> visited;
      visited.reserve(std::min((size_t) 100, numClusters()));
      queue.emplace_back(cluster1Id, 0, 0);
      visited.emplace_back(queue.front());

      Node neighNode{0, 0, 0};
      while (!queue.empty()) {
        const Node &currentNode = queue.front();
        const tagint cluster = currentNode.cluster;

        // TODO search sorted
        for (const auto &t : _transitions) {
          if (t.cluster1 != cluster) { continue; }

          neighNode.cluster = t.cluster2;
          neighNode.parent = currentNode.cluster;
          neighNode.length = currentNode.length + 1;

          // we found a path
          if (t.cluster2 == cluster2Id) { break; }

          if (neighNode.length < numSteps &&
              std::find(visited.begin(), visited.end(), neighNode) == visited.end()) {
            queue.push_back(neighNode);
            visited.push_back(neighNode);
          }
        }

        // retrace path
        if (neighNode.cluster == cluster2Id) {
          const Node *parent = nullptr;
          const Node *child = &neighNode;
          Matrix3d transition = Matrix3d::Identity();
          do {
            auto pos = std::find(visited.begin(), visited.end(), child->parent);

            assert(pos != visited.end());
            parent = &(*pos);
            transition = transition * getTransitionMatrix(parent->cluster, child->cluster);
            child = parent;
          } while (parent->cluster != cluster1Id);
          return addClusterTransition(cluster1Id, cluster2Id, transition);
        }

        queue.pop_front();
      }

      // no transition was found
      _disconnected.emplace_back(cluster1Id, cluster2Id);
      _disconnected.emplace_back(cluster2Id, cluster1Id);
      return std::numeric_limits<size_t>::max();
    }

    [[nodiscard]] size_t numClusters() const { return _clusters.size(); }
    [[nodiscard]] size_t numTransitions() const { return _transitions.size(); }

    const Cluster &getCluster(size_t index) const
    {
      assert(index < _clusters.size());
      return _clusters[index];
    }

    void setClusterOrientation(size_t index, const Matrix3d &orientation)
    {
      assert(index < _clusters.size());
      _clusters[index].orientation = orientation;
    }

    bool containsCluster(tagint clusterId) const
    {
      return findCluster(clusterId) < _clusters.size();
    }

    size_t findCluster(tagint clusterId) const
    {
      for (size_t i = 0; i < _clusters.size(); ++i) {
        if (_clusters[i].id == clusterId) { return i; }
      }
      return _clusters.size();
    }
    bool containsTransition(tagint clusterId1, tagint clusterId2) const
    {
      if (clusterId1 == clusterId2) { return true; }
      return findTransition(clusterId1, clusterId2) < _transitions.size();
    }

    size_t findTransition(tagint clusterId1, tagint clusterId2) const
    {
      for (size_t i = 0; i < _transitions.size(); ++i) {
        if ((_transitions[i].cluster1 == clusterId1) && (_transitions[i].cluster2 == clusterId2)) {
          return i;
        }
      }
      return _transitions.size();
    }

    const ClusterTransition &getTransition(size_t index) const
    {
      assert(index < _transitions.size());
      return _transitions[index];
    }

    // TODO: make this return a reference
    // -> make identity matrix static or member if this!
    Matrix3d getTransitionMatrix(tagint clusterId1, tagint clusterId2) const
    {
      if (clusterId1 == clusterId2) { return Matrix3d::Identity(); }
      const size_t pos = findTransition(clusterId1, clusterId2);
      return _transitions[pos].transition;
    }
    // TODO: make this return a reference
    // -> make identity matrix static or member if this!
    Matrix3d getReverseTransitionMatrix(tagint clusterId1, tagint clusterId2) const
    {
      if (clusterId1 == clusterId2) { return Matrix3d::Identity(); }
      const size_t pos = findTransition(clusterId2, clusterId1);
      return _transitions[pos].transition;
    }

    Vector3d applyTransition(tagint clusterId1, tagint clusterId2, const Vector3d &vector) const
    {
      if (clusterId1 == clusterId2) { return vector; }
      size_t transition = findTransition(clusterId1, clusterId2);
      if (transition >= _transitions.size()) {
        std::cerr << fmt::format("ERROR: {} {}\n", clusterId1, clusterId2);
      }

      assert(transition < _transitions.size());
      return _transitions[transition].transition * vector;
    }

    Vector3d applyReverseTransition(tagint clusterId1, tagint clusterId2,
                                    const Vector3d &vector) const
    {
      return applyTransition(clusterId2, clusterId1, vector);
    }

    // TODO: DynamicDisjointSet vs DisjointSet
    DynamicDisjointSet<tagint> getDynamicDisjointSet() const
    {
      DynamicDisjointSet<tagint> ds;
      for (const auto &t : _transitions) { ds.unite(t.cluster1, t.cluster2); }
      return ds;
    }

    DisjointSet<tagint> getDisjointSet() const
    {
      tagint maxID = 0;
      for (const auto &t : _transitions) {
        if (t.cluster1 > maxID) { maxID = t.cluster1; }
      }
      DisjointSet<tagint> ds(maxID);
      for (const auto &t : _transitions) { ds.unite(t.cluster1, t.cluster2); }
      return ds;
    }

   private:
    std::vector<Cluster> _clusters;
    std::vector<ClusterTransition> _transitions;
    std::vector<std::pair<tagint, tagint>> _disconnected;
  };

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // FIXDXA
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  class FixDXA : public Fix {
   public:
    FixDXA(class LAMMPS *, int, char **);
    ~FixDXA();

    void init() override;
    void init_list(int, NeighList *) override;
    void setup(int) override;

    int setmask() override;
    void end_of_step() override;

    double memory_usage() override;

    void grow_arrays(int) override;

    void copy_arrays(int, int, int) override;

    void set_arrays(int) override;

    // int pack_exchange(int, double *) override;

    // int unpack_exchange(int, double *) overr/ide;

    int pack_forward_comm(int, int *, double *, int, int *) override;
    void unpack_forward_comm(int, int, double *) override;

    const unsigned char VERSION = 1;

   private:
    // Structure Identification
    void initialize_neighborIndices(size_t);
    void pack_neighborIndices_forward_comm();
    void unpack_neighborIndices_forward_comm();

    bool addNeighborIndex(int, int);
    bool getCNANeighbors(std::vector<CNANeighbor> &, const int, const int) const;
    void identifyCrystalStructure();
    void initializeStructures();

    void buildNNList(int, int);
    void buildClusters();
    void buildClustersPostTess();
    void connectClusters();
    void connectClustersPostTess();

    int findNeighborIndex(size_t, size_t) const;
    const Vector3d &neighborVector(size_t, size_t) const;
    double getSqNeighDistance(int, int);

    void write_cluster_transitions() const;
    void write_cluster_transitions_parallel() const;

    void updateClustersFromNeighbors();

    // Tessllation
    bool firstTessllation();
    bool validateTessllation();
    void write_tessellation_parallel() const;
    void write_per_rank_tessellation() const;
    void write_per_rank_edges() const;
    void buildEdges();
    void assignIdealLatticeVectorsToEdges();
    std::pair<Vector3d, int> findPath(const size_t, const size_t, const char);

    // Check dislocation cells
    bool classifyElasticCompatible(size_t) const;
    int classifyCell(size_t) const;
    void classifyRegions();
    void constructMesh() const;

   private:
    static constexpr size_t _maxNeighCount = 16;
    static constexpr size_t _minNarg = 5;
    const StructureType _inputStructure;
    const int _neighCount;

    int me;

    CommSteps _commStep = NOCOM;

    // std::vector<tagint> _nnListIdx;
    std::array<int, _maxNeighCount> _nnList;
    std::vector<std::pair<int, double>> _nnListBuffer;
    std::vector<std::array<tagint, _maxNeighCount>> _neighborIndices;
    decltype(_neighborIndices) _neighborTags;
    double _maxNeighDistance = -1;

    std::vector<StructureType> _structureType;
    std::vector<int> _atomSymmetryPermutations;
    std::vector<tagint> _atomClusterType;
    ClusterGraph _clusterGraph;

    std::vector<Vector3d> _displacedAtoms;
    // Todo: _regions and _atomSymmetryPermutations probably not needed at the same time
    // could be merged
    // >=0 -> region value
    // -1 -> elastic incompatible
    // -2 -> cell is not filled
    // -3 -> cell is not requred
    std::vector<int> _regions;

    // std::unique_ptr<Delaunay> _dt = nullptr;
    Delaunay _dt;

    double **_output = nullptr;
    const std::string _outputName = "dxa:Output";

    class NeighList *_neighList = nullptr;
    static std::array<CrystalStructure<_maxNeighCount>, MAXSTRUCTURECOUNT> _crystalStructures;

    // Half edge structure
    struct Edge {
      size_t a;
      size_t b;

      bool operator==(const Edge &other) const { return (a == other.a && b == other.b); }
      bool operator<(const Edge &other) const
      {
        return (a < other.a) || (a == other.a && b < other.b);
      }
    };
    std::vector<Edge> _edges;

    struct EdgeVector {
      Vector3d vector{0, 0, 0};
      size_t transition1 = 0;
      size_t transition2 = 0;
    };

    std::vector<EdgeVector> _edgeVectors;
  };
}    // namespace FIXDXA_NS
}    // namespace LAMMPS_NS

#endif
#endif
