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

namespace LAMMPS_NS {
namespace FIXDXA_NS {
  static constexpr double EPSILON = 1e-12;
  static constexpr double TRANSITION_MATRIX_EPSILON = 1e-4;

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // VECTOR3
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  template <typename T> class Vector3 : public std::array<T, 3> {
   public:
    Vector3() = default;
    Vector3(T x, T y, T z) : std::array<T, 3>{{x, y, z}} {};

    constexpr inline T x() const { return (*this)[0]; }
    constexpr inline T y() const { return (*this)[1]; }
    constexpr inline T z() const { return (*this)[2]; }

    T lengthSquared() const { return x() * x() + y() * y() + z() * z(); }
    T length() const { return sqrt(lengthSquared()); }

    bool isZero(T eps)
    {
      for (auto v : *this) {
        if (std::abs(v) >= eps) { return false; }
      }
      return true;
    };

    constexpr inline Vector3 cross(const Vector3 &v) const
    {
      return Vector3(y() * v.z() - z() * v.y(), z() * v.x() - x() * v.z(),
                     x() * v.y() - y() * v.x());
    }

    constexpr inline bool equals(const Vector3 &v, T eps) const
    {
      return (std::abs(x() - v.x()) < eps) && std::abs(y() - v.y()) < eps &&
          std::abs(z() - v.z()) < eps;
    }

    constexpr Vector3 operator-(const Vector3 &v) const
    {
      return Vector3(x() - v.x(), y() - v.y(), z() - v.z());
    }
    constexpr Vector3 operator+(const Vector3 &v) const
    {
      return Vector3(x() + v.x(), y() + v.y(), z() + v.z());
    }
    friend Vector3 operator*(double d, const Vector3 &v)
    {
      return Vector3(d * v[0], d * v[1], d * v[2]);
    }
    constexpr Vector3 operator-() const { return Vector3(-x(), -y(), -z()); }
  };
  using Vector3d = Vector3<double>;

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  //MATRIX3
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  template <typename T> class Matrix3 : public std::array<Vector3<T>, 3> {
   public:
    Matrix3() = default;
    Matrix3(T e00, T e01, T e02, T e10, T e11, T e12, T e20, T e21, T e22) :
        std::array<Vector3<T>, 3>{
            {Vector3<T>(e00, e10, e20), Vector3<T>(e01, e11, e21), Vector3<T>(e02, e12, e22)}} {};

    static constexpr Matrix3 Identity()
    {
      return Matrix3((T) 1, (T) 0, (T) 0, (T) 0, (T) 1, (T) 0, (T) 0, (T) 0, (T) 1);
    };
    static constexpr Matrix3 Zero()
    {
      return Matrix3((T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0);
    };

    constexpr T operator()(size_t row, size_t col) const { return (*this)[col][row]; }
    T &operator()(size_t row, size_t col) { return (*this)[col][row]; }

    constexpr const Vector3<T> &operator()(size_t col) const { return (*this)[col]; }
    Vector3<T> &column(size_t col) { return (*this)[col]; }
    constexpr const Vector3<T> &column(size_t col) const { return (*this)[col]; }

    friend constexpr Vector3<T> operator*(const Matrix3<T> &m, const Vector3<T> &v)
    {
      return Vector3<T>(m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2],
                        m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2],
                        m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2]);
    }

    friend constexpr Matrix3 operator*(const Matrix3<T> &a, const Matrix3<T> &b)
    {
      return Matrix3(a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0) + a(0, 2) * b(2, 0),
                     a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1) + a(0, 2) * b(2, 1),
                     a(0, 0) * b(0, 2) + a(0, 1) * b(1, 2) + a(0, 2) * b(2, 2),
                     a(1, 0) * b(0, 0) + a(1, 1) * b(1, 0) + a(1, 2) * b(2, 0),
                     a(1, 0) * b(0, 1) + a(1, 1) * b(1, 1) + a(1, 2) * b(2, 1),
                     a(1, 0) * b(0, 2) + a(1, 1) * b(1, 2) + a(1, 2) * b(2, 2),
                     a(2, 0) * b(0, 0) + a(2, 1) * b(1, 0) + a(2, 2) * b(2, 0),
                     a(2, 0) * b(0, 1) + a(2, 1) * b(1, 1) + a(2, 2) * b(2, 1),
                     a(2, 0) * b(0, 2) + a(2, 1) * b(1, 2) + a(2, 2) * b(2, 2));
    }

    bool equals(const Matrix3<T> &m, T eps) const
    {
      for (size_t i = 0; i < 3; i++) {
        if (!((*this).column(i).equals(m.column(i), eps))) { return false; }
      }
      return true;
    }

    void setZero()
    {
      for (size_t col = 0; col < 3; col++) { (*this)[col] = {0, 0, 0}; }
    }

    constexpr T determinant() const
    {
      return (((*this)[0][0] * (*this)[1][1] - (*this)[0][1] * (*this)[1][0]) * ((*this)[2][2]) -
              ((*this)[0][0] * (*this)[1][2] - (*this)[0][2] * (*this)[1][0]) * ((*this)[2][1]) +
              ((*this)[0][1] * (*this)[1][2] - (*this)[0][2] * (*this)[1][1]) * ((*this)[2][0]));
    }
    constexpr bool isOrthogonal(T eps) const
    {
      return (std::abs((*this)[0][0] * (*this)[1][0] + (*this)[0][1] * (*this)[1][1] +
                       (*this)[0][2] * (*this)[1][2]) <= eps) &&
          (std::abs((*this)[0][0] * (*this)[2][0] + (*this)[0][1] * (*this)[2][1] +
                    (*this)[0][2] * (*this)[2][2]) <= eps) &&
          (std::abs((*this)[1][0] * (*this)[2][0] + (*this)[1][1] * (*this)[2][1] +
                    (*this)[1][2] * (*this)[2][2]) <= eps) &&
          (std::abs((*this)[0][0] * (*this)[0][0] + (*this)[0][1] * (*this)[0][1] +
                    (*this)[0][2] * (*this)[0][2] - T(1)) <= eps) &&
          (std::abs((*this)[1][0] * (*this)[1][0] + (*this)[1][1] * (*this)[1][1] +
                    (*this)[1][2] * (*this)[1][2] - T(1)) <= eps) &&
          (std::abs((*this)[2][0] * (*this)[2][0] + (*this)[2][1] * (*this)[2][1] +
                    (*this)[2][2] * (*this)[2][2] - T(1)) <= eps);
    }

    Matrix3 inverse() const
    {
      T det = determinant();
      assert(std::abs(det) >= EPSILON);
      return Matrix3<T>(((*this)[1][1] * (*this)[2][2] - (*this)[1][2] * (*this)[2][1]) / det,
                        ((*this)[2][0] * (*this)[1][2] - (*this)[1][0] * (*this)[2][2]) / det,
                        ((*this)[1][0] * (*this)[2][1] - (*this)[1][1] * (*this)[2][0]) / det,
                        ((*this)[2][1] * (*this)[0][2] - (*this)[0][1] * (*this)[2][2]) / det,
                        ((*this)[0][0] * (*this)[2][2] - (*this)[2][0] * (*this)[0][2]) / det,
                        ((*this)[0][1] * (*this)[2][0] - (*this)[0][0] * (*this)[2][1]) / det,
                        ((*this)[0][1] * (*this)[1][2] - (*this)[1][1] * (*this)[0][2]) / det,
                        ((*this)[0][2] * (*this)[1][0] - (*this)[0][0] * (*this)[1][2]) / det,
                        ((*this)[0][0] * (*this)[1][1] - (*this)[1][0] * (*this)[0][1]) / det);
    }

    bool inverse(Matrix3<T> &outMat) const
    {
      T det = determinant();
      if (std::abs(det) <= EPSILON) { return false; }
      outMat = Matrix3<T>(((*this)[1][1] * (*this)[2][2] - (*this)[1][2] * (*this)[2][1]) / det,
                          ((*this)[2][0] * (*this)[1][2] - (*this)[1][0] * (*this)[2][2]) / det,
                          ((*this)[1][0] * (*this)[2][1] - (*this)[1][1] * (*this)[2][0]) / det,
                          ((*this)[2][1] * (*this)[0][2] - (*this)[0][1] * (*this)[2][2]) / det,
                          ((*this)[0][0] * (*this)[2][2] - (*this)[2][0] * (*this)[0][2]) / det,
                          ((*this)[0][1] * (*this)[2][0] - (*this)[0][0] * (*this)[2][1]) / det,
                          ((*this)[0][1] * (*this)[1][2] - (*this)[1][1] * (*this)[0][2]) / det,
                          ((*this)[0][2] * (*this)[1][0] - (*this)[0][0] * (*this)[1][2]) / det,
                          ((*this)[0][0] * (*this)[1][1] - (*this)[1][0] * (*this)[0][1]) / det);
      return true;
    }
  };
  using Matrix3d = Matrix3<double>;

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
  // STRUCTURETYPE
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  enum StructureType { BCC = 0, CUBIC_DIA, FCC, HCP, HEX_DIA, OTHER, MAXSTRUCTURECOUNT };

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
    Matrix3d orientation = Matrix3d::Identity();
    Cluster(tagint id, StructureType structure) : id{id}, structure{structure} {}
  };
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // CLUSTERTRANSITION
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  struct ClusterTransition {
    size_t cluster1;
    size_t cluster2;
    Matrix3d transition = Matrix3d::Identity();

    bool operator==(const ClusterTransition &rhs) const
    {
      return (cluster1 == rhs.cluster1) && (cluster2 == rhs.cluster2) &&
          (transition.equals(rhs.transition, EPSILON));
    }
  };
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // CLUSTERGRAPH
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  class ClusterGraph {
   public:
    ClusterGraph() = default;
    explicit ClusterGraph(size_t n) { reserve(n); }

    int addCluster(tagint id, StructureType structureType)
    {
      _clusterVector.emplace_back(id, structureType);
      return _clusterVector.size() - 1;
    }
    void reserve(size_t n)
    {
      _clusterVector.reserve(n);
      _clusterTransitions.reserve(2 * n);
    }

   private:
    std::vector<Cluster> _clusterVector;
    std::vector<ClusterTransition> _clusterTransitions;
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

    int pack_exchange(int, double *) override;

    int unpack_exchange(int, double *) override;

    const unsigned char VERSION = 1;

   private:
    bool getCNANeighbors(std::vector<CNANeighbor> &, const int, const int) const;

    void identifyCrystalStructure();

    void initializeStructures();

    void buildNNList();
    void buildClusters();

    static constexpr size_t _maxNeighCount = 16;
    static constexpr size_t _minNarg = 5;
    const StructureType _inputStructure;
    const int _neighCount;

    // std::vector<tagint> _nnListIdx;
    std::vector<int> _nnList;

    std::vector<std::array<tagint, _maxNeighCount>> _neighborIndices;
    std::vector<StructureType> _structureType;
    std::vector<int> _atomSymmetryPermutations;
    std::vector<tagint> _atomClusterType;
    ClusterGraph _clusterGraph;

    double **_output = nullptr;
    const std::string _outputName = "dxa:Output";

    class NeighList *_neighList = nullptr;
    static std::array<CrystalStructure<_maxNeighCount>, MAXSTRUCTURECOUNT> _crystalStructures;
  };
}    // namespace FIXDXA_NS
}    // namespace LAMMPS_NS

#endif
#endif
