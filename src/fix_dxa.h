
#ifdef FIX_CLASS
// clang-format off
FixStyle(dxa,FixDXA);
// clang-format on
#else
#ifndef LMP_FIX_DXA_H
#define LMP_FIX_DXA_H

#include "fix.h"

namespace LAMMPS_NS {
template <typename T> class Vector3 {
 public:
  Vector3(){};
  Vector3(T x, T y, T z) : _data{x, y, z} {};

  T x() const { return _data[0]; }
  T y() const { return _data[1]; }
  T z() const { return _data[2]; }
  T operator[](size_t index) const { return _data[index]; }
  T &operator[](size_t index) { return _data[index]; }

  T lengthSquared() const { return x() * x() + y() * y() + z() * z(); }
  T length() const { return sqrt(lengthSquared()); }
  bool isZero(T eps)
  {
    for (auto v : _data) {
      if (std::abs(v) >= eps) { return false; }
    }
    return true;
  };
  Vector3 operator-(const Vector3 &v) const
  {
    return Vector3(x() - v.x(), y() - v.y(), z() - v.z());
  }
  Vector3 operator+(const Vector3 &v) const
  {
    return Vector3(x() + v.x(), y() + v.y(), z() + v.z());
  }
  friend Vector3 operator*(double d, const Vector3 &v)
  {
    return Vector3(d * v[0], d * v[1], d * v[2]);
  }
  Vector3 operator-() const { return Vector3(-x(), -y(), -z()); }

 private:
  T _data[3] = {0, 0, 0};
};

class FixDXA : public Fix {
 public:
  FixDXA(class LAMMPS *, int, char **);

  void end_of_step() override;

  void init() override;

  void init_list(int, NeighList *) override;

  int setmask() override;

  void setup(int) override;

  const unsigned char VERSION = 1;

  enum StructureType { FCC = 0, HCP, BCC, CUBIC_DIA, HEX_DIA, OTHER };

 protected:
  struct CNANeighbor {
    Vector3<double> xyz = {0, 0, 0};
    double lengthSq = 0;
    int idx = -1;
    int neighIdx = -1;

    bool operator<(const CNANeighbor &o) const { return lengthSq < o.lengthSq; }
    bool operator>(const CNANeighbor &o) const { return lengthSq > o.lengthSq; }
  };

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
      // const int numNeighbors = countCommonNeighbors(n);
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

  bool getCNANeighbors(std::vector<CNANeighbor> &, const int, const int) const;

  void identifyCrystalStructure() const;

  StructureType _inputStructure;
  class NeighList *_neighList = nullptr;
  static constexpr size_t _maxNeighCount = 16;
};
}    // namespace LAMMPS_NS

#endif
#endif
