////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright 2022 OVITO GmbH, Germany
//
//  This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY KIND,
//  either express or implied. See the GPL or the MIT License for the specific language
//  governing rights and limitations.
//
////////////////////////////////////////////////////////////////////////////////////////

#ifndef LMP_FIX_DXA_MATH_H
#define LMP_FIX_DXA_MATH_H

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

    bool invertible() const { return std::abs(determinant()) > EPSILON; }

    Matrix3 inverse() const
    {
      assert(invertible());
      T det = determinant();
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

  using Vector3d = Vector3<double>;
  using Matrix3d = Matrix3<double>;
}    // namespace FIXDXA_NS
}    // namespace LAMMPS_NS

#endif
