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

  template <typename T> constexpr inline bool almostEqual(T a, T b, T eps = EPSILON)
  {
    return std::abs(a - b) < eps;
  };

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
    constexpr inline T dot(const Vector3 &v) const
    {
      return x() * v.x() + y() * v.y() + z() * v.z();
    }

    constexpr inline bool equals(const Vector3 &v, T eps) const
    {
      return (std::abs(x() - v.x()) < eps) && (std::abs(y() - v.y()) < eps) &&
          (std::abs(z() - v.z()) < eps);
    }

    constexpr Vector3 operator-(const Vector3 &v) const
    {
      return Vector3(x() - v.x(), y() - v.y(), z() - v.z());
    }
    constexpr Vector3 operator+(const Vector3 &v) const
    {
      return Vector3(x() + v.x(), y() + v.y(), z() + v.z());
    }
    constexpr Vector3 operator+(const T d) const { return Vector3(x() + d, y() + d, z() + d); }

    Vector3 &operator+=(const Vector3 &rhs)
    {
      (*this)[0] += rhs[0];
      (*this)[1] += rhs[1];
      (*this)[2] += rhs[2];
      return *this;
    }
    Vector3 &operator-=(const Vector3 &rhs)
    {
      (*this)[0] -= rhs[0];
      (*this)[1] -= rhs[1];
      (*this)[2] -= rhs[2];
      return *this;
    }

    friend Vector3 operator+(T d, const Vector3 &v)
    {
      return Vector3(d + v[0], d + v[1], d + v[2]);
    }
    friend Vector3 operator*(T d, const Vector3 &v)
    {
      return Vector3(d * v[0], d * v[1], d * v[2]);
    }
    friend Vector3 operator/(const Vector3 &v, T d)
    {
      return Vector3(v[0] / d, v[1] / d, v[2] / d);
    }
    constexpr Vector3 operator-() const { return Vector3(-x(), -y(), -z()); }
  };

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // VECTOR4
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  template <typename T> class Vector4 : public std::array<T, 4> {
   public:
    Vector4() = default;
    Vector4(T a, T b, T c, T d) : std::array<T, 4>{{a, b, c, d}} {};
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

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // Matrix4
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  template <typename T> class Matrix4 : public std::array<Vector4<T>, 4> {
   public:
    Matrix4() = default;
    Matrix4(T e00, T e01, T e02, T e03, T e10, T e11, T e12, T e13, T e20, T e21, T e22, T e23,
            T e30, T e31, T e32, T e33) :
        std::array<Vector4<T>, 4>{{Vector4<T>(e00, e10, e20, e30), Vector4<T>(e01, e11, e21, e31),
                                   Vector4<T>(e02, e12, e22, e32),
                                   Vector4<T>(e03, e13, e23, e33)}} {};
    Matrix4(const Vector4<T> &v0, const Vector4<T> &v1, const Vector4<T> &v2,
            const Vector4<T> &v3) :
        std::array<Vector4<T>, 4>{{v0, v1, v2, v3}} {};

    constexpr const Vector3<T> &column(size_t col) const { return (*this)[col]; }

    constexpr inline T determinant() const
    {
      return ((*this)[0][3] * (*this)[1][2] * (*this)[2][1] * (*this)[3][0] -
              (*this)[0][2] * (*this)[1][3] * (*this)[2][1] * (*this)[3][0] -
              (*this)[0][3] * (*this)[1][1] * (*this)[2][2] * (*this)[3][0] +
              (*this)[0][1] * (*this)[1][3] * (*this)[2][2] * (*this)[3][0] +
              (*this)[0][2] * (*this)[1][1] * (*this)[2][3] * (*this)[3][0] -
              (*this)[0][1] * (*this)[1][2] * (*this)[2][3] * (*this)[3][0] -
              (*this)[0][3] * (*this)[1][2] * (*this)[2][0] * (*this)[3][1] +
              (*this)[0][2] * (*this)[1][3] * (*this)[2][0] * (*this)[3][1] +
              (*this)[0][3] * (*this)[1][0] * (*this)[2][2] * (*this)[3][1] -
              (*this)[0][0] * (*this)[1][3] * (*this)[2][2] * (*this)[3][1] -
              (*this)[0][2] * (*this)[1][0] * (*this)[2][3] * (*this)[3][1] +
              (*this)[0][0] * (*this)[1][2] * (*this)[2][3] * (*this)[3][1] +
              (*this)[0][3] * (*this)[1][1] * (*this)[2][0] * (*this)[3][2] -
              (*this)[0][1] * (*this)[1][3] * (*this)[2][0] * (*this)[3][2] -
              (*this)[0][3] * (*this)[1][0] * (*this)[2][1] * (*this)[3][2] +
              (*this)[0][0] * (*this)[1][3] * (*this)[2][1] * (*this)[3][2] +
              (*this)[0][1] * (*this)[1][0] * (*this)[2][3] * (*this)[3][2] -
              (*this)[0][0] * (*this)[1][1] * (*this)[2][3] * (*this)[3][2] -
              (*this)[0][2] * (*this)[1][1] * (*this)[2][0] * (*this)[3][3] +
              (*this)[0][1] * (*this)[1][2] * (*this)[2][0] * (*this)[3][3] +
              (*this)[0][2] * (*this)[1][0] * (*this)[2][1] * (*this)[3][3] -
              (*this)[0][0] * (*this)[1][2] * (*this)[2][1] * (*this)[3][3] -
              (*this)[0][1] * (*this)[1][0] * (*this)[2][2] * (*this)[3][3] +
              (*this)[0][0] * (*this)[1][1] * (*this)[2][2] * (*this)[3][3]);
      // // Adapted from: https://graphics.stanford.edu/~mdfisher/Code/Engine/Matrix4.cpp.html
      // std::array<T, 12> tmp;
      // tmp[0] = (*this)[2][2] * (*this)[3][3];
      // tmp[1] = (*this)[2][3] * (*this)[3][2];
      // tmp[2] = (*this)[2][1] * (*this)[3][3];
      // tmp[3] = (*this)[2][3] * (*this)[3][1];
      // tmp[4] = (*this)[2][1] * (*this)[3][2];
      // tmp[5] = (*this)[2][2] * (*this)[3][1];
      // tmp[6] = (*this)[2][0] * (*this)[3][3];
      // tmp[7] = (*this)[2][3] * (*this)[3][0];
      // tmp[8] = (*this)[2][0] * (*this)[3][2];
      // tmp[9] = (*this)[2][2] * (*this)[3][0];
      // tmp[10] = (*this)[2][0] * (*this)[3][1];
      // tmp[11] = (*this)[2][1] * (*this)[3][0];

      // std::array<T, 4> tmp2;
      // tmp2[0] = (tmp[0] * (*this)[1][1] + tmp[3] * (*this)[1][2] + tmp[4] * (*this)[1][3]) -
      //     (tmp[1] * (*this)[1][1] + tmp[2] * (*this)[1][2] + tmp[5] * (*this)[1][3]);
      // tmp2[1] = (tmp[1] * (*this)[1][0] + tmp[6] * (*this)[1][2] + tmp[9] * (*this)[1][3]) -
      //     (tmp[0] * (*this)[1][0] + tmp[7] * (*this)[1][2] + tmp[8] * (*this)[1][3]);
      // tmp2[2] = (tmp[2] * (*this)[1][0] + tmp[7] * (*this)[1][1] + tmp[10] * (*this)[1][3]) -
      //     (tmp[3] * (*this)[1][0] + tmp[6] * (*this)[1][1] + tmp[11] * (*this)[1][3]);
      // tmp2[3] = (tmp[5] * (*this)[1][0] + tmp[8] * (*this)[1][1] + tmp[11] * (*this)[1][2]) -
      //     (tmp[4] * (*this)[1][0] + tmp[9] * (*this)[1][1] + tmp[10] * (*this)[1][2]);

      // return (*this)[0][0] * tmp2[0] + (*this)[0][1] * tmp2[1] + (*this)[0][2] * tmp2[2] +
      //     (*this)[0][3] * tmp2[3];
    }
  };

  template <typename T> class Plane {
   public:
    // https://mathworld.wolfram.com/Plane.html
    Plane() = default;
    Plane(const Vector3<T> &p0, const Vector3<T> &p1, const Vector3<T> &p2)
    {
      replaceData(p0, p1, p2);
    }

    void replaceData(const Vector3<T> &p0, const Vector3<T> &p1, const Vector3<T> &p2)
    {
      Vector3<T> n = (p1 - p0).cross(p2 - p0);
      T d = -n.x() * p0.x() - n.y() * p0.y() - n.z() * p0.z();
      T length_n = n.length();
      _p = d / length_n;
      _n = n / length_n;
      assert(almostEqual(_n.length(), 1.0));
    }

    T getSignedPointDistance(const Vector3<T> &p0) const { return _n.dot(p0) + _p; }
    T getPointDistance(const Vector3<T> &p0) const { return std::abs(getSignedPointDistance(p0)); }

    const Vector3<T> &getPlaneNormal() const { return _n; }
    T getP() const { return _p; }

   private:
    Vector3<T> _n;
    T _p;
  };

  template <typename T> class Sphere {
    // https://people.math.sc.edu/Burkardt/classes/cg_2007/cg_lab_tetrahedrons.pdf
   public:
    Sphere(Vector3<T> p0, Vector3<T> p1, Vector3<T> p2, Vector3<T> p3)
    {
      // Translate 1 corner to the origin -> helps with nummeric stability
      // https://stackoverflow.com/a/12259182
      const Vector3<T> p0o = p0;
      p0 -= p0o;
      p1 -= p0o;
      p2 -= p0o;
      p3 -= p0o;
      const Vector4<T> pxp = {p0.dot(p0), p1.dot(p1), p2.dot(p2), p3.dot(p3)};
      Matrix4<T> matrix = {p0[0], p0[1], p0[2], 1.0, p1[0], p1[1], p1[2], 1.0,
                           p2[0], p2[1], p2[2], 1.0, p3[0], p3[1], p3[2], 1.0};
      const T alpha = matrix.determinant();
      matrix = {pxp[0], p0[0], p0[1], p0[2], pxp[1], p1[0], p1[1], p1[2],
                pxp[2], p2[0], p2[1], p2[2], pxp[3], p3[0], p3[1], p3[2]};
      const T gamma = matrix.determinant();
      matrix = {pxp[0], p0[1], p0[2], 1.0, pxp[1], p1[1], p1[2], 1.0,
                pxp[2], p2[1], p2[2], 1.0, pxp[3], p3[1], p3[2], 1.0};
      const T Dx = matrix.determinant();
      matrix = {pxp[0], p0[0], p0[2], 1.0, pxp[1], p1[0], p1[2], 1.0,
                pxp[2], p2[0], p2[2], 1.0, pxp[3], p3[0], p3[2], 1.0};
      const T Dy = -1.0 * matrix.determinant();
      matrix = {pxp[0], p0[0], p0[1], 1.0, pxp[1], p1[0], p1[1], 1.0,
                pxp[2], p2[0], p2[1], 1.0, pxp[3], p3[0], p3[1], 1.0};
      const T Dz = matrix.determinant();
      if (std::abs(alpha) < EPSILON) {
        _valid = false;
      } else {
        _origin = {Dx / (2.0 * alpha), Dy / (2.0 * alpha), Dz / (2.0 * alpha)};
        // Reset the translation original translation
        _origin += p0o;
        _radius =
            std::sqrt(Dx * Dx + Dy * Dy + Dz * Dz - 4.0 * alpha * gamma) / (2.0 * std::abs(alpha));
        _valid = true;
      }
#ifndef NDEBUG
      if (valid()) {
        const T eps = 1e-6;
        T r0 = (p0 + p0o - _origin).length();
        T r1 = (p1 + p0o - _origin).length();
        T r2 = (p2 + p0o - _origin).length();
        T r3 = (p3 + p0o - _origin).length();

        // if (!almostEqual(_radius, r0, eps) || !almostEqual(_radius, r1, eps) ||
        //     !almostEqual(_radius, r2, eps) || !almostEqual(_radius, r3, eps)) {

        //   std::cerr << std::setiosflags(std::ios::fixed);
        //   std::cerr << std::setprecision(16);
        //   std::cerr << _radius << " r0: " << r0 << "\n";
        //   std::cerr << _radius << " r1: " << r1 << "\n";
        //   std::cerr << _radius << " r2: " << r2 << "\n";
        //   std::cerr << _radius << " r3: " << r3 << "\n";
        //   std::cerr << "origin: " << _origin[0] << " " << _origin[1] << " " << _origin[2] << "\n";
        //   std::cerr << "Dx: " << Dx << " "
        //             << "Dy: " << Dy << " "
        //             << "Dz: " << Dz << " "
        //             << "alpha: " << alpha << " gamma: " << gamma << "\n";
        //   std::cerr << "p0: " << p0[0] << " " << p0[1] << " " << p0[2] << "\n";
        //   std::cerr << "p1: " << p1[0] << " " << p1[1] << " " << p1[2] << "\n";
        //   std::cerr << "p2: " << p2[0] << " " << p2[1] << " " << p2[2] << "\n";
        //   std::cerr << "p3: " << p3[0] << " " << p3[1] << " " << p3[2] << "\n";
        //   std::cerr << '\n';
        // }
        assert(almostEqual(_radius, r0, eps));
        assert(almostEqual(_radius, r1, eps));
        assert(almostEqual(_radius, r2, eps));
        assert(almostEqual(_radius, r3, eps));
      }
#endif
    }
    bool valid() const { return _valid; }
    const Vector3<T> &origin() const { return _origin; }
    T radius() const { return _radius; }

   private:
    double get_radius(double Dx, double Dy, double Dz, double alpha, double gamma)
    {
      if (Dx > Dz) { std::swap(Dx, Dz); }
      if (Dx > Dy) { std::swap(Dx, Dy); }
      if (Dy > Dz) { std::swap(Dy, Dz); }
      assert(Dx < Dy && Dy < Dz);
      double t_0 = 2.0 * std::fabs(alpha);
      double tmp;
      if ((((Dx * Dx) + (Dy * Dy)) + (Dz * Dz)) <= 5e-323) {
        tmp = std::hypot(std::hypot(Dz, std::sqrt((gamma * (alpha * -4.0)))), std::hypot(Dx, Dy)) /
            t_0;
      } else {
        tmp = std::sqrt(
                  std::fma(Dy, Dy, std::fma(Dx, Dx, std::fma(Dz, Dz, (alpha * (gamma * -4.0)))))) /
            t_0;
      }
      return tmp;
    }

    bool _valid = false;
    Vector3<T> _origin;
    T _radius;
  };

  using Vector3d = Vector3<double>;
  using Vector4d = Vector4<double>;
  using Matrix3d = Matrix3<double>;
  using Matrix4d = Matrix4<double>;

}    // namespace FIXDXA_NS
}    // namespace LAMMPS_NS

#endif
