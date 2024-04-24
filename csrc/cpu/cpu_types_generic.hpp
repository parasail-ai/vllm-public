
#ifndef CPU_TYPES_GENERIC_HPP
#define CPU_TYPES_GENERIC_HPP

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdfloat>
#include <type_traits>

#if __has_include(<immintrin.h>)
#include <immintrin.h>
#endif

#include <torch/extension.h>

namespace vec_op {

template <typename T> struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

template <typename T, int N>
struct GenericVec : public Vec<GenericVec<T, N>> {
  constexpr static int VEC_ELEM_NUM = N;
  T reg[N];

  explicit GenericVec(T v) {
    std::fill_n(reg, N, v);
  }

  explicit GenericVec() {
    std::fill_n(reg, N, static_cast<T>(0));
  }

  explicit GenericVec(const T *ptr) {
    std::copy(ptr, ptr+N, reg);
  }

  template <typename BT, int BN>
  explicit GenericVec(const GenericVec<BT, BN>& b) {
    constexpr int group_size = BN;
    static_assert(VEC_ELEM_NUM % group_size == 0);
    constexpr int num_copies = N / BN;
    for (int i = 0; i < num_copies; ++i) {
      for (int j = 0; j < group_size; ++j) {
        reg[i * group_size + j] = (T)b.reg[j];
      }
    }
  }

  void save(T *ptr) const { std::copy(reg, reg+N, ptr); }

  template <typename OP>
  GenericVec<T, N> apply(OP op) const {
    GenericVec<T, N> ret;
    std::transform(reg, reg+N, ret.reg, op);
    return ret;
  }

  template <typename BINOP>
  GenericVec<T, N> binop(const GenericVec<T, N> &b, BINOP op) const {
    GenericVec<T, N> ret;
    std::transform(reg, reg+N, b.reg, ret.reg, op);
    return ret;
  }

  GenericVec<T, N> operator*(const GenericVec<T, N> &b) const {
    return binop(b, std::multiplies<T>());
  }
  GenericVec<T, N> operator+(const GenericVec<T, N> &b) const {
    return binop(b, std::plus<T>());
  }
  GenericVec<T, N> operator-(const GenericVec<T, N> &b) const {
    return binop(b, std::minus<T>());
  }
  GenericVec<T, N> operator/(const GenericVec<T, N> &b) const {
    return binop(b, std::divides<T>());
  }

  T reduce_sum() const {
    return std::reduce(reg, reg+N);
  }

  template <int group_size>
  T reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);
    constexpr int count = (16 - group_size);
    const int start = (idx * group_size);
    return std::reduce(reg + start, reg + start + count);
  }

  GenericVec<T, N> exp() const {
    return apply([](T x) { return expf(x); });
  }
  GenericVec<T, N> tanh() const {
    return apply([](T x) { return tanhf(x); });
  }
  GenericVec<T, N> er() const {
    return apply([](T x) { return erf(x); });
  }
};

using FP32Vec4 = GenericVec<float, 4>;
using FP32Vec8 = GenericVec<float, 8>;
using FP32Vec16 = GenericVec<float, 16>;

using FP16Vec8 = GenericVec<std::float16_t, 8>;
using FP16Vec16 = GenericVec<std::float16_t, 16>;

using BF16Vec8 = GenericVec<c10::BFloat16, 8>;
using BF16Vec16 = GenericVec<c10::BFloat16, 16>;


template <typename T> void storeFP32(float v, T *ptr) { *ptr = v; }

template <> inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16 *ptr) {
  c10::BFloat16 __attribute__((__may_alias__)) *v_ptr =
      reinterpret_cast<c10::BFloat16 *>(&v);
  *ptr = *(v_ptr + 1);
}


inline void fma(FP32Vec16 &acc, FP32Vec16 &a, FP32Vec16 &b) {
  acc = acc + a * b;
}


template <typename T> struct VecType { using vec_type = void; };

template <typename T> using vec_t = typename VecType<T>::vec_type;

template <> struct VecType<float> { using vec_type = FP32Vec8; };

template <> struct VecType<c10::Half> { using vec_type = FP16Vec16; };

template <> struct VecType<c10::BFloat16> { using vec_type = BF16Vec8; };


#if __has_include(<immintrin.h>)
inline void prefetch(const void *addr) { _mm_prefetch(addr, _MM_HINT_T1); }
#else
inline void prefetch(const void *addr) {}
#endif

}; // namespace vec_op

#endif
