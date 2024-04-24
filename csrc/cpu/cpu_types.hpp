#ifndef CPU_TYPES_HPP
#define CPU_TYPES_HPP

#ifdef VLLM_CPU_GENERIC
#include "cpu_types_generic.hpp"
#else
#include "cpu_types_avx512.hpp"
#endif

namespace vec_op {

// FIXME: FP16 is not fully supported in Torch-CPU
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)                                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef CPU_OP_GUARD
#define CPU_KERNEL_GUARD_IN(NAME)
#define CPU_KERNEL_GUARD_OUT(NAME)
#else
#define CPU_KERNEL_GUARD_IN(NAME)                                              \
  std::cout << #NAME << " invoked." << std::endl;
#define CPU_KERNEL_GUARD_OUT(NAME) std::cout << #NAME << " exit." << std::endl;
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F &&f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
}; // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F &&f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

}; // namespace vec_op

#endif