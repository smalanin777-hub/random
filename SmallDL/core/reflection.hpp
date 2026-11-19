#ifndef REFLECTION_HPP
#define REFLECTION_HPP

#include <type_traits>
#include <vector>

struct Shape {
  unsigned int rows = 0;
  unsigned int cols = 0;
};

namespace reflect {

struct Type {
  template <typename T> operator T() const;
};

template <typename T, typename... Args> consteval unsigned int count_fields() {
  if constexpr (requires { T{Args{}..., Type{}}; }) {
    return count_fields<T, Args..., Type>();
  } else {
    return sizeof...(Args);
  }
}

template <typename T> constexpr unsigned int number_of_fields = count_fields<std::remove_cvref_t<T>>();

template <typename T>
concept Aggregate = std::is_aggregate_v<std::remove_cvref_t<T>> && !std::is_array_v<std::remove_cvref_t<T>>;

template <typename T, typename Func> constexpr void for_each_field(std::vector<T> &vec, Func &&fn) {
  for (auto &i : vec)
    fn(i);
}

template <Aggregate T, typename Func> constexpr void for_each_field(T &a, Func &&fn) {
  constexpr unsigned int n = number_of_fields<T>;
  if constexpr (n == 1) {
    auto &[f0] = a;
    fn(f0);
  } else if constexpr (n == 2) {
    auto &[f0, f1] = a;
    fn(f0);
    fn(f1);
  } else if constexpr (n == 3) {
    auto &[f0, f1, f2] = a;
    fn(f0);
    fn(f1);
    fn(f2);
  } else if constexpr (n == 4) {
    auto &[f0, f1, f2, f3] = a;
    fn(f0);
    fn(f1);
    fn(f2);
    fn(f3);
  }
}

template <typename Target, typename Root, typename Func> void bypass(Root &r, Func &&fn) {
  using root_type = std::remove_cvref_t<Root>;
  if constexpr (std::is_same_v<root_type, Target>) {
    fn(r);
  } else if constexpr (Aggregate<root_type> || requires {
                         r.begin();
                         r.end();
                       }) {
    for_each_field(r, [&](auto &field) { bypass<Target>(field, fn); });
  }
}

enum OpFlags : unsigned char {
  OP_NONE = 0,
  OP_ELEMENTWISE = 1,
  OP_CAN_ACCEPT_EPI = 1 << 1,
  OP_CAN_BE_EPI = 1 << 2,
  OP_INPLACE_SAFE = 1 << 3
};

using forward_func = void (*)(void *graph, unsigned int node_id);
using backward_func = void (*)(void *graph, unsigned int node_id);

struct OpMetaData {
  const char *name;
  unsigned char flags;
  unsigned char arguments;
  forward_func forward;
  backward_func backward;
  constexpr bool has_flag(OpFlags f) const { return (flags & f); }
};

inline std::vector<OpMetaData> &op_registry() {
  static std::vector<OpMetaData> reg;
  return reg;
}

template <typename Operation> unsigned char get_op_id() {
  static const unsigned char id = []() {
    unsigned char new_id = static_cast<unsigned char>(op_registry().size());
    op_registry().push_back(Operation::meta());
    return new_id;
  }();
  return id;
}

inline const OpMetaData &get_op_meta(unsigned char id) { return op_registry()[id]; }

} // namespace reflect

#endif
