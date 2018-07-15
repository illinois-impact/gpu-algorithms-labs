#ifndef __SHAPE_HPP__
#define __SHAPE_HPP__

#include <numeric>
#include <type_traits>

#ifdef __GNUC__
#define unused __attribute__((unused))
#else // __GNUC__
#define unused
#endif // __GNUC__

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

struct shape {
  size_t num{0};    // number of images in mini-batch
  size_t depth{0};  // number of input/output feature maps
  size_t height{0}; // height of the image
  size_t width{0};  // width of the image

  shape(size_t height, size_t width) : num(1), depth(1), height(height), width(width) {
  }

  shape(size_t num, size_t depth, size_t height, size_t width) : num(num), depth(depth), height(height), width(width) {
  }

  size_t flattened_length() const {
    return num * depth * height * width;
  }
};

template <typename T, size_t N>
constexpr size_t array_size(const T (&)[N]) {
  return N;
}

template <typename SzTy, size_t N>
static size_t flattened_length(const SzTy (&idims)[N]) {
  const auto dims = std::valarray<SzTy>(idims, N);
  size_t len      = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<SzTy>());
  return len;
}

template <typename SzTy>
static size_t flattened_length(const SzTy n) {
  return static_cast<size_t>(n);
}

template <>
size_t flattened_length(const shape &shp) {
  return shp.flattened_length();
}

template <typename T, typename SzTy>
static enable_if_t<std::is_integral<SzTy>::value, T> *allocate(const SzTy len) {
  T *res = new T[len];
  return res;
}

template <typename T, typename ShapeT>
static enable_if_t<std::is_same<ShapeT, shape>::value, T *> allocate(const ShapeT &shp) {
  T *res = new T[shp.flattened_length()];
  return res;
}

template <typename T, typename SzTy, size_t N>
static T *allocate(const SzTy (&dims)[N]) {
  const auto len = flattened_length(dims);
  return allocate<T>(len);
}

template <typename T, typename SzTy, size_t N>
static T *zeros(const SzTy (&dims)[N]) {
  const auto len = flattened_length(dims);
  auto res       = allocate<T, SzTy>(len);
  std::fill(res, res + len, static_cast<T>(0));
  return res;
}

template <typename T, typename SzTy>
static enable_if_t<std::is_integral<SzTy>::value, T *> zeros(const SzTy len) {
  auto res = allocate<T, SzTy>(len);
  std::fill(res, res + len, static_cast<T>(0));
  return res;
}

template <typename T, typename ShapeT>
static enable_if_t<std::is_same<ShapeT, shape>::value, T *> zeros(const ShapeT &shp) {
  auto res = allocate<T, shape>(shp);
  std::fill(res, res + shp.flattened_length(), static_cast<T>(0));
  return res;
}

#endif // _SHAPE_HPP__