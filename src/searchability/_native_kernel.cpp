// Copyright (c) Searchability-Obligations contributors.
//
// A deliberately small, dependency-free (apart from pybind11) native kernel
// for Issue #3.  This file does not call Faiss or a BLAS: the certified path
// has an explicit scalar operation order and uses ordinary binary64 L2 units.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __FAST_MATH__
#error "The certified native kernel must not be compiled with fast-math"
#endif
#ifndef SEARCHABILITY_NATIVE_SOURCE_SHA256
#error "Native source identity must be embedded by setup.py"
#endif
#ifndef SEARCHABILITY_COMPILER_EXECUTABLE
#error "Native compiler executable must be embedded by setup.py"
#endif
#ifndef SEARCHABILITY_COMPILER_COMMAND
#error "Native compiler command must be embedded by setup.py"
#endif
#ifndef SEARCHABILITY_COMPILER_COMMAND_SOURCE
#error "Native compiler provenance must be embedded by setup.py"
#endif

namespace py = pybind11;

namespace {

constexpr std::size_t kAlignment = 64;
constexpr std::size_t kMaxDimension = 4096;
constexpr float kMaxAbsComponent = 1.0e15F;
constexpr std::int64_t kBaseSourceGroup = -2;
constexpr std::int64_t kRawSourceGroup = -1;

std::atomic<std::uint64_t> g_call_counter{0};

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(const Clock::time_point start) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - start);
    return static_cast<std::uint64_t>(elapsed.count());
}

std::size_t checked_product(
    const std::size_t left,
    const std::size_t right,
    const char *const name) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw py::value_error(std::string(name) + " is too large");
    }
    return left * right;
}

std::size_t checked_add(
    const std::size_t left,
    const std::size_t right,
    const char *const name) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw py::value_error(std::string(name) + " is too large");
    }
    return left + right;
}

std::size_t shape_size(const py::ssize_t value, const char *const name) {
    if (value < 0) {
        throw py::value_error(std::string(name) + " has a negative shape");
    }
    const auto converted = static_cast<unsigned long long>(value);
    if (converted > std::numeric_limits<std::size_t>::max()) {
        throw py::value_error(std::string(name) + " shape is too large");
    }
    return static_cast<std::size_t>(converted);
}

template <typename T>
T load_unaligned(const void *const base, const std::size_t index) noexcept {
    T result;
    const auto *const bytes = static_cast<const unsigned char *>(base);
    std::memcpy(&result, bytes + index * sizeof(T), sizeof(T));
    return result;
}

template <typename T>
struct BorrowedArray {
    py::array owner;
    py::buffer_info buffer;

    [[nodiscard]] std::size_t size() const {
        return shape_size(buffer.size, "array");
    }

    [[nodiscard]] T load(const std::size_t index) const noexcept {
        return load_unaligned<T>(buffer.ptr, index);
    }
};

template <typename T>
BorrowedArray<T> require_array(
    const py::handle input,
    const char *const name,
    const int dimensions) {
    if (!py::isinstance<py::array>(input)) {
        throw py::type_error(std::string(name) + " must be a NumPy array");
    }
    py::array array = py::reinterpret_borrow<py::array>(input);
    if (!array.dtype().equal(py::dtype::of<T>())) {
        throw py::type_error(
            std::string(name) + " must have the exact native dtype " +
            py::str(py::dtype::of<T>()).cast<std::string>());
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(
            std::string(name) + " must be C-contiguous; implicit copies are disabled");
    }
    py::buffer_info information = array.request();
    if (information.ndim != dimensions) {
        throw py::value_error(
            std::string(name) + " must have " + std::to_string(dimensions) +
            (dimensions == 1 ? " dimension" : " dimensions"));
    }
    if (information.itemsize != static_cast<py::ssize_t>(sizeof(T))) {
        throw py::type_error(std::string(name) + " has an unexpected item size");
    }
    return BorrowedArray<T>{std::move(array), std::move(information)};
}

std::size_t require_nonnegative_size(
    const py::handle value,
    const char *const name,
    const bool require_positive) {
    if (PyBool_Check(value.ptr()) || !PyLong_Check(value.ptr())) {
        throw py::type_error(std::string(name) + " must be an integer");
    }
    const long long converted = PyLong_AsLongLong(value.ptr());
    if (converted == -1 && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    if (converted < 0 || (require_positive && converted == 0)) {
        throw py::value_error(
            std::string(name) + (require_positive ? " must be positive"
                                                  : " must be non-negative"));
    }
    const auto unsigned_value = static_cast<unsigned long long>(converted);
    if (unsigned_value > std::numeric_limits<std::size_t>::max()) {
        throw py::value_error(std::string(name) + " is too large");
    }
    return static_cast<std::size_t>(unsigned_value);
}

void validate_component(const float value, const char *const name) {
    if (!std::isfinite(value)) {
        throw py::value_error(std::string(name) + " contains NaN or infinity");
    }
    if (value > kMaxAbsComponent || value < -kMaxAbsComponent) {
        throw py::value_error(
            std::string(name) + " component magnitude exceeds 1e15");
    }
}

void validate_float_array(
    const BorrowedArray<float> &array,
    const char *const name) {
    for (std::size_t index = 0; index < array.size(); ++index) {
        validate_component(array.load(index), name);
    }
}

bool has_immutable_bytes_backing(const py::array &array) {
    if (array.writeable()) {
        return false;
    }
    py::object current = py::reinterpret_borrow<py::object>(array);
    // NumPy reshape/view chains normally have depth two here.  The finite
    // guard is defensive against a malicious array subclass with a cyclic
    // ``base`` attribute.
    for (int depth = 0; depth < 64; ++depth) {
        if (PyBytes_Check(current.ptr())) {
            return true;
        }
        if (!py::isinstance<py::array>(current)) {
            return false;
        }
        py::object base = current.attr("base");
        if (base.is_none()) {
            return false;
        }
        current = std::move(base);
    }
    return false;
}

template <typename T>
class AlignedBuffer {
  public:
    AlignedBuffer() = default;

    explicit AlignedBuffer(const std::size_t count) : size_(count) {
        if (count == 0) {
            return;
        }
        const std::size_t bytes = checked_product(count, sizeof(T), "aligned buffer");
        data_ = static_cast<T *>(
            ::operator new(bytes, std::align_val_t(kAlignment)));
    }

    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;

    AlignedBuffer(AlignedBuffer &&other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer &operator=(AlignedBuffer &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        release();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
        return *this;
    }

    ~AlignedBuffer() { release(); }

    void copy_from(const void *const source) {
        if (size_ != 0) {
            std::memcpy(data_, source, size_ * sizeof(T));
        }
    }

    [[nodiscard]] const T *data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] const T &operator[](const std::size_t index) const noexcept {
        return data_[index];
    }

  private:
    void release() noexcept {
        if (data_ != nullptr) {
            ::operator delete(data_, std::align_val_t(kAlignment));
            data_ = nullptr;
        }
        size_ = 0;
    }

    T *data_ = nullptr;
    std::size_t size_ = 0;
};

class PackedDeltaView {
  public:
    PackedDeltaView(
        const py::handle vectors_object,
        const py::handle logical_ids_object,
        const py::handle version_ids_object,
        const py::handle begin_seqs_object,
        const py::handle ordinals_object,
        const std::size_t raw_count,
        const py::handle group_offsets_object,
        const py::handle group_ids_object,
        const py::handle centers_object,
        const py::handle radius_uppers_object) {
        const auto vectors = require_array<float>(vectors_object, "vectors", 2);
        const auto logical_ids =
            require_array<std::int64_t>(logical_ids_object, "logical_ids", 1);
        const auto version_ids =
            require_array<std::int64_t>(version_ids_object, "version_ids", 1);
        const auto begin_seqs =
            require_array<std::int64_t>(begin_seqs_object, "begin_seqs", 1);
        const auto ordinals =
            require_array<std::int64_t>(ordinals_object, "ordinals", 1);
        const auto group_offsets =
            require_array<std::int64_t>(group_offsets_object, "group_offsets", 1);
        const auto group_ids =
            require_array<std::int64_t>(group_ids_object, "group_ids", 1);
        const auto centers = require_array<float>(centers_object, "centers", 2);
        const auto radius_uppers =
            require_array<double>(radius_uppers_object, "radius_uppers", 1);

        size_ = shape_size(vectors.buffer.shape[0], "vectors");
        dimension_ = shape_size(vectors.buffer.shape[1], "vectors");
        if (dimension_ < 1 || dimension_ > kMaxDimension) {
            throw py::value_error("vectors dimension must be in [1, 4096]");
        }
        if (raw_count > size_) {
            throw py::value_error("raw_count exceeds the packed vector count");
        }
        raw_count_ = raw_count;

        for (const auto *const item :
             {&logical_ids, &version_ids, &begin_seqs, &ordinals}) {
            if (item->size() != size_) {
                throw py::value_error(
                    "logical_ids, version_ids, begin_seqs, and ordinals must "
                    "match vectors.shape[0]");
            }
        }

        group_count_ = group_ids.size();
        if (group_offsets.size() != checked_add(group_count_, 1, "group count")) {
            throw py::value_error("group_offsets must have group_count + 1 entries");
        }
        if (shape_size(centers.buffer.shape[0], "centers") != group_count_ ||
            shape_size(centers.buffer.shape[1], "centers") != dimension_) {
            throw py::value_error("centers must have shape (group_count, dimension)");
        }
        if (radius_uppers.size() != group_count_) {
            throw py::value_error("radius_uppers must match group_count");
        }

        if (group_offsets.load(0) != static_cast<std::int64_t>(raw_count_)) {
            throw py::value_error("group_offsets[0] must equal raw_count");
        }
        std::int64_t previous_offset = group_offsets.load(0);
        for (std::size_t index = 1; index < group_offsets.size(); ++index) {
            const std::int64_t offset = group_offsets.load(index);
            if (offset < previous_offset || offset < 0 ||
                static_cast<unsigned long long>(offset) > size_) {
                throw py::value_error(
                    "group_offsets must be non-decreasing absolute packed offsets");
            }
            previous_offset = offset;
        }
        if (previous_offset != static_cast<std::int64_t>(size_)) {
            throw py::value_error("group_offsets[-1] must equal vectors.shape[0]");
        }

        validate_float_array(vectors, "vectors");
        validate_float_array(centers, "centers");

        std::unordered_set<std::int64_t> unique_ordinals;
        std::unordered_set<std::int64_t> unique_logical_ids;
        std::vector<std::pair<std::int64_t, std::int64_t>> logical_lookup;
        unique_ordinals.reserve(size_);
        unique_logical_ids.reserve(size_);
        logical_lookup.reserve(size_);
        for (std::size_t index = 0; index < size_; ++index) {
            const std::int64_t logical_id = logical_ids.load(index);
            const std::int64_t version_id = version_ids.load(index);
            const std::int64_t begin_seq = begin_seqs.load(index);
            const std::int64_t ordinal = ordinals.load(index);
            if (logical_id < 0 || version_id < 0 || begin_seq < 0 || ordinal < 0) {
                throw py::value_error(
                    "Delta logical IDs, version IDs, begin sequences, and ordinals "
                    "must be non-negative");
            }
            if (!unique_ordinals.insert(ordinal).second) {
                throw py::value_error("Delta ordinals must be unique");
            }
            // The Python visibility layer normally performs this resolution.
            // Rejecting duplicates here prevents any accidental post-threshold
            // replacement inside the native monotonicity argument.
            if (!unique_logical_ids.insert(logical_id).second) {
                throw py::value_error(
                    "PackedDeltaView contains duplicate visible logical IDs");
            }
            logical_lookup.emplace_back(logical_id, static_cast<std::int64_t>(index));
        }
        std::sort(logical_lookup.begin(), logical_lookup.end());

        std::unordered_set<std::int64_t> unique_group_ids;
        unique_group_ids.reserve(group_count_);
        for (std::size_t index = 0; index < group_count_; ++index) {
            const std::int64_t group_id = group_ids.load(index);
            const double radius = radius_uppers.load(index);
            if (group_id < 0) {
                throw py::value_error("group_ids must be non-negative");
            }
            if (!unique_group_ids.insert(group_id).second) {
                throw py::value_error("group_ids must be unique");
            }
            if (!std::isfinite(radius) || radius < 0.0) {
                throw py::value_error(
                    "radius_uppers must be finite and non-negative");
            }
        }

        const std::size_t vector_elements =
            checked_product(size_, dimension_, "packed vector matrix");
        const std::size_t center_elements =
            checked_product(group_count_, dimension_, "center matrix");

        vectors_ = AlignedBuffer<float>(vector_elements);
        logical_ids_ = AlignedBuffer<std::int64_t>(size_);
        version_ids_ = AlignedBuffer<std::int64_t>(size_);
        begin_seqs_ = AlignedBuffer<std::int64_t>(size_);
        ordinals_ = AlignedBuffer<std::int64_t>(size_);
        group_offsets_ = AlignedBuffer<std::int64_t>(group_offsets.size());
        group_ids_ = AlignedBuffer<std::int64_t>(group_count_);
        centers_ = AlignedBuffer<float>(center_elements);
        radius_uppers_ = AlignedBuffer<double>(group_count_);
        logical_lookup_ids_ = AlignedBuffer<std::int64_t>(size_);
        logical_lookup_rows_ = AlignedBuffer<std::int64_t>(size_);

        vectors_.copy_from(vectors.buffer.ptr);
        logical_ids_.copy_from(logical_ids.buffer.ptr);
        version_ids_.copy_from(version_ids.buffer.ptr);
        begin_seqs_.copy_from(begin_seqs.buffer.ptr);
        ordinals_.copy_from(ordinals.buffer.ptr);
        group_offsets_.copy_from(group_offsets.buffer.ptr);
        group_ids_.copy_from(group_ids.buffer.ptr);
        centers_.copy_from(centers.buffer.ptr);
        radius_uppers_.copy_from(radius_uppers.buffer.ptr);
        std::vector<std::int64_t> lookup_ids;
        std::vector<std::int64_t> lookup_rows;
        lookup_ids.reserve(size_);
        lookup_rows.reserve(size_);
        for (const auto &[logical_id, row] : logical_lookup) {
            lookup_ids.push_back(logical_id);
            lookup_rows.push_back(row);
        }
        logical_lookup_ids_.copy_from(lookup_ids.data());
        logical_lookup_rows_.copy_from(lookup_rows.data());

        owned_bytes_ = checked_product(vector_elements, sizeof(float), "owned bytes");
        owned_bytes_ = checked_add(
            owned_bytes_, checked_product(size_, 4 * sizeof(std::int64_t), "owned bytes"),
            "owned bytes");
        owned_bytes_ = checked_add(
            owned_bytes_,
            checked_product(group_offsets.size(), sizeof(std::int64_t), "owned bytes"),
            "owned bytes");
        owned_bytes_ = checked_add(
            owned_bytes_,
            checked_product(group_count_, sizeof(std::int64_t), "owned bytes"),
            "owned bytes");
        owned_bytes_ = checked_add(
            owned_bytes_, checked_product(center_elements, sizeof(float), "owned bytes"),
            "owned bytes");
        owned_bytes_ = checked_add(
            owned_bytes_, checked_product(group_count_, sizeof(double), "owned bytes"),
            "owned bytes");
        owned_bytes_ = checked_add(
            owned_bytes_,
            checked_product(size_, 2 * sizeof(std::int64_t), "owned bytes"),
            "owned bytes");
    }

    PackedDeltaView(const PackedDeltaView &) = delete;
    PackedDeltaView &operator=(const PackedDeltaView &) = delete;

    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t raw_count() const noexcept { return raw_count_; }
    [[nodiscard]] std::size_t group_count() const noexcept { return group_count_; }
    [[nodiscard]] std::size_t owned_bytes() const noexcept { return owned_bytes_; }

    [[nodiscard]] float vector_value(
        const std::size_t row,
        const std::size_t column) const noexcept {
        return vectors_[row * dimension_ + column];
    }
    [[nodiscard]] float center_value(
        const std::size_t group,
        const std::size_t column) const noexcept {
        return centers_[group * dimension_ + column];
    }
    [[nodiscard]] std::int64_t logical_id(const std::size_t row) const noexcept {
        return logical_ids_[row];
    }
    [[nodiscard]] std::int64_t version_id(const std::size_t row) const noexcept {
        return version_ids_[row];
    }
    [[nodiscard]] std::int64_t begin_seq(const std::size_t row) const noexcept {
        return begin_seqs_[row];
    }
    [[nodiscard]] std::int64_t ordinal(const std::size_t row) const noexcept {
        return ordinals_[row];
    }
    [[nodiscard]] std::size_t group_begin(const std::size_t group) const noexcept {
        return static_cast<std::size_t>(group_offsets_[group]);
    }
    [[nodiscard]] std::size_t group_end(const std::size_t group) const noexcept {
        return static_cast<std::size_t>(group_offsets_[group + 1]);
    }
    [[nodiscard]] std::int64_t group_id(const std::size_t group) const noexcept {
        return group_ids_[group];
    }
    [[nodiscard]] double radius_upper(const std::size_t group) const noexcept {
        return radius_uppers_[group];
    }
    [[nodiscard]] std::optional<std::size_t> row_for_logical_id(
        const std::int64_t logical_id) const noexcept {
        if (size_ == 0) {
            return std::nullopt;
        }
        const std::int64_t *const begin = logical_lookup_ids_.data();
        const std::int64_t *const end = begin + size_;
        const std::int64_t *const found =
            std::lower_bound(begin, end, logical_id);
        if (found == end || *found != logical_id) {
            return std::nullopt;
        }
        const std::size_t offset = static_cast<std::size_t>(found - begin);
        return static_cast<std::size_t>(logical_lookup_rows_[offset]);
    }

  private:
    std::size_t dimension_ = 0;
    std::size_t size_ = 0;
    std::size_t raw_count_ = 0;
    std::size_t group_count_ = 0;
    std::size_t owned_bytes_ = 0;
    AlignedBuffer<float> vectors_;
    AlignedBuffer<std::int64_t> logical_ids_;
    AlignedBuffer<std::int64_t> version_ids_;
    AlignedBuffer<std::int64_t> begin_seqs_;
    AlignedBuffer<std::int64_t> ordinals_;
    AlignedBuffer<std::int64_t> group_offsets_;
    AlignedBuffer<std::int64_t> group_ids_;
    AlignedBuffer<float> centers_;
    AlignedBuffer<double> radius_uppers_;
    AlignedBuffer<std::int64_t> logical_lookup_ids_;
    AlignedBuffer<std::int64_t> logical_lookup_rows_;
};

class ScopedNearestRounding {
  public:
    ScopedNearestRounding() : prior_(std::fegetround()) {
        if (prior_ == -1) {
            throw std::runtime_error("fegetround failed in the native kernel");
        }
        if (std::fesetround(FE_TONEAREST) != 0 || std::fegetround() != FE_TONEAREST) {
            throw std::runtime_error("failed to enforce FE_TONEAREST");
        }
    }

    ScopedNearestRounding(const ScopedNearestRounding &) = delete;
    ScopedNearestRounding &operator=(const ScopedNearestRounding &) = delete;

    ~ScopedNearestRounding() noexcept {
        if (prior_ != FE_TONEAREST) {
            (void)std::fesetround(prior_);
        }
    }

  private:
    int prior_;
};

// Volatile stores force every documented elementary operation to be rounded
// to binary64 even on targets whose expression evaluator has excess precision.
// The build separately disables contraction and unsafe reassociation.
inline double rounded_add(const double left, const double right) noexcept {
    volatile double result = left + right;
    return result;
}

inline double rounded_subtract(const double left, const double right) noexcept {
    volatile double result = left - right;
    return result;
}

inline double rounded_multiply(const double left, const double right) noexcept {
    volatile double result = left * right;
    return result;
}

inline double rounded_divide(const double left, const double right) noexcept {
    volatile double result = left / right;
    return result;
}

inline double rounded_sqrt(const double value) noexcept {
    volatile double result = std::sqrt(value);
    return result;
}

#if defined(__SIZEOF_INT128__)
bool double_is_at_least_positive_rational(
    const double value,
    const std::uint64_t numerator,
    const std::uint64_t denominator) noexcept {
    if (!(value > 0.0) || !std::isfinite(value)) {
        return false;
    }
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected binary64 size");
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint64_t raw_exponent = (bits >> 52U) & 0x7ffU;
    const std::uint64_t fraction = bits & ((std::uint64_t{1} << 52U) - 1U);
    if (raw_exponent == 0 || raw_exponent == 0x7ffU) {
        return false;
    }
    const std::uint64_t mantissa = (std::uint64_t{1} << 52U) | fraction;
    const int power = static_cast<int>(raw_exponent) - 1023 - 52;
    using UInt128 = unsigned __int128;
    if (power >= 0) {
        if (power >= 128) {
            return true;
        }
        const UInt128 left =
            (static_cast<UInt128>(mantissa) << power) * denominator;
        return left >= static_cast<UInt128>(numerator);
    }
    const int shift = -power;
    if (shift >= 128) {
        return false;
    }
    const UInt128 left = static_cast<UInt128>(mantissa) * denominator;
    const UInt128 right = static_cast<UInt128>(numerator) << shift;
    return left >= right;
}
#endif

double gamma_upper(const std::size_t operation_count) {
    if (operation_count >= (std::uint64_t{1} << 53U)) {
        throw std::runtime_error("dimension is too large for the gamma bound");
    }
    const std::uint64_t numerator = static_cast<std::uint64_t>(operation_count);
    const std::uint64_t denominator =
        (std::uint64_t{1} << 53U) - numerator;
    double result = rounded_divide(
        static_cast<double>(numerator), static_cast<double>(denominator));
#if defined(__SIZEOF_INT128__)
    // This exact integer comparison mirrors Fraction.from_float in the Python
    // reference and only advances gamma when nearest conversion rounded down.
    if (!double_is_at_least_positive_rational(result, numerator, denominator)) {
        result = std::nextafter(result, std::numeric_limits<double>::infinity());
    }
#else
    // One unconditional ulp is conservative on compilers without a 128-bit
    // integer.  It can only widen intervals and therefore cause extra scans.
    result = std::nextafter(result, std::numeric_limits<double>::infinity());
#endif
    return result;
}

struct DistanceInterval {
    double estimate = 0.0;
    double lower = 0.0;
    double upper = 0.0;
};

struct DistanceParameters {
    double denominator_high = 1.0;
    double denominator_low = 1.0;
};

DistanceParameters distance_parameters(const std::size_t dimension) {
    const double gamma = gamma_upper(dimension + 2);
    return DistanceParameters{
        std::nextafter(
            rounded_add(1.0, gamma), std::numeric_limits<double>::infinity()),
        std::nextafter(
            rounded_subtract(1.0, gamma),
            -std::numeric_limits<double>::infinity())};
}

template <typename QueryGetter, typename VectorGetter>
DistanceInterval distance_interval(
    const std::size_t dimension,
    const DistanceParameters &parameters,
    QueryGetter &&query_value,
    VectorGetter &&vector_value) {
    double sum = 0.0;
    for (std::size_t column = 0; column < dimension; ++column) {
        const double query = static_cast<double>(query_value(column));
        const double vector = static_cast<double>(vector_value(column));
        const double difference = rounded_subtract(query, vector);
        const double square = rounded_multiply(difference, difference);
        sum = rounded_add(sum, square);
    }
    if (!std::isfinite(sum) || sum < 0.0) {
        throw std::runtime_error("native distance accumulation overflowed");
    }
    if (sum == 0.0) {
        return DistanceInterval{0.0, 0.0, 0.0};
    }

    double lower_squared = std::nextafter(
        rounded_divide(sum, parameters.denominator_high),
        -std::numeric_limits<double>::infinity());
    const double upper_squared = std::nextafter(
        rounded_divide(sum, parameters.denominator_low),
        std::numeric_limits<double>::infinity());
    lower_squared = std::max(0.0, lower_squared);
    double lower = std::nextafter(
        rounded_sqrt(lower_squared), -std::numeric_limits<double>::infinity());
    lower = std::max(0.0, lower);
    const double upper = std::nextafter(
        rounded_sqrt(upper_squared), std::numeric_limits<double>::infinity());
    const double estimate = rounded_sqrt(sum);
    if (!std::isfinite(estimate) || !std::isfinite(lower) ||
        !std::isfinite(upper) || lower < 0.0 || lower > upper) {
        throw std::runtime_error("native distance interval postcondition failed");
    }
    return DistanceInterval{estimate, lower, upper};
}

double lower_bound_rounded_down(
    const double center_distance_lower,
    const double radius_upper) noexcept {
    const double rounded = rounded_subtract(center_distance_lower, radius_upper);
    const double downward =
        std::nextafter(rounded, -std::numeric_limits<double>::infinity());
    return std::max(0.0, downward);
}

double threshold_rounded_up(const double tau_upper, const double beta) noexcept {
    const double rounded = rounded_subtract(tau_upper, beta);
    return std::nextafter(rounded, std::numeric_limits<double>::infinity());
}

struct EvaluatedCandidate {
    std::uint8_t origin = 0;
    std::int64_t ordinal = 0;
    std::int64_t logical_id = 0;
    std::int64_t version_id = 0;
    std::int64_t source_group_id = kBaseSourceGroup;
    double estimate = 0.0;
    double lower = 0.0;
    double upper = 0.0;
};

class ThresholdTracker {
  public:
    ThresholdTracker(const std::size_t k, std::string mode)
        : k_(k), mode_(std::move(mode)) {}

    void add(const double upper) {
        if (mode_ != "heap") {
            return;
        }
        if (heap_.size() < k_) {
            heap_.push(upper);
        } else if (upper < heap_.top()) {
            heap_.pop();
            heap_.push(upper);
        }
    }

    std::optional<double> current(
        const std::vector<EvaluatedCandidate> &evaluated) const {
        if (evaluated.size() < k_) {
            return std::nullopt;
        }
        if (mode_ == "heap") {
            if (heap_.size() != k_) {
                throw std::runtime_error("incremental kth-upper heap is inconsistent");
            }
            return heap_.top();
        }
        std::vector<double> values;
        values.reserve(evaluated.size());
        for (const auto &candidate : evaluated) {
            values.push_back(candidate.upper);
        }
        std::nth_element(
            values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k_ - 1),
            values.end());
        return values[k_ - 1];
    }

  private:
    std::size_t k_;
    std::string mode_;
    std::priority_queue<double> heap_;
};

struct AuditRow {
    std::int64_t group_id = 0;
    std::size_t members_visible = 0;
    double radius_upper = 0.0;
    double center_distance_lower = 0.0;
    double center_distance_upper = 0.0;
    double lb_lower = 0.0;
    std::optional<double> tau_upper_at_decision;
    std::optional<double> threshold_tau_minus_beta;
    std::string action;
};

struct ComponentTimings {
    std::uint64_t deduplication_ns = 0;
    std::uint64_t candidate_scan_ns = 0;
    std::uint64_t raw_scan_ns = 0;
    std::uint64_t lb_calculation_ns = 0;
    std::uint64_t group_ordering_ns = 0;
    std::uint64_t threshold_selection_ns = 0;
    std::uint64_t group_scan_ns = 0;
    std::uint64_t retention_ns = 0;
    std::uint64_t kernel_total_ns = 0;
};

struct KernelResult {
    std::uint64_t call_index = 0;
    std::string mode_requested;
    std::string mode_executed;
    std::string threshold_mode;
    bool trusted_prevalidated_inputs = false;
    std::optional<std::string> fallback_reason;
    std::optional<double> final_kth_upper;
    std::optional<double> min_skipped_lb;
    std::size_t population = 0;
    std::size_t raw_pending_scanned = 0;
    std::size_t groups_scanned = 0;
    std::size_t groups_skipped = 0;
    std::size_t vectors_scanned = 0;
    std::size_t distance_evaluations = 0;
    std::size_t center_distance_evaluations = 0;
    std::size_t bytes_read = 0;
    std::vector<EvaluatedCandidate> retained;
    std::vector<AuditRow> audit_rows;
    ComponentTimings timings;
};

struct CandidateInputs {
    const void *query = nullptr;
    const void *vectors = nullptr;
    const void *logical_ids = nullptr;
    const void *version_ids = nullptr;
    const void *begin_seqs = nullptr;
    std::size_t count = 0;
};

float query_value(const CandidateInputs &inputs, const std::size_t column) noexcept {
    return load_unaligned<float>(inputs.query, column);
}

float candidate_vector_value(
    const CandidateInputs &inputs,
    const std::size_t row,
    const std::size_t dimension,
    const std::size_t column) noexcept {
    return load_unaligned<float>(inputs.vectors, row * dimension + column);
}

std::int64_t candidate_metadata_value(
    const void *const values,
    const std::size_t row) noexcept {
    return load_unaligned<std::int64_t>(values, row);
}

KernelResult execute_kernel(
    const PackedDeltaView &view,
    const CandidateInputs &inputs,
    const std::size_t k,
    const double beta,
    const std::string &requested_mode,
    const std::string &threshold_mode,
    const bool collect_audit,
    const bool trusted_prevalidated_inputs,
    const std::uint64_t call_index) {
    ScopedNearestRounding rounding;
    const auto kernel_start = Clock::now();
    KernelResult result;
    result.call_index = call_index;
    result.mode_requested = requested_mode;
    result.mode_executed = requested_mode;
    result.threshold_mode = threshold_mode;
    result.trusted_prevalidated_inputs = trusted_prevalidated_inputs;

    const auto deduplication_start = Clock::now();
    std::vector<std::uint8_t> drop_candidates(inputs.count, 0);
    std::vector<std::size_t> drop_delta_rows;
    std::size_t cross_origin_duplicate_count = 0;
    for (std::size_t row = 0; row < inputs.count; ++row) {
        const std::int64_t logical_id =
            candidate_metadata_value(inputs.logical_ids, row);
        const std::optional<std::size_t> delta_row =
            view.row_for_logical_id(logical_id);
        if (!delta_row.has_value()) {
            continue;
        }
        ++cross_origin_duplicate_count;
        const std::int64_t candidate_begin =
            candidate_metadata_value(inputs.begin_seqs, row);
        const std::int64_t candidate_version =
            candidate_metadata_value(inputs.version_ids, row);
        const std::int64_t delta_begin = view.begin_seq(*delta_row);
        const std::int64_t delta_version = view.version_id(*delta_row);
        // All packed Delta positions follow all C positions.  Therefore an
        // equal (begin_seq, version_id) is won by Delta under the required
        // (begin_seq, version_id, position) newest ordering.
        if (std::tie(delta_begin, delta_version) >=
            std::tie(candidate_begin, candidate_version)) {
            drop_candidates[row] = 1;
        } else {
            drop_delta_rows.push_back(*delta_row);
        }
    }
    std::sort(drop_delta_rows.begin(), drop_delta_rows.end());
    const auto delta_is_kept = [&](const std::size_t row) {
        return drop_delta_rows.empty() ||
               !std::binary_search(
                   drop_delta_rows.begin(), drop_delta_rows.end(), row);
    };
    result.population = checked_add(inputs.count, view.size(), "search population") -
                        cross_origin_duplicate_count;
    if (cross_origin_duplicate_count != 0 && requested_mode != "F") {
        result.mode_executed = "F";
        result.fallback_reason =
            "duplicate_visible_logical_id_forced_native_full_scan";
    }
    result.timings.deduplication_ns = elapsed_ns(deduplication_start);

    const DistanceParameters interval_parameters =
        distance_parameters(view.dimension());
    ThresholdTracker threshold(k, threshold_mode);
    std::vector<EvaluatedCandidate> evaluated;
    evaluated.reserve(result.population);

    const auto evaluate_candidate = [&](const std::size_t row) {
        const DistanceInterval interval = distance_interval(
            view.dimension(),
            interval_parameters,
            [&](const std::size_t column) { return query_value(inputs, column); },
            [&](const std::size_t column) {
                return candidate_vector_value(inputs, row, view.dimension(), column);
            });
        EvaluatedCandidate candidate;
        candidate.origin = 0;
        candidate.ordinal = static_cast<std::int64_t>(row);
        candidate.logical_id = candidate_metadata_value(inputs.logical_ids, row);
        candidate.version_id = candidate_metadata_value(inputs.version_ids, row);
        candidate.source_group_id = kBaseSourceGroup;
        candidate.estimate = interval.estimate;
        candidate.lower = interval.lower;
        candidate.upper = interval.upper;
        evaluated.push_back(candidate);
        threshold.add(interval.upper);
        ++result.distance_evaluations;
    };

    const auto evaluate_delta = [&](const std::size_t row, const std::int64_t source) {
        const DistanceInterval interval = distance_interval(
            view.dimension(),
            interval_parameters,
            [&](const std::size_t column) { return query_value(inputs, column); },
            [&](const std::size_t column) {
                return view.vector_value(row, column);
            });
        EvaluatedCandidate candidate;
        candidate.origin = 1;
        candidate.ordinal = view.ordinal(row);
        candidate.logical_id = view.logical_id(row);
        candidate.version_id = view.version_id(row);
        candidate.source_group_id = source;
        candidate.estimate = interval.estimate;
        candidate.lower = interval.lower;
        candidate.upper = interval.upper;
        evaluated.push_back(candidate);
        threshold.add(interval.upper);
        ++result.distance_evaluations;
        ++result.vectors_scanned;
    };

    auto phase_start = Clock::now();
    for (std::size_t row = 0; row < inputs.count; ++row) {
        if (drop_candidates[row] == 0) {
            evaluate_candidate(row);
        }
    }
    result.timings.candidate_scan_ns = elapsed_ns(phase_start);

    // Raw pending vectors are unconditionally handled before any grouped
    // lower-bound decision.  A superseded cross-origin duplicate was already
    // removed before the threshold existed and cannot invalidate it later.
    phase_start = Clock::now();
    for (std::size_t row = 0; row < view.raw_count(); ++row) {
        if (delta_is_kept(row)) {
            evaluate_delta(row, kRawSourceGroup);
            ++result.raw_pending_scanned;
        }
    }
    result.timings.raw_scan_ns = elapsed_ns(phase_start);

    struct GroupBound {
        std::size_t group = 0;
        std::size_t selected_members = 0;
        double center_lower = 0.0;
        double center_upper = 0.0;
        double lower_bound = 0.0;
    };

    if (result.mode_executed == "F") {
        phase_start = Clock::now();
        for (std::size_t group = 0; group < view.group_count(); ++group) {
            bool scanned_any = false;
            for (std::size_t row = view.group_begin(group);
                 row < view.group_end(group);
                 ++row) {
                if (delta_is_kept(row)) {
                    evaluate_delta(row, view.group_id(group));
                    scanned_any = true;
                }
            }
            if (scanned_any) {
                ++result.groups_scanned;
            }
        }
        result.timings.group_scan_ns = elapsed_ns(phase_start);
    } else {
        phase_start = Clock::now();
        std::vector<GroupBound> bounded_groups;
        bounded_groups.reserve(view.group_count());
        for (std::size_t group = 0; group < view.group_count(); ++group) {
            std::size_t selected_members = 0;
            for (std::size_t row = view.group_begin(group);
                 row < view.group_end(group);
                 ++row) {
                selected_members += delta_is_kept(row) ? 1U : 0U;
            }
            if (selected_members == 0) {
                continue;
            }
            const DistanceInterval center_interval = distance_interval(
                view.dimension(),
                interval_parameters,
                [&](const std::size_t column) { return query_value(inputs, column); },
                [&](const std::size_t column) {
                    return view.center_value(group, column);
                });
            const double lower_bound = lower_bound_rounded_down(
                center_interval.lower, view.radius_upper(group));
            bounded_groups.push_back(GroupBound{
                group,
                selected_members,
                center_interval.lower,
                center_interval.upper,
                lower_bound});
            ++result.center_distance_evaluations;
            ++result.distance_evaluations;
        }
        result.timings.lb_calculation_ns = elapsed_ns(phase_start);

        phase_start = Clock::now();
        std::sort(
            bounded_groups.begin(), bounded_groups.end(),
            [&](const GroupBound &left, const GroupBound &right) {
                if (left.lower_bound != right.lower_bound) {
                    return left.lower_bound < right.lower_bound;
                }
                return view.group_id(left.group) < view.group_id(right.group);
            });
        result.timings.group_ordering_ns = elapsed_ns(phase_start);

        for (const GroupBound &bounded : bounded_groups) {
            const auto threshold_start = Clock::now();
            const std::optional<double> tau = threshold.current(evaluated);
            result.timings.threshold_selection_ns = checked_add(
                result.timings.threshold_selection_ns,
                elapsed_ns(threshold_start),
                "threshold timing");
            std::optional<double> comparison_threshold;
            bool should_skip = false;
            if (tau.has_value()) {
                comparison_threshold = threshold_rounded_up(*tau, beta);
                should_skip = result.mode_executed == "P" &&
                              bounded.lower_bound > *comparison_threshold;
            }

            std::string action;
            if (should_skip) {
                ++result.groups_skipped;
                if (!result.min_skipped_lb.has_value() ||
                    bounded.lower_bound < *result.min_skipped_lb) {
                    result.min_skipped_lb = bounded.lower_bound;
                }
                action = "skip";
            } else {
                const auto group_scan_start = Clock::now();
                for (std::size_t row = view.group_begin(bounded.group);
                     row < view.group_end(bounded.group);
                     ++row) {
                    if (delta_is_kept(row)) {
                        evaluate_delta(row, view.group_id(bounded.group));
                    }
                }
                result.timings.group_scan_ns = checked_add(
                    result.timings.group_scan_ns,
                    elapsed_ns(group_scan_start),
                    "group scan timing");
                ++result.groups_scanned;
                action = "scan";
            }

            if (collect_audit) {
                result.audit_rows.push_back(AuditRow{
                    view.group_id(bounded.group),
                    bounded.selected_members,
                    view.radius_upper(bounded.group),
                    bounded.center_lower,
                    bounded.center_upper,
                    bounded.lower_bound,
                    tau,
                    comparison_threshold,
                    std::move(action)});
            }
        }
    }

    const auto final_threshold_start = Clock::now();
    result.final_kth_upper = threshold.current(evaluated);
    result.timings.threshold_selection_ns = checked_add(
        result.timings.threshold_selection_ns,
        elapsed_ns(final_threshold_start),
        "threshold timing");
    phase_start = Clock::now();
    if (result.population < k) {
        if (result.final_kth_upper.has_value() || result.groups_skipped != 0) {
            throw std::runtime_error(
                "population-below-k native invariant was violated");
        }
        result.retained = std::move(evaluated);
    } else {
        if (!result.final_kth_upper.has_value()) {
            throw std::runtime_error(
                "native search did not evaluate k candidates before pruning");
        }
        result.retained.reserve(evaluated.size());
        for (const auto &candidate : evaluated) {
            if (candidate.lower <= *result.final_kth_upper) {
                result.retained.push_back(candidate);
            }
        }
        if (result.retained.size() < k) {
            throw std::runtime_error(
                "native boundary retention discarded a required top-k candidate");
        }
    }
    result.timings.retention_ns = elapsed_ns(phase_start);

    result.bytes_read = checked_product(
        checked_product(result.distance_evaluations, view.dimension(), "bytes read"),
        sizeof(float),
        "bytes read");
    result.timings.kernel_total_ns = elapsed_ns(kernel_start);
    return result;
}

template <typename T, typename Getter>
py::array_t<T> make_array(const std::size_t count, Getter &&getter) {
    py::array_t<T> result(static_cast<py::ssize_t>(count));
    T *const output = result.mutable_data();
    for (std::size_t index = 0; index < count; ++index) {
        output[index] = getter(index);
    }
    return result;
}

void assign_optional_double(
    py::dict &dictionary,
    const char *const key,
    const std::optional<double> &value) {
    if (value.has_value()) {
        dictionary[py::str(key)] = py::float_(*value);
    } else {
        dictionary[py::str(key)] = py::none();
    }
}

py::dict kernel_result_to_python(
    const KernelResult &result,
    const bool collect_audit) {
    py::dict output;
    output["backend"] = "pybind11_cpp17";
    output["native_executed"] = true;
    output["call_index"] = result.call_index;
    output["mode_requested"] = result.mode_requested;
    output["mode_executed"] = result.mode_executed;
    output["threshold_mode"] = result.threshold_mode;
    output["trusted_prevalidated_inputs"] =
        result.trusted_prevalidated_inputs;
    if (result.fallback_reason.has_value()) {
        output["fallback_reason"] = *result.fallback_reason;
    } else {
        output["fallback_reason"] = py::none();
    }

    const std::size_t count = result.retained.size();
    output["origin"] = make_array<std::uint8_t>(
        count, [&](const std::size_t index) { return result.retained[index].origin; });
    output["ordinal"] = make_array<std::int64_t>(
        count, [&](const std::size_t index) { return result.retained[index].ordinal; });
    output["logical_ids"] = make_array<std::int64_t>(
        count,
        [&](const std::size_t index) { return result.retained[index].logical_id; });
    output["version_ids"] = make_array<std::int64_t>(
        count,
        [&](const std::size_t index) { return result.retained[index].version_id; });
    output["source_group_ids"] = make_array<std::int64_t>(
        count,
        [&](const std::size_t index) {
            return result.retained[index].source_group_id;
        });
    output["estimate"] = make_array<double>(
        count,
        [&](const std::size_t index) { return result.retained[index].estimate; });
    output["lower"] = make_array<double>(
        count, [&](const std::size_t index) { return result.retained[index].lower; });
    output["upper"] = make_array<double>(
        count, [&](const std::size_t index) { return result.retained[index].upper; });

    assign_optional_double(output, "final_kth_upper", result.final_kth_upper);
    assign_optional_double(output, "min_skipped_lb", result.min_skipped_lb);
    output["population"] = result.population;
    output["raw_pending_scanned"] = result.raw_pending_scanned;
    output["groups_scanned"] = result.groups_scanned;
    output["groups_skipped"] = result.groups_skipped;
    output["vectors_scanned"] = result.vectors_scanned;
    output["distance_evaluations"] = result.distance_evaluations;
    output["center_distance_evaluations"] = result.center_distance_evaluations;
    output["bytes_read"] = result.bytes_read;
    output["retained_count"] = result.retained.size();

    py::dict timings;
    timings["deduplication_ns"] = result.timings.deduplication_ns;
    timings["candidate_scan_ns"] = result.timings.candidate_scan_ns;
    timings["raw_scan_ns"] = result.timings.raw_scan_ns;
    timings["lb_calculation_ns"] = result.timings.lb_calculation_ns;
    timings["group_ordering_ns"] = result.timings.group_ordering_ns;
    timings["threshold_selection_ns"] = result.timings.threshold_selection_ns;
    timings["group_scan_ns"] = result.timings.group_scan_ns;
    timings["retention_ns"] = result.timings.retention_ns;
    timings["kernel_total_ns"] = result.timings.kernel_total_ns;
    output["component_timings_ns"] = std::move(timings);

    py::dict feature_flags;
    feature_flags["packed_owned_aligned"] = true;
    feature_flags["input_forcecast"] = false;
    feature_flags["gil_released_during_kernel"] = true;
    feature_flags["ordinary_l2_gamma_interval"] = true;
    feature_flags["strict_outward_skip_operands"] = true;
    feature_flags["incremental_kth_heap"] = result.threshold_mode == "heap";
    feature_flags["kth_rescan_ablation"] = result.threshold_mode == "rescan";
    feature_flags["duplicate_full_scan_fallback"] = true;
    feature_flags["exact_boundary_deferred_to_python"] = true;
    feature_flags["trusted_prevalidated_inputs"] =
        result.trusted_prevalidated_inputs;
    output["feature_flags"] = std::move(feature_flags);

    if (collect_audit) {
        py::list rows;
        for (const AuditRow &row : result.audit_rows) {
            py::dict item;
            item["group_id"] = row.group_id;
            item["members_visible"] = row.members_visible;
            item["radius_upper"] = row.radius_upper;
            item["center_distance_lower"] = row.center_distance_lower;
            item["center_distance_upper"] = row.center_distance_upper;
            item["lb_lower"] = row.lb_lower;
            if (row.tau_upper_at_decision.has_value()) {
                item["tau_upper_at_decision"] = *row.tau_upper_at_decision;
            } else {
                item["tau_upper_at_decision"] = py::none();
            }
            if (row.threshold_tau_minus_beta.has_value()) {
                item["threshold_tau_minus_beta"] =
                    *row.threshold_tau_minus_beta;
            } else {
                item["threshold_tau_minus_beta"] = py::none();
            }
            item["action"] = row.action;
            rows.append(std::move(item));
        }
        output["audit"] = std::move(rows);
    } else {
        output["audit"] = py::none();
    }
    return output;
}

py::dict build_info() {
    py::dict macros;
#ifdef __FAST_MATH__
    macros["__FAST_MATH__"] = true;
#else
    macros["__FAST_MATH__"] = false;
#endif
#ifdef __FMA__
    macros["__FMA__"] = true;
#else
    macros["__FMA__"] = false;
#endif
#ifdef __AVX2__
    macros["__AVX2__"] = true;
#else
    macros["__AVX2__"] = false;
#endif
#ifdef __AVX512F__
    macros["__AVX512F__"] = true;
#else
    macros["__AVX512F__"] = false;
#endif
#ifdef __STDC_IEC_559__
    macros["__STDC_IEC_559__"] = true;
#else
    macros["__STDC_IEC_559__"] = false;
#endif
#if defined(__SIZEOF_INT128__)
    macros["__SIZEOF_INT128__"] = static_cast<int>(__SIZEOF_INT128__);
#else
    macros["__SIZEOF_INT128__"] = py::none();
#endif

    py::dict result;
    result["backend"] = "pybind11_cpp17";
    result["cxx_standard"] = static_cast<long long>(__cplusplus);
    result["compiled_source_sha256"] = SEARCHABILITY_NATIVE_SOURCE_SHA256;
    result["compiler_executable"] = SEARCHABILITY_COMPILER_EXECUTABLE;
    result["compile_driver_and_flags"] = SEARCHABILITY_COMPILER_COMMAND;
    result["compile_command_scope"] =
        "compiler driver/default flags/Extension.extra_compile_args before "
        "per-source include, macro, compile, and output arguments";
    result["compile_command_provenance"] =
        SEARCHABILITY_COMPILER_COMMAND_SOURCE;
#ifdef __VERSION__
    result["compiler_version"] = __VERSION__;
#else
    result["compiler_version"] = "unknown";
#endif
    result["alignment_bytes"] = kAlignment;
    result["max_dimension"] = kMaxDimension;
    result["max_abs_component"] = 1.0e15;
    result["numeric_mode"] =
        "float32_storage+scalar_float64_gamma_interval+python_exact_boundary";
    result["requires_rounding_mode"] = "FE_TONEAREST";
    result["rounding_mode_enforced"] = true;
    py::list compile_flags;
    compile_flags.append("-O3");
    compile_flags.append("-std=c++17");
    compile_flags.append("-fno-fast-math");
    compile_flags.append("-ffp-contract=off");
    result["compile_flags"] = std::move(compile_flags);
    result["compile_macros"] = std::move(macros);
    return result;
}

}  // namespace

PYBIND11_MODULE(_native_kernel, module) {
    module.doc() =
        "Issue #3 packed certified F/N/P native search kernel (C++17/pybind11).";

    py::class_<PackedDeltaView, std::shared_ptr<PackedDeltaView>>(
        module, "PackedDeltaView")
        .def(
            py::init([](
                         const py::handle vectors,
                         const py::handle logical_ids,
                         const py::handle version_ids,
                         const py::handle begin_seqs,
                         const py::handle ordinals,
                         const py::handle raw_count,
                         const py::handle group_offsets,
                         const py::handle group_ids,
                         const py::handle centers,
                         const py::handle radius_uppers) {
                return std::make_shared<PackedDeltaView>(
                    vectors,
                    logical_ids,
                    version_ids,
                    begin_seqs,
                    ordinals,
                    require_nonnegative_size(raw_count, "raw_count", false),
                    group_offsets,
                    group_ids,
                    centers,
                    radius_uppers);
            }),
            py::arg("vectors"),
            py::arg("logical_ids"),
            py::arg("version_ids"),
            py::arg("begin_seqs"),
            py::arg("ordinals"),
            py::arg("raw_count"),
            py::arg("group_offsets"),
            py::arg("group_ids"),
            py::arg("centers"),
            py::arg("radius_uppers"))
        .def_property_readonly("dimension", &PackedDeltaView::dimension)
        .def_property_readonly("size", &PackedDeltaView::size)
        .def_property_readonly("raw_count", &PackedDeltaView::raw_count)
        .def_property_readonly("group_count", &PackedDeltaView::group_count)
        .def_property_readonly("owned_bytes", &PackedDeltaView::owned_bytes);

    module.def(
        "run_kernel",
        [](const PackedDeltaView &view,
           const py::handle query_object,
           const py::handle candidate_vectors_object,
           const py::handle candidate_logical_ids_object,
           const py::handle candidate_version_ids_object,
           const py::handle candidate_begin_seqs_object,
           const py::handle k_object,
           const double beta,
           const std::string &mode,
           const std::string &threshold_mode,
           const bool audit,
           const bool trusted_prevalidated) {
            const auto query = require_array<float>(query_object, "query", 1);
            const auto candidate_vectors =
                require_array<float>(candidate_vectors_object, "candidate_vectors", 2);
            const auto candidate_logical_ids = require_array<std::int64_t>(
                candidate_logical_ids_object, "candidate_logical_ids", 1);
            const auto candidate_version_ids = require_array<std::int64_t>(
                candidate_version_ids_object, "candidate_version_ids", 1);
            const auto candidate_begin_seqs = require_array<std::int64_t>(
                candidate_begin_seqs_object, "candidate_begin_seqs", 1);

            const std::size_t dimension = shape_size(query.buffer.shape[0], "query");
            if (dimension != view.dimension()) {
                throw py::value_error("query/view dimension mismatch");
            }
            const std::size_t candidate_count =
                shape_size(candidate_vectors.buffer.shape[0], "candidate_vectors");
            if (shape_size(candidate_vectors.buffer.shape[1], "candidate_vectors") !=
                dimension) {
                throw py::value_error("candidate_vectors/query dimension mismatch");
            }
            if (candidate_logical_ids.size() != candidate_count ||
                candidate_version_ids.size() != candidate_count ||
                candidate_begin_seqs.size() != candidate_count) {
                throw py::value_error(
                    "candidate metadata arrays must match candidate_vectors.shape[0]");
            }

            if (trusted_prevalidated) {
                if (!has_immutable_bytes_backing(query.owner) ||
                    !has_immutable_bytes_backing(candidate_vectors.owner) ||
                    !has_immutable_bytes_backing(candidate_logical_ids.owner) ||
                    !has_immutable_bytes_backing(candidate_version_ids.owner) ||
                    !has_immutable_bytes_backing(candidate_begin_seqs.owner)) {
                    throw py::value_error(
                        "trusted_prevalidated inputs require immutable bytes backing");
                }
            } else {
                validate_float_array(query, "query");
                validate_float_array(candidate_vectors, "candidate_vectors");
                std::unordered_set<std::int64_t> candidate_logical_set;
                candidate_logical_set.reserve(candidate_count);
                for (std::size_t row = 0; row < candidate_count; ++row) {
                    const std::int64_t logical_id = candidate_logical_ids.load(row);
                    const std::int64_t version_id = candidate_version_ids.load(row);
                    const std::int64_t begin_seq = candidate_begin_seqs.load(row);
                    if (logical_id < 0 || version_id < 0 || begin_seq < 0) {
                        throw py::value_error(
                            "candidate logical IDs, version IDs, and begin sequences "
                            "must be non-negative");
                    }
                    if (!candidate_logical_set.insert(logical_id).second) {
                        throw py::value_error(
                            "candidate inputs contain duplicate visible logical IDs");
                    }
                }
            }

            const std::size_t k = require_nonnegative_size(k_object, "k", true);
            if (!std::isfinite(beta) || beta < 0.0) {
                throw py::value_error("beta must be finite and non-negative");
            }
            if (mode != "F" && mode != "N" && mode != "P") {
                throw py::value_error("mode must be one of 'F', 'N', or 'P'");
            }
            if (threshold_mode != "heap" && threshold_mode != "rescan") {
                throw py::value_error(
                    "threshold_mode must be either 'heap' or 'rescan'");
            }

            const CandidateInputs inputs{
                query.buffer.ptr,
                candidate_vectors.buffer.ptr,
                candidate_logical_ids.buffer.ptr,
                candidate_version_ids.buffer.ptr,
                candidate_begin_seqs.buffer.ptr,
                candidate_count};
            const std::uint64_t call_index =
                g_call_counter.fetch_add(1, std::memory_order_relaxed) + 1;
            KernelResult result;
            {
                py::gil_scoped_release release;
                result = execute_kernel(
                    view,
                    inputs,
                    k,
                    beta,
                    mode,
                    threshold_mode,
                    audit,
                    trusted_prevalidated,
                    call_index);
            }
            return kernel_result_to_python(result, audit);
        },
        py::arg("view"),
        py::arg("query"),
        py::arg("candidate_vectors"),
        py::arg("candidate_logical_ids"),
        py::arg("candidate_version_ids"),
        py::arg("candidate_begin_seqs"),
        py::arg("k"),
        py::arg("beta"),
        py::arg("mode"),
        py::arg("threshold_mode") = "heap",
        py::arg("audit") = false,
        py::arg("trusted_prevalidated") = false);

    module.def("build_info", &build_info);
    module.def(
        "call_counter",
        []() { return g_call_counter.load(std::memory_order_relaxed); });
    const std::string module_path =
        py::str(module.attr("__file__")).cast<std::string>();
    module.def("shared_object_path", [module_path]() { return module_path; });
}
