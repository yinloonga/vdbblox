#pragma once

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <climits>
#include <cstdint>
#include <ostream>  
#include <Eigen/Core>
namespace vdbblox_core {

struct ESDFCell {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ESDFCell(const float& raw_distance_, const float& distance_, const bool& behind_, const bool& newly_, const bool& observed_, const bool& occupied_,
                const std::uint32_t& coc_idx_, const std::uint32_t& prev_idx_, const std::uint32_t& next_idx_, const std::uint32_t& head_idx_, const std::uint32_t& self_idx_)
        :raw_distance(raw_distance_), distance(distance_), behind(behind_), newly(newly_), observed(observed_), occupied(occupied_),
         coc_idx(coc_idx_), prev_idx(prev_idx_), next_idx(next_idx_), head_idx(head_idx_), self_idx(self_idx_) {}

    explicit ESDFCell(const bool& val = false)
        : raw_distance(-100.0f), distance(-100.0f), behind(val), newly(val), observed(val), occupied(val),
          coc_idx(UINT_MAX), prev_idx(UINT_MAX), next_idx(UINT_MAX), head_idx(UINT_MAX), self_idx(UINT_MAX) {}

    float raw_distance = -100.0f, distance = -100.0f;
    bool /* fixed = false, */ behind = false, newly = false, observed = false, occupied = false;
    std::uint32_t coc_idx = UINT_MAX, prev_idx = UINT_MAX, next_idx = UINT_MAX, head_idx = UINT_MAX, self_idx = UINT_MAX;
};

inline std::ostream& operator<<(std::ostream& os, const ESDFCell& data) {
    os << "{self_idx: " << data.self_idx
       << ", raw_distance: " << data.raw_distance
       << ", distance: " << data.distance
       << ", behind: " << data.behind
       << ", newly: " << data.newly
       << ", observed: " << data.observed
       << ", coc_idx: " << data.coc_idx
       << ", prev_idx: " << data.prev_idx
       << ", next_idx: " << data.next_idx
       << ", head_idx: " << data.head_idx << "}";

    return os;
}

inline ESDFCell operator-(const ESDFCell& lhs, const ESDFCell& rhs) {
    return ESDFCell{
        lhs.raw_distance - rhs.raw_distance,
        lhs.distance - rhs.distance,
        0/* behind */, 0/* newly */, 0/* observed */, 0/* occupied */,
        0/* coc_idx */, 0/* prev_idx */, 0/* next_idx */, 0/* head_idx */, 0/* self_idx */
    };
}

inline ESDFCell operator+(const ESDFCell& lhs, const ESDFCell& rhs) {
    return ESDFCell{
        lhs.raw_distance + rhs.raw_distance,
        lhs.distance + rhs.distance,
        0/* behind */, 0/* newly */, 0/* observed */, 0/* occupied */,
        0/* coc_idx */, 0/* prev_idx */, 0/* next_idx */, 0/* head_idx */, 0/* self_idx */
    };
}

inline ESDFCell operator-(const ESDFCell& data) {
    return ESDFCell{
        -data.raw_distance,
        -data.distance,
        0/* behind */, 0/* newly */, 0/* observed */, 0/* occupied */,
        0/* coc_idx */, 0/* prev_idx */, 0/* next_idx */, 0/* head_idx */, 0/* self_idx */
    };
}

inline ESDFCell operator+(const ESDFCell& data, float scalar) {
    return ESDFCell{
        data.raw_distance + static_cast<float>(scalar),
        data.distance + static_cast<float>(scalar),
        0/* behind */, 0/* newly */, 0/* observed */, 0/* occupied */,
        0/* coc_idx */, 0/* prev_idx */, 0/* next_idx */, 0/* head_idx */, 0/* self_idx */
    };
}

inline ESDFCell operator+(float scalar, const ESDFCell& data) {
    return data + scalar;
}

inline ESDFCell Abs(const ESDFCell& data) {
    return ESDFCell{
        openvdb::math::Abs(data.raw_distance),
        openvdb::math::Abs(data.distance),
        0/* behind */, 0/* newly */, 0/* observed */, 0/* occupied */,
        0/* coc_idx */, 0/* prev_idx */, 0/* next_idx */, 0/* head_idx */, 0/* self_idx */
    };
}

inline bool operator==(const ESDFCell& lhs, const ESDFCell& rhs) {
    return lhs.self_idx == rhs.self_idx;
}

inline bool operator!=(const ESDFCell& lhs, const ESDFCell& rhs) {
    return !(lhs == rhs);
}

inline bool operator<(const ESDFCell& lhs, const ESDFCell& rhs) {
    return lhs.distance < rhs.distance;
}

inline bool operator>(const ESDFCell& lhs, const ESDFCell& rhs) {
    return rhs < lhs;
}

inline bool operator<=(const ESDFCell& lhs, const ESDFCell& rhs) {
    return !(rhs < lhs);
}

inline bool operator>=(const ESDFCell& lhs, const ESDFCell& rhs) {
    return !(lhs < rhs);
}

}