#pragma once

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <ostream>  
#include <Eigen/Core>

namespace vdbblox_core {

struct TSDFCell {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TSDFCell(const float& tsdf_, const float& weight_, const Eigen::Vector3f& gradient_)
        :tsdf(tsdf_), weight(weight_), gradient(gradient_) {}

    explicit TSDFCell(float val = 0)
        : tsdf(val), weight(val), gradient(val, val, val) {}

    float tsdf = 0.0f;
    float weight = 0.0f;
    Eigen::Vector3f gradient{0, 0, 0};
};

inline std::ostream& operator<<(std::ostream& os, const TSDFCell& data) {
    os << "{position: " << data.gradient
       << ", tsdf: " << data.tsdf
       << ", color: " << data.weight << "}";
    return os;
}

inline TSDFCell operator-(const TSDFCell& lhs, const TSDFCell& rhs) {
    return TSDFCell{ 0 /* tsdf */, 0 /* weight */, Eigen::Vector3f(0, 0, 0) };
}

inline TSDFCell operator+(const TSDFCell& lhs, const TSDFCell& rhs) {
    return TSDFCell{ 0 /* tsdf */, 0 /* weight */, Eigen::Vector3f(0, 0, 0) };
}

inline TSDFCell operator-(const TSDFCell& data) {
    return TSDFCell{ 0 /* tsdf */, 0 /* weight */, Eigen::Vector3f(0, 0, 0) };
}

inline TSDFCell operator+(const TSDFCell& data, float scalar) {
    return TSDFCell{ 0 /* tsdf */, 0 /* weight */, Eigen::Vector3f(0, 0, 0) };
}

inline TSDFCell operator+(float scalar, const TSDFCell& data) {
    return data + scalar;
}

inline TSDFCell Abs(const TSDFCell& data) {
    return TSDFCell{ 0/* tsdf */, 0/* weight */, Eigen::Vector3f(0, 0, 0) };
}

inline bool operator==(const TSDFCell& lhs, const TSDFCell& rhs) {
    return lhs.gradient == rhs.gradient &&
           lhs.tsdf == rhs.tsdf &&
           lhs.weight == rhs.weight;
}

inline bool operator!=(const TSDFCell& lhs, const TSDFCell& rhs) {
    return !(lhs == rhs);
}

inline bool operator<(const TSDFCell& lhs, const TSDFCell& rhs) {
    if (lhs.tsdf != rhs.tsdf)
        return lhs.tsdf < rhs.tsdf;
    return lhs.weight < rhs.weight;
}

inline bool operator>(const TSDFCell& lhs, const TSDFCell& rhs) {
    return rhs < lhs;
}

inline bool operator<=(const TSDFCell& lhs, const TSDFCell& rhs) {
    return !(rhs < lhs);
}

inline bool operator>=(const TSDFCell& lhs, const TSDFCell& rhs) {
    return !(lhs < rhs);
}

}

#include <oneapi/tbb/concurrent_vector.h>
namespace vdbblox_core {
struct Voxel {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    openvdb::Coord coord;
    float tsdf, weight;
    Eigen::Vector3f gradient;
};

using BandVoxelVector = tbb::concurrent_vector<Voxel, Eigen::aligned_allocator<Voxel>>;

}
