#pragma once

#include <Eigen/Core>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/partitioner.h>

// mindir
#include <kindr/minimal/quat-transformation.h>

#include <ros/console.h>
#include <tools/Timing.h>

namespace vdbblox_tools {

typedef float FloatingPoint;
// Transformation type for defining sensor orientation.
typedef kindr::minimal::QuatTransformationTemplate<double> Transformation;
typedef kindr::minimal::RotationQuaternionTemplate<double> Rotation;
typedef kindr::minimal::RotationQuaternionTemplate<FloatingPoint>::Implementation Quaternion;

inline void Eigen2KindrT(Eigen::Matrix4d& input, Transformation& res) {
    Eigen::Matrix3d rotation_matrix = input.block<3, 3>(0, 0);
    res = Transformation(Rotation(rotation_matrix), Eigen::Matrix<double, 3, 1>(input.block<3, 1>(0, 3)));
};

// Constants used across the library.
constexpr float kEpsilon = 1e-6; /**< Used for coordinates. */
constexpr float kFloatEpsilon = 1e-8;    /**< Used for weights. */

struct Color {
    Color() : r(0), g(0), b(0), a(0) {}
    Color(uint8_t _r, uint8_t _g, uint8_t _b) : Color(_r, _g, _b, 255) {}
    Color(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a) : r(_r), g(_g), b(_b), a(_a) {}

    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;

    static Color blendTwoColors(const Color& first_color,
                                FloatingPoint first_weight,
                                const Color& second_color,
                                FloatingPoint second_weight) {
        FloatingPoint total_weight = first_weight + second_weight;

        first_weight /= total_weight;
        second_weight /= total_weight;

        Color new_color;
        new_color.r = static_cast<uint8_t>(
            round(first_color.r * first_weight + second_color.r * second_weight));
        new_color.g = static_cast<uint8_t>(
            round(first_color.g * first_weight + second_color.g * second_weight));
        new_color.b = static_cast<uint8_t>(
            round(first_color.b * first_weight + second_color.b * second_weight));
        new_color.a = static_cast<uint8_t>(
            round(first_color.a * first_weight + second_color.a * second_weight));

        return new_color;
    }

    // Now a bunch of static colors to use! :)
    static const Color White() { return Color(255, 255, 255); }
    static const Color Black() { return Color(0, 0, 0); }
    static const Color Gray() { return Color(127, 127, 127); }
    static const Color Red() { return Color(255, 0, 0); }
    static const Color Green() { return Color(0, 255, 0); }
    static const Color Blue() { return Color(0, 0, 255); }
    static const Color Yellow() { return Color(255, 255, 0); }
    static const Color Orange() { return Color(255, 127, 0); }
    static const Color Purple() { return Color(127, 0, 255); }
    static const Color Teal() { return Color(0, 255, 255); }
    static const Color Pink() { return Color(255, 0, 127); }
};

// Color maps.

/**
 * Maps an input h from a value between 0.0 and 1.0 into a rainbow. Copied from
 * OctomapProvider in octomap.
 */
inline Color rainbowColorMap(double h) {
    Color color;
    color.a = 255;
    // blend over HSV-values (more colors)

    double s = 1.0;
    double v = 1.0;

    h -= floor(h);
    h *= 6;
    int i;
    double m, n, f;

    i = floor(h);
    f = h - i;
    if (!(i & 1)) f = 1 - f;  // if i is even
    m = v * (1 - s);
    n = v * (1 - s * f);

    switch (i) {
        case 6:
        case 0:
            color.r = 255 * v;
            color.g = 255 * n;
            color.b = 255 * m;
            break;
        case 1:
            color.r = 255 * n;
            color.g = 255 * v;
            color.b = 255 * m;
            break;
        case 2:
            color.r = 255 * m;
            color.g = 255 * v;
            color.b = 255 * n;
            break;
        case 3:
            color.r = 255 * m;
            color.g = 255 * n;
            color.b = 255 * v;
            break;
        case 4:
            color.r = 255 * n;
            color.g = 255 * m;
            color.b = 255 * v;
            break;
        case 5:
            color.r = 255 * v;
            color.g = 255 * m;
            color.b = 255 * n;
            break;
        default:
            color.r = 255;
            color.g = 127;
            color.b = 127;
            break;
    }

    return color;
}

/// Maps an input h from a value between 0.0 and 1.0 into a grayscale color.
inline Color grayColorMap(double h) {
    Color color;
    color.a = 255;

    color.r = round(h * 255);
    color.b = color.r;
    color.g = color.r;

    return color;
}

inline Color randomColor() {
    Color color;

    color.a = 255;

    color.r = rand() % 256;  // NOLINT
    color.b = rand() % 256;  // NOLINT
    color.g = rand() % 256;  // NOLINT

    return color;
}

class ColorMap {
public:
    ColorMap() : min_value_(0.0), max_value_(1.0) {}
    virtual ~ColorMap() {}

    void setMinValue(float min_value) { min_value_ = min_value; }

    void setMaxValue(float max_value) { max_value_ = max_value; }

    virtual Color colorLookup(float value) const = 0;

protected:
    float min_value_;
    float max_value_;
};

class GrayscaleColorMap : public ColorMap {
public:
    virtual Color colorLookup(float value) const {
        float new_value = std::min(max_value_, std::max(min_value_, value));
        new_value = (new_value - min_value_) / (max_value_ - min_value_);
        return grayColorMap(new_value);
    }
};

class InverseGrayscaleColorMap : public ColorMap {
public:
    virtual Color colorLookup(float value) const {
        float new_value = std::min(max_value_, std::max(min_value_, value));
        new_value = (new_value - min_value_) / (max_value_ - min_value_);
        return grayColorMap(1.0 - new_value);
    }
};

class RainbowColorMap : public ColorMap {
public:
    virtual Color colorLookup(float value) const {
        float new_value = std::min(max_value_, std::max(min_value_, value));
        new_value = (new_value - min_value_) / (max_value_ - min_value_);
        return rainbowColorMap(new_value);
    }
};

class InverseRainbowColorMap : public ColorMap {
public:
    virtual Color colorLookup(float value) const {
        float new_value = std::min(max_value_, std::max(min_value_, value));
        new_value = (new_value - min_value_) / (max_value_ - min_value_);
        return rainbowColorMap(1.0 - new_value);
    }
};

class IronbowColorMap : public ColorMap {
public:
    IronbowColorMap() : ColorMap() {
        palette_colors_.push_back(Color(0, 0, 0));
        palette_colors_.push_back(Color(145, 20, 145));
        palette_colors_.push_back(Color(255, 138, 0));
        palette_colors_.push_back(Color(255, 230, 40));
        palette_colors_.push_back(Color(255, 255, 255));
        // Add an extra to avoid overflow.
        palette_colors_.push_back(Color(255, 255, 255));

        increment_ = 1.0 / (palette_colors_.size() - 2);
    }

    virtual Color colorLookup(float value) const {
        float new_value = std::min(max_value_, std::max(min_value_, value));
        new_value = (new_value - min_value_) / (max_value_ - min_value_);

        size_t index = static_cast<size_t>(std::floor(new_value / increment_));

        return Color::blendTwoColors(palette_colors_[index], increment_ * (index + 1) - new_value,
                                     palette_colors_[index + 1], new_value - increment_ * (index));
    }

protected:
    std::vector<Color> palette_colors_;
    float increment_;
};

class IdColorMap {
public:
    IdColorMap() {}
    virtual ~IdColorMap() {}

    virtual Color colorLookup(size_t value) const = 0;
};
/**
 * Map unique IDs from [0, Inf[ to unique colors that have constant (irrational)
 * spacing and high spread on the color wheel.
 * An important advantage of this color map is that the colors are independent
 * of the max ID value, e.g. previous colors don't change when adding new IDs.
 */
class IrrationalIdColorMap : IdColorMap {
public:
    IrrationalIdColorMap() : irrational_base_(3.f * M_PI) {}

    void setIrrationalBase(float value) { irrational_base_ = value; }

    virtual Color colorLookup(const size_t value) const {
        const float normalized_color = std::fmod(value / irrational_base_, 1.f);
        return rainbowColorMap(normalized_color);
    }

private:
    float irrational_base_;
};
/**
 * Map unique IDs from [0, Inf] to unique colors
 * with piecewise constant spacing and higher spread on the color wheel.
 * An important advantage of this color map is that the colors are independent
 * of the max ID value, e.g. previous colors don't change when adding new IDs.
 */
class ExponentialOffsetIdColorMap : IdColorMap {
public:
    ExponentialOffsetIdColorMap() : items_per_revolution_(10u) {}

    void setItemsPerRevolution(uint value) { items_per_revolution_ = value; }

    virtual Color colorLookup(const size_t value) const {
        const size_t revolution = value / items_per_revolution_;
        const float progress_along_revolution = std::fmod(value / items_per_revolution_, 1.f);
        // NOTE: std::modf could be used to simultaneously get the integer and
        //       fractional parts, but the above code is assumed to be more readable

        // Calculate the offset if appropriate
        float offset = 0;
        if (items_per_revolution_ < value + 1u) {
            const size_t current_episode = std::floor(std::log2(revolution));
            const size_t episode_start = std::exp2(current_episode);
            const size_t episode_num_subdivisions = episode_start;
            const size_t current_subdivision = revolution - episode_start;
            const float subdivision_step_size =
                1 / (items_per_revolution_ * 2 * episode_num_subdivisions);
            offset = (2 * current_subdivision + 1) * subdivision_step_size;
        }

        const float normalized_color = progress_along_revolution + offset;
        return rainbowColorMap(normalized_color);
    }

private:
    float items_per_revolution_;
};


/// Check if all coordinates in the PCL point are finite.
template <typename PCLPoint>
inline bool isPointFinite(const PCLPoint& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

template <typename PCLPoint>
Color convertColor(const PCLPoint& point, const std::shared_ptr<ColorMap>& color_map);

template <>
inline Color convertColor(const pcl::PointXYZRGBL& point, const std::shared_ptr<ColorMap>& /*color_map*/) {
    return Color(point.r, point.g, point.b);
}

template <>
inline Color convertColor(const pcl::PointXYZRGB& point, const std::shared_ptr<ColorMap>& /*color_map*/) {
    return Color(point.r, point.g, point.b, point.a);
}

template <>
inline Color convertColor(const pcl::PointXYZI& point, const std::shared_ptr<ColorMap>& color_map) {
    return color_map->colorLookup(point.intensity);
}

template <>
inline Color convertColor(const pcl::PointXYZ& /*point*/, const std::shared_ptr<ColorMap>& color_map) {
    return color_map->colorLookup(0);
}

template <typename Type>
using AlignedVector = tbb::concurrent_vector<Type, Eigen::aligned_allocator<Type>>;
typedef float FloatingPoint; ;
typedef Eigen::Matrix<FloatingPoint, 3, 1> Point;
typedef AlignedVector<Point> PointCloud;
typedef AlignedVector<vdbblox_tools::Color> Colors;

/// Convert pointclouds of different PCL types to a voxblox pointcloud.
template <typename PCLPoint>
inline void convertPointcloud(const typename pcl::PointCloud<PCLPoint>& pointcloud_pcl, const std::shared_ptr<ColorMap>& color_map, 
                                vdbblox_tools::PointCloud* points_C, vdbblox_tools::Colors* colors) {
    points_C->reserve(pointcloud_pcl.size());
    colors->reserve(pointcloud_pcl.size());

    for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
        // check isPointFinite
        if (!isPointFinite(pointcloud_pcl.points[i])) continue;

        points_C->push_back(Point(pointcloud_pcl.points[i].x, pointcloud_pcl.points[i].y, pointcloud_pcl.points[i].z));
        colors->emplace_back(convertColor<PCLPoint>(pointcloud_pcl.points[i], color_map));
    }
}

template <typename Scalar, typename PointType>
inline void PCL2Eigen(const pcl::PointCloud<PointType>& ptcloud_pcl, std::vector<Eigen::Matrix<Scalar, 3, 1>>& ptcloud_eig) {
    ptcloud_eig.clear();
    ptcloud_eig.reserve(ptcloud_pcl.size());
    for (const auto point : ptcloud_pcl) {
        if (!isPointFinite(point)) continue;
        Eigen::Matrix<Scalar, 3, 1> pt;
        pt(0) = static_cast<Scalar>(point.x);
        pt(1) = static_cast<Scalar>(point.y);
        pt(2) = static_cast<Scalar>(point.z);
        ptcloud_eig.push_back(pt);
    }
}

inline int getMaxNumIdealThreads() {
    // get the total number of threads supported by the system
    int totalThreads = std::thread::hardware_concurrency();
    if (totalThreads == 0) {
        std::cerr << "Can not Get The Total Threads!" << std::endl;
        return 0;
    }

    // get the average load over the last 1 minute
    double load[3];  // store the 1, 5, and 15 minute load averages
    if (getloadavg(load, 3) == -1) {
        std::cerr << "Can not Get The System Load!" << std::endl;
        return 0;
    }

    // estimate the number of idle threads
    int idleThreads = std::max(0, totalThreads - static_cast<int>(std::ceil(load[0])));
    return idleThreads;
}

}  // namespace vdbblox_tools