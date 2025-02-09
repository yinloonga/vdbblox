#pragma  once

#include <libmorton/morton.h>
#include <openvdb/Types.h>
#include <ros/publisher.h>
#include <ros/ros.h>
#include <openvdb/openvdb.h>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <utility>

#include "TSDFMap.h"
#include "cell/ESDFCell.hpp"
#include "ros/node_handle.h"

namespace vdbblox_core {
struct ESDFConfig {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // output log or not
    bool verbose = true;

    float voxel_size_;
    
    // Minimum weight to consider a TSDF value seen at.
    float min_weight = 1e-6;
    /**
     * Use finer ESDF with the consideration of the inner voxel distance
     * from the voxel center to the actual surface.
     * the non-projective TSDF and signed distance gradient are needed
     * for the calculation.
     */
    bool finer_esdf_on = false;

    /**
     * When judge a voxel is occupied or not,
     * the threshold of TSDF distance is occ_voxel_size_ratio * voxel size
     */
    float occ_voxel_size_ratio = 0.865;  // Sqrt(3) / 2
    /**
     * Maximum distance to calculate the actual distance to.
     * Any values above this will be set to infinity_distance_m.
     */
    float max_distance_m = 10.0;

    // Default distance set for unknown values and values > max_distance_m.
    float infinity_distance_m = 2.0; // infinity_

    // Trunction distance behind surface, unit: m
    float max_behind_surface_m = 1.0f;

    // Fixed band distance threshold, unit: m
    float band_distance_m = 1.0f;

    // use a fixed band for esdf to directly copy the tsdf value
    bool fixed_band_esdf_on = true;

    float sdf_trunc_ = 0.6f;

    // update range
    float update_range_ = 5.0f, update_range_z_ = 2.0f;
    Eigen::Vector3f left_bottom_, right_top_;

    // udpate the free space
    bool space_carving_ = false;
    bool update_local_ = false;
    float offset_x_, offset_y_, offset_z_;

};

using ESDFVDBTree = openvdb::tree::Tree4<vdbblox_core::ESDFCell, 5, 4, 3>::Type;
using ESDFVDBGrid = openvdb::Grid<ESDFVDBTree>;

class ESDFMap {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ESDFMap(ros::NodeHandle& nh, std::shared_ptr<TSDFMap> pTSDFMap);
    ~ESDFMap();

    void Integrate(const ros::TimerEvent& event);
    void updateFreeSpace();
    void updateESDFMap();
    bool processInsertList();
    bool processDeleteList();
    bool processUpdateList();

    inline vdbblox_core::ESDFCell& CoordEncoder(const openvdb::Coord& coord) {
        std::unique_lock<std::shared_mutex> lock(mutex_update_esdf_);
        if (!pESDFVDBGridAcc_->isValueOn(coord)) {
            pESDFVDBGridAcc_->setValue(coord, vdbblox_core::ESDFCell(false));
            vdbblox_core::ESDFCell& esdf_cell = const_cast<vdbblox_core::ESDFCell&>(pESDFVDBGridAcc_->getValue(coord));
            std::uint32_t morton_code = libmorton::morton3D_32_encode(coord.x(), coord.y(), coord.z());
            esdf_cell.self_idx = morton_code;
            esdf_cell.raw_distance = esdf_cfg_.infinity_distance_m;
            return esdf_cell;
        }
        return const_cast<vdbblox_core::ESDFCell&>(pESDFVDBGridAcc_->getValue(coord));
    }

    inline vdbblox_core::ESDFCell& CoordDecoder(const openvdb::Coord& coord, const bool& needCheckState = false, const bool& needCheckIdx = false) {
        std::unique_lock<std::shared_mutex> lock(mutex_update_esdf_);
        if (needCheckState) {
            if (!pESDFVDBGridAcc_->isValueOn(coord)) {
                // ROS_ERROR("[CoordDecoder] The current voxel in DLL is not activate");
                return default_esdf_cell_;
            }
        }

        vdbblox_core::ESDFCell& esdf_cell = const_cast<vdbblox_core::ESDFCell&>(pESDFVDBGridAcc_->getValue(coord));
        if (needCheckIdx) {
            if (esdf_cell.self_idx == UINT_MAX) {
                // ROS_ERROR("[CoordDecoder] The current voxel in DLL is UINT_MAX");
                return default_esdf_cell_;
            }
        }
        return esdf_cell;   
    }
    
    inline std::pair<openvdb::Coord, vdbblox_core::ESDFCell&> IndexDecoder(const std::size_t& idx, const bool& needCheckState = false, const bool& needCheckIdx = false) {
        std::unique_lock<std::shared_mutex> lock(mutex_update_esdf_);
        std::uint64_t coord_x_, coord_y_, coord_z_;
        libmorton::morton3D_32_decode(idx, coord_x_, coord_y_, coord_z_);
        openvdb::Coord coord(coord_x_, coord_y_, coord_z_);
        if (needCheckState) {
            if (!pESDFVDBGridAcc_->isValueOn(coord)) {
                ROS_ERROR("[IndexDecoder] The current voxel in DLL is not activate");
            }
        }
        vdbblox_core::ESDFCell& esdf_cell = const_cast<vdbblox_core::ESDFCell&>(pESDFVDBGridAcc_->getValue(coord));
        if (needCheckIdx) {
            if (esdf_cell.self_idx == UINT_MAX) {
                ROS_ERROR("[IndexDecoder] The current voxel in DLL is UINT_MAX");
            }
        }
        return {coord, esdf_cell};   
    }

    inline bool isObserved(float weight) {
        return weight > esdf_cfg_.min_weight;        
    }

    inline bool isOccupied(const float& dist_m) const {
        return std::abs(dist_m) <= esdf_cfg_.occ_voxel_size_ratio * esdf_cfg_.voxel_size_;
    }

    inline bool isOccupied(const float& dist_m, const Eigen::Vector3f& grad) const {
        if (grad.norm() > vdbblox_tools::kFloatEpsilon) {
            const Eigen::Vector3f dist_on_axis = dist_m * grad;
            bool is_occupied = true;
            for (int i = 0; i < 3; ++i) {
                if (std::abs(dist_on_axis(i)) > 0.5 * esdf_cfg_.voxel_size_) {
                    is_occupied = false;
                    break;
                }
            }
            return is_occupied;
        } else {
            return isOccupied(dist_m);
        }
    }

    inline bool isFixed(const float& dist_m) const {
        return std::abs(dist_m) <= esdf_cfg_.sdf_trunc_;
    }

    inline float getDistance(const openvdb::Coord& lhs, const openvdb::Coord& rhs) {
        return (lhs.asVec3s() - rhs.asVec3s()).length() * esdf_cfg_.voxel_size_;
    }

    void deleteFromList(vdbblox_core::ESDFCell& occ_cell, vdbblox_core::ESDFCell& cur_cell);
    void insertIntoList(vdbblox_core::ESDFCell& occ_cell, vdbblox_core::ESDFCell& cur_cell);
    bool setUpdateRange();
    bool isInRange(const openvdb::Coord& coord);
    void pushPointsIntoQueue(const openvdb::Coord& coord);
    openvdb::Coord& getESDFMapPointsFromQueue();
    void popESDFMapPointsFromQueue();
    bool isEmtpyESDFMapPointsInQueue();
public:
    // Integrate
    std::mutex mtx_integrate_;

    std::shared_mutex mutex_update_esdf_;
    ros::NodeHandle nh_;
    std::shared_ptr<TSDFMap> pTSDFMap_;
    ESDFConfig esdf_cfg_;
    // ESDF Grid
    ESDFVDBGrid::Ptr pESDFVDBGrid_;
    std::shared_ptr<ESDFVDBGrid::Accessor> pESDFVDBGridAcc_;
    vdbblox_core::ESDFCell default_esdf_cell_;

    std::list<openvdb::Coord> insert_list_;
    std::list<openvdb::Coord> delete_list_;
    std::multimap<float/* distance */, openvdb::Coord/* self_idx */> update_list_;
    std::deque<std::pair<openvdb::Coord/* coordinate */, float/* distance */>> update_tsdfmap_;

    // UpdateESDFMap
    std::mutex mutex_process_insert_, mutex_process_delete_, mutex_process_update_;
    std::size_t cnt_process_insert_ = 0, cnt_process_delete_ = 0, cnt_process_update_ = 0;

    // Update FreeSpace
    std::thread thread_update_free_space_;
    std::atomic<bool> should_exit_ = false;
    std::size_t range_num_voxles_ = 0;

    // Visualization
    std::condition_variable cv_update_done_;
    bool update_done_ = false;

    static const int num_dirs_ = 24;  // faces 2 steps
    const openvdb::math::Coord dirs_[num_dirs_] = {
                                                    openvdb::math::Coord(-1, 0, 0),  openvdb::math::Coord(1, 0, 0),   openvdb::math::Coord(0, -1, 0),
                                                    openvdb::math::Coord(0, 1, 0),   openvdb::math::Coord(0, 0, -1),  openvdb::math::Coord(0, 0, 1),

                                                    openvdb::math::Coord(-1, -1, 0), openvdb::math::Coord(1, 1, 0),   openvdb::math::Coord(0, -1, -1),
                                                    openvdb::math::Coord(0, 1, 1),   openvdb::math::Coord(-1, 0, -1), openvdb::math::Coord(1, 0, 1),
                                                    openvdb::math::Coord(-1, 1, 0),  openvdb::math::Coord(1, -1, 0),  openvdb::math::Coord(0, -1, 1),
                                                    openvdb::math::Coord(0, 1, -1),  openvdb::math::Coord(1, 0, -1),  openvdb::math::Coord(-1, 0, 1),

                                                    openvdb::math::Coord(-2, 0, 0),  openvdb::math::Coord(2, 0, 0),   openvdb::math::Coord(0, -2, 0),
                                                    openvdb::math::Coord(0, 2, 0),   openvdb::math::Coord(0, 0, -2),  openvdb::math::Coord(0, 0, 2)};
};
}  // namespace vdbblox