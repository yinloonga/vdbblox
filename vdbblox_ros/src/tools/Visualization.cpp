#include <atomic>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>

#include <geometry_msgs/Point32.h>
#include <sensor_msgs/PointCloud.h>

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>

#include "cell/TSDFCell.hpp"
#include "core/ESDFMap.h"
#include "core/TSDFMap.h"
#include "ros/console.h"
#include "ros/node_handle.h"
#include "tools/Timing.h"
#include "tools/Visualization.h"
#include <tools/common.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

vdbblox_tools::Visualization::Visualization(ros::NodeHandle& nh, std::shared_ptr<vdbblox_core::TSDFMap> pTSDFMap, std::shared_ptr<vdbblox_core::ESDFMap> pESDFMap)
    :nh_(nh), pTSDFMap_(pTSDFMap), pESDFMap_(pESDFMap) {
    // Constructor
    ROS_INFO("Visualization Constructor");

    vis_cfg_.voxel_size_ = pTSDFMap_->tsdf_cfg_.voxel_size_;
    vis_cfg_.use_fpv = pTSDFMap_->tsdf_cfg_.use_fpv;
    vis_cfg_.infinity_distance_m = pESDFMap_->esdf_cfg_.infinity_distance_m; 
    vis_cfg_.update_range_ = pTSDFMap_->tsdf_cfg_.update_range_;
    vis_cfg_.update_range_z_ = pTSDFMap_->tsdf_cfg_.update_range_z_;

    vis_cfg_.offset_x_ = pTSDFMap_->tsdf_cfg_.offset_x_;
    vis_cfg_.offset_y_ = pTSDFMap_->tsdf_cfg_.offset_y_;
    vis_cfg_.offset_z_ = pTSDFMap_->tsdf_cfg_.offset_z_;

    nh_.getParam("/updateESDF", vis_cfg_.updateESDF_);
    nh_.getParam("/vis_min_range", vis_cfg_.min_range_);
    nh_.getParam("/vis_max_range", vis_cfg_.max_range_);
    nh_.getParam("/sdf_trunc", vis_cfg_.sdf_trunc_);
    nh_.getParam("/slice_vis_level", vis_cfg_.slice_vis_level_);

    // publish global point clouds
    tsdf_pub_ = nh_.advertise<sensor_msgs::PointCloud>("tsdf_pointcloud", 1);
    esdf_pub_ = nh_.advertise<sensor_msgs::PointCloud>("esdf_pointcloud", 1);
    slice_pub_ = nh_.advertise<visualization_msgs::Marker>("esdf_slice", 1);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("update_range", 1);

    // TSDF VDB (deep copy)
    pTSDFVDBGridCopy_ = vdbblox_core::TSDFVDBGrid::create(vdbblox_core::TSDFCell());
    pTSDFVDBGridCopy_->setName("TSDFVDBGridCopy");
    pTSDFVDBGridCopy_->setTransform(openvdb::math::Transform::createLinearTransform(pTSDFMap_->tsdf_cfg_.voxel_size_));
    pTSDFVDBGridCopy_->setGridClass(openvdb::GRID_LEVEL_SET);
    pTSDFVDBGridAccCopy_ = std::make_shared<vdbblox_core::TSDFVDBGrid::Accessor>(pTSDFVDBGridCopy_->getAccessor());

    // ESDF VDB (deep copy)
    pESDFVDBGridCopy_ = vdbblox_core::ESDFVDBGrid::create(vdbblox_core::ESDFCell(false));
    pESDFVDBGridCopy_->setName("ESDFVDBGridCopy");
    pESDFVDBGridCopy_->setTransform(openvdb::math::Transform::createLinearTransform(pTSDFMap_->tsdf_cfg_.voxel_size_));
    pESDFVDBGridCopy_->setGridClass(openvdb::GRID_LEVEL_SET);
    pESDFVDBGridAccCopy_ = std::make_shared<vdbblox_core::ESDFVDBGrid::Accessor>(pESDFVDBGridCopy_->getAccessor());

    // slice
    pESDFVDBGridSlice_ = vdbblox_core::ESDFVDBGrid::create(vdbblox_core::ESDFCell(false));
    pESDFVDBGridSlice_->setName("ESDFVDBGridSlice");
    pESDFVDBGridSlice_->setTransform(openvdb::math::Transform::createLinearTransform(pTSDFMap_->tsdf_cfg_.voxel_size_));
    pESDFVDBGridSlice_->setGridClass(openvdb::GRID_LEVEL_SET);
    pESDFVDBGridAccSlice_ = std::make_shared<vdbblox_core::ESDFVDBGrid::Accessor>(pESDFVDBGridSlice_->getAccessor());

    thread_update_tsdf_ = std::thread(&vdbblox_tools::Visualization::publishTSDFPointCloud, this);
    if (vis_cfg_.updateESDF_) thread_update_esdf_slice_ = std::thread(&vdbblox_tools::Visualization::publishESDFMapSlice, this);

}

vdbblox_tools::Visualization::~Visualization() {
    // Destructor
    
}

void vdbblox_tools::Visualization::Visualize(const ros::TimerEvent& event) {
    if (!pTSDFMap_->tsdf_cfg_.init_mapping_) return;

    std::future<void> future_tsdf = std::async(std::launch::async, &vdbblox_tools::Visualization::publishTSDFPointCloud, this);
    // std::future<bool> future_slice = std::async(std::launch::async, &vdbblox_tools::Visualization::publishESDFMapSlice, this);

    future_tsdf.get();
    return;
}

void vdbblox_tools::Visualization::publishTSDFPointCloud() {
    while (!should_exit_tsdf_.load()) {
        // wait
        std::unique_lock<std::mutex> tsdf_pub(mutex_tsdf_pub_);
        pTSDFMap_->cv_update_done_.wait(tsdf_pub, [&] { return pTSDFMap_->update_done_ || should_exit_tsdf_.load(); });
        if (should_exit_tsdf_.load()) break;
        pTSDFMap_->update_done_ = false;

        if (tsdf_pub_.getNumSubscribers() <= 0) return;

        vdbblox_tools::Timer publishTSDFPointCloud("TSDF-vizPublishTSDFPointCloud");
        float offset_x = vis_cfg_.offset_x_;
        float offset_y = vis_cfg_.offset_y_;
        float offset_z = vis_cfg_.offset_z_;
        while (true) {
            if (pTSDFMap_->mutex_update_tsdf_.try_lock_shared()) {  // try to get the shared lock
                pTSDFVDBGridCopy_ = pTSDFMap_->pTSDFVDBGrid_->deepCopy();
                pTSDFMap_->mutex_update_tsdf_.unlock_shared();
                break;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));  // sleep for 1ms
            }
        }
        geometry_msgs::Point32 p;
        sensor_msgs::PointCloud pcd;
        pcd.header.frame_id = pTSDFMap_->pTransformer_->getWorldframe();
        pcd.points.clear();
        int num_voxels = pTSDFVDBGridCopy_->activeVoxelCount();
        pcd.points.reserve(num_voxels);

        for (vdbblox_core::TSDFVDBGrid::ValueOnCIter iter = pTSDFVDBGridCopy_->cbeginValueOn(); iter.test(); iter.next()) {
            if (!iter.isVoxelValue()) continue;

            const vdbblox_core::TSDFCell& tsdf_cell = iter.getValue();
            if (std::abs(tsdf_cell.tsdf) > vis_cfg_.voxel_size_) continue;

            const openvdb::Coord& coord = iter.getCoord();
            const openvdb::Vec3d& pos = pTSDFVDBGridCopy_->transform().indexToWorld(coord);
            p.x = pos.x() - offset_x; p.y = pos.y() - offset_y; p.z = pos.z() - offset_z;
            pcd.points.emplace_back(p);
        }
        tsdf_pub_.publish(pcd);

        pTSDFVDBGridCopy_->clear();
        publishTSDFPointCloud.Stop();
    }
}

bool vdbblox_tools::Visualization::publishESDFPointCloud() {
    std::unique_lock<std::mutex> esdf_pub(mutex_esdf_pub_);
    if (esdf_pub_.getNumSubscribers() <= 0) return false;

    vdbblox_tools::Timer publishESDFPointCloud("vizPublishESDFPointCloud");
    sensor_msgs::PointCloud pcd;
    pcd.header.frame_id = pTSDFMap_->pTransformer_->getWorldframe();
    pcd.points.clear();

    geometry_msgs::Point32 p;
    float offset_x = vis_cfg_.offset_x_;
    float offset_y = vis_cfg_.offset_y_;
    float offset_z = vis_cfg_.offset_z_;

    // vdbblox_tools::Timer publishESDFPointCloudCopy("publishESDFPointCloud [Copy]");
    {
        std::shared_lock<std::shared_mutex> lock_esdf(pESDFMap_->mutex_update_esdf_);
        pESDFVDBGridCopy_ = pESDFMap_->pESDFVDBGrid_->deepCopy();  // 300:30 (ms)
    }
    // publishESDFPointCloudCopy.Stop();

    std::size_t num_voxels = pESDFVDBGridCopy_->activeVoxelCount();
    // pcd.points.reserve(num_voxels); 
    pcd.points.resize(num_voxels);
    std::atomic<std::size_t> cnt(0);

    for (vdbblox_core::ESDFVDBGrid::ValueOnCIter iter = pESDFVDBGridCopy_->cbeginValueOn(); iter.test(); iter.next()) {
        if (!iter.isVoxelValue()) continue;

        const vdbblox_core::ESDFCell& esdf_cell = iter.getValue();
        if (std::abs(esdf_cell.distance) > vis_cfg_.sdf_trunc_) continue;
        if (!esdf_cell.newly) continue;

        const openvdb::Coord& coord = iter.getCoord();
        const openvdb::Vec3d& pos = pESDFVDBGridCopy_->transform().indexToWorld(coord);
        p.x = pos.x() - offset_x; p.y = pos.y() - offset_y; p.z = pos.z() - offset_z;
        // pcd.points.emplace_back(p);
        pcd.points[cnt.fetch_add(1)] = p;
    }
    pcd.points.resize(cnt.load());
    esdf_pub_.publish(pcd);

    pESDFVDBGridCopy_->clear();
    publishESDFPointCloud.Stop();

    return true;
}


void vdbblox_tools::Visualization::publishESDFMapSlice() {
    while (!should_exit_esdf_.load()) {

        // wait
        std::unique_lock<std::mutex> slice_pub(mutex_slice_pub_);
        pESDFMap_->cv_update_done_.wait(slice_pub, [&] { return pESDFMap_->update_done_ || should_exit_esdf_.load(); });
        if (should_exit_esdf_.load()) break;
        pESDFMap_->update_done_ = false;

        if (slice_pub_.getNumSubscribers() <= 0) continue;

        vdbblox_tools::Timer publishESDFMapSlice("updateESDFMap-vizPublishESDFMapSlice");
        visualization_msgs::Marker marker;
        marker.header.frame_id = pTSDFMap_->pTransformer_->getWorldframe();
        marker.id = 100;
        marker.type = visualization_msgs::Marker::POINTS;
        marker.action = visualization_msgs::Marker::MODIFY;
        marker.scale.x = vis_cfg_.voxel_size_;
        marker.scale.y = vis_cfg_.voxel_size_;
        marker.scale.z = vis_cfg_.voxel_size_;
        marker.pose.orientation.w = 1.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;

        {
            std::shared_lock<std::shared_mutex> lock(pESDFMap_->mutex_update_esdf_);
            pESDFVDBGridSlice_ = pESDFMap_->pESDFVDBGrid_->deepCopy();
        }
        std::size_t num_voxels = pESDFVDBGridSlice_->activeVoxelCount();
        std::atomic<std::size_t> cnt(0);
        std::size_t idx = 0;

        marker.points.clear();
        marker.points.reserve(pESDFMap_->range_num_voxles_);
        marker.colors.clear();
        marker.colors.reserve(pESDFMap_->range_num_voxles_);

        // iter the map
        std_msgs::ColorRGBA c;
        geometry_msgs::Point p;

        float offset_x = vis_cfg_.offset_x_;
        float offset_y = vis_cfg_.offset_y_;
        float offset_z = vis_cfg_.offset_z_;

        static float slice_level = 0.0f;
        static Eigen::Vector3f cur_pose = Eigen::Vector3f::Zero();
        if (vis_cfg_.use_fpv) {
            if (pTSDFMap_->pTransformer_->isPoseQeueEmpty()) {
                ROS_ERROR("[vdbblox_core::ESDFMap::updateFreeSpace] Pose Queue is empty!");
                return;
            }
            cur_pose = pTSDFMap_->pTransformer_->getBackPoseFromQueue();  // get the latest tf
            if (!cur_pose.allFinite()) {
                ROS_ERROR("[vdbblox_core::ESDFMap::updateFreeSpace] cur_pose is not finite!");
                return;
            }
            slice_level = cur_pose.z() / vis_cfg_.voxel_size_;

        } else slice_level = vis_cfg_.slice_vis_level_ / vis_cfg_.voxel_size_;

        static Eigen::Vector3f left_bottom = Eigen::Vector3f::Zero(), right_top = Eigen::Vector3f::Zero();
        if (vis_cfg_.use_fpv) {
            left_bottom = cur_pose - Eigen::Vector3f(vis_cfg_.update_range_, vis_cfg_.update_range_, vis_cfg_.update_range_z_);
            left_bottom.z() = vis_cfg_.offset_z_;
            left_bottom /= vis_cfg_.voxel_size_;
            left_bottom = (left_bottom.array() < 0).select(0, left_bottom);
            right_top = cur_pose + Eigen::Vector3f(vis_cfg_.update_range_, vis_cfg_.update_range_, vis_cfg_.update_range_z_);
            right_top /= vis_cfg_.voxel_size_;
            right_top = (right_top.array() < 0).select(0, right_top);
        }

        for (vdbblox_core::ESDFVDBGrid::ValueOnCIter iter = pESDFVDBGridSlice_->cbeginValueOn(); iter.test(); iter.next()) {
            if (!iter.isVoxelValue()) continue;

            const openvdb::Coord& coord = iter.getCoord();
            if (coord.z() != slice_level) continue;

            if (vis_cfg_.use_fpv) {
                if (coord.x() < left_bottom.x() || coord.x() > right_top.x() ||
                    coord.y() < left_bottom.y() || coord.y() > right_top.y()) continue;
            }

            const vdbblox_core::ESDFCell& esdf_cell = iter.getValue();
            double dist = esdf_cell.raw_distance;
            if (dist < 0) {
                continue;
            }
            // if (dist < 0 || dist > vis_cfg_.infinity_distance_m) continue;
            vis_cfg_.minx_dist_ = std::min(vis_cfg_.minx_dist_, dist);  vis_cfg_.max_dist_ = std::max(vis_cfg_.max_dist_, dist);

            const openvdb::Vec3d& pos = pESDFVDBGridSlice_->transform().indexToWorld(coord);
            p.x = pos.x() - offset_x; p.y = pos.y() - offset_y; p.z = pos.z() - offset_z;
            vis_cfg_.min_slice_z_ = std::min(vis_cfg_.min_slice_z_, p.z);  vis_cfg_.max_slice_z_ = std::max(vis_cfg_.max_slice_z_, p.z);

            c = RainbowColorMap(dist <= vis_cfg_.slice_vis_max_dist_ ? dist / vis_cfg_.slice_vis_max_dist_ : 1.0);

            idx = cnt.fetch_add(1);
            marker.points.emplace_back(p);
            marker.colors.emplace_back(c);
        }

        slice_pub_.publish(marker);

        // clear
        pESDFVDBGridSlice_->clear();
        publishESDFMapSlice.Stop();

        if (vis_cfg_.use_fpv){
            // publish visualization local map range
            cur_pose = cur_pose - Eigen::Vector3f(vis_cfg_.offset_x_, vis_cfg_.offset_y_, vis_cfg_.offset_z_);
            Eigen::Vector3f left_bottom = cur_pose - Eigen::Vector3f(vis_cfg_.update_range_, vis_cfg_.update_range_, vis_cfg_.update_range_z_);
            left_bottom.z() = 0;
            Eigen::Vector3f right_top = cur_pose + Eigen::Vector3f(vis_cfg_.update_range_, vis_cfg_.update_range_, vis_cfg_.update_range_z_);

            visualization_msgs::MarkerArray marker_array;
            marker_array.markers.clear();
            visualizeText(marker_array, "Local ESDF Map Text", "Local ESDF Map (Top)", right_top + Eigen::Vector3f(0.0, 0.0, 0.5), NormalizedRGB::White(), 0.6, 0);
            visualizeText(marker_array, "Local ESDF Map Text", "Local ESDF Map (Bottom)", left_bottom - Eigen::Vector3f(0.0, 0.0, 0.5), NormalizedRGB::White(), 0.6, 1);
            visualizeBoundingBox(marker_array, left_bottom, right_top, "Local Update Range", NormalizedRGB::Blue());
            marker_pub_.publish(marker_array);
        }

    }
}

std_msgs::ColorRGBA vdbblox_tools::Visualization::RainbowColorMap(float h) {
    std_msgs::ColorRGBA color;
    color.a = 1;
    // blend over HSV-values (more colors)

    float s = 1.0;
    float v = 1.0;

    h -= floor(h);
    h *= 6;
    int i;
    float m, n, f;

    i = floor(h);
    f = h - i;
    if (!(i & 1)) f = 1 - f;  // if i is even
    m = v * (1 - s);
    n = v * (1 - s * f);

    switch (i) {
        case 6:
        case 0:
            color.r = v;
            color.g = n;
            color.b = m;
            break;
        case 1:
            color.r = n;
            color.g = v;
            color.b = m;
            break;
        case 2:
            color.r = m;
            color.g = v;
            color.b = n;
            break;
        case 3:
            color.r = m;
            color.g = n;
            color.b = v;
            break;
        case 4:
            color.r = n;
            color.g = m;
            color.b = v;
            break;
        case 5:
            color.r = v;
            color.g = m;
            color.b = n;
            break;
        default:
            color.r = 1;
            color.g = 0.5;
            color.b = 0.5;
            break;
    }

    return color;
}