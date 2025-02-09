#pragma once

#include <oneapi/tbb/concurrent_vector.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/math/Vec4.h>
#include <openvdb/openvdb.h>
#include <ros/publisher.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Core>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "cell/TSDFCell.hpp"
#include "core/Transform.h"
#include "ros/node_handle.h"
#include "ros/subscriber.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h> // 计算法线需要包含的头文件
#include <pcl/features/normal_3d_omp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include "tools/common.h"
#include "core/transform.h"

namespace vdbblox_core {

struct TSDFConfig {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool use_multithread_ = false;
    bool space_carving_ = false;
    bool init_mapping_ = false;

    bool downsample_ = false;

    bool preprocess_, apply_pose_;
    float voxel_size_, sdf_trunc_;
    float min_range_, max_range_;
    float offset_x_, offset_y_, offset_z_;

    bool enable_LRU_ = false;
    bool update_local_ = false;
    float update_range_ = 1.0f, update_range_z_ = 0.5f;
    Eigen::Vector3f left_bottom_, right_top_;

    std::size_t kf_interval_ = 2;

    float tsdf_resize_factor_ = 5.0f;
    float esdf_resize_factor_ = 1.0f;

    // ROS::Bag
    ros::Time bag_start_time_;
    ros::Time bag_end_time_;

    // Model parameters
    bool invert_T_C_CH_ = false;
    Eigen::Matrix4d T_C_CH_;

    // Weighting function
    // if true, then we apply weight reduction
    bool use_const_weight = false;
    float weight_reduction_exp = 2.0f;
    bool allow_clear = true;
    bool use_weight_dropoff = true;
    // if negative, then the value would be the ratio of voxel size
    float weight_dropoff_epsilon = -1.0f;
    bool use_sparsity_compensation_factor = false;
    float sparsity_compensation_factor = 1.0f;
    float max_weight = 10.0;

    /// non-projective correction specific
    bool normal_available = true;
    float reliable_band_ratio = 2.0f;
    bool curve_assumption = true;
    float reliable_normal_ratio_thre = 0.3f;
    float reliable_lambda_thre = 0.1f;

    // MARSIM specific
    bool use_fpv = false;

};

using TSDFVDBTree = openvdb::tree::Tree4<vdbblox_core::TSDFCell, 5, 4, 3>::Type;
using TSDFVDBGrid = openvdb::Grid<TSDFVDBTree>;

class TSDFMap {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TSDFMap(ros::NodeHandle& nh);
    TSDFMap(const TSDFMap& tsdf_map, const std::lock_guard<std::mutex>&)
        : pTSDFVDBGrid_(tsdf_map.pTSDFVDBGrid_), tsdf_cfg_(tsdf_map.tsdf_cfg_) {}
    TSDFMap(const TSDFMap& tsdf_map)
        : TSDFMap(tsdf_map, std::lock_guard<std::mutex>(tsdf_map.mutex_integrate_)) {}
    ~TSDFMap();

public:
    /// @brief Integrates a new (globally aligned) PointCloud into the current
    void UpdateTSDFCallback( const vdbblox_tools::Transformation& T_G_C, const Eigen::Vector3f& origin,
                             const vdbblox_tools::PointCloud& normals_C, const vdbblox_tools::PointCloud& points_G, const vdbblox_tools::Colors& colors,
                             const std::function<float(float)>& weighting_function);
    std::pair<float, float> computeVoxelWeightRatio(const vdbblox_tools::Transformation& T_C_G, const vdbblox_tools::Point& normal_C, 
                            const vdbblox_tools::Point& point_C, const openvdb::Coord& coord);
    float updateVoxelWeight(const vdbblox_tools::Point& Point_C, const float sdf, const bool with_init_weight, const float init_weight) const;
    Eigen::Vector3f updateVoxelGradient(const openvdb::Coord& voxel , const Eigen::Vector3f& normal_G, float weight );
    void RunUpdateTSDF();

    void RunUpdateVDB();

    /// @brief Extracts a TriangleMesh as the iso-surface in the actual volume
    [[nodiscard]] std::tuple<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3i>>
    ExtractTriangleMesh(bool fill_holes = true, float min_weight = 0.5) const;

    template <typename PCLPoint>
    void convertPointcloud(const typename pcl::PointCloud<PCLPoint>& pointcloud_pcl, const std::shared_ptr<vdbblox_tools::ColorMap>& color_map, 
                                vdbblox_tools::PointCloud& points_C, vdbblox_tools::Colors& colors) ;

public:
    // Intergrate
    std::size_t cnt_queue_ = 0, cnt_pop_ = 0, cnt_push_ = 0;
    std::mutex mutex_pcd_queue_;
    std::deque<sensor_msgs::PointCloud2> pcd_queue_;
    std::thread integrate_;
    std::atomic<bool> is_integrate_ = true;
    void pushPCDIntoQueueBack(const sensor_msgs::PointCloud2& pcd);
    void pushPCDIntoQueueFront(const sensor_msgs::PointCloud2& pcd);
    sensor_msgs::PointCloud2&& getPCDFromQueue();
    sensor_msgs::PointCloud2& getFrontPCDFromQueue();
    void popPCDFromQueue();
    bool isEmtpyPCDQueue();

    mutable std::mutex mutex_integrate_;
    std::mutex mutex_integrate_ros_;
    std::condition_variable cv_integrate_;
    void IntegrateROSInterval(const sensor_msgs::PointCloud2& pcd);
    void processPointCloud();
    void Integrate();
    ros::NodeHandle nh_;
    std::shared_ptr<vdbblox_core::Transform> pTf_;
    std::shared_ptr<vdbblox_core::Transformer> pTransformer_;
    std::shared_ptr<vdbblox_tools::ColorMap> pColorMap_;
    std::string intensity_colormap_ = "rainbow";
    float intensity_max_value_ = 100.0f;

    ros::Duration timestamp_tolerance_;
    ros::Subscriber sub_;
    bool init_tracking_ = false;

    TSDFConfig tsdf_cfg_;

    // process pointclouds (UpdateTSDF)
    mutable std::shared_mutex mutex_update_tsdf_;
    std::mutex mutex_pcl_;
    std::queue<std::pair<Eigen::Vector3f, std::vector<Eigen::Vector3f>>> pcl_queue_;
    void pushPCLAndOrin(const Eigen::Vector3f& origin, std::vector<Eigen::Vector3f>& pcl);
    std::pair<Eigen::Vector3f, std::vector<Eigen::Vector3f>>&& getOrinAndPCL();
    void popOrinAndPCL();
    bool isEmtpyPCLQueue();
    std::size_t cnt_update_tsdf_ = 0;
    std::size_t plus_cnt_update_tsdf_ = 0;
    std::size_t minus_cnt_update_tsdf_ = 0;

    // UpdateVDB
    std::mutex mutex_update_vdb_;
    std::mutex mutex_updates_;
    std::queue<std::vector<std::tuple<openvdb::Coord, float, float>>> updates_queue_;
    void pushUpdates(std::vector<std::tuple<openvdb::Coord, float, float>>& updates);
    std::vector<std::tuple<openvdb::Coord, float, float>>&& getUpdates();
    void popUpdates();
    bool isEmtpyUpdatesQueue();
    std::size_t cnt_update_vdb_ = 0;
    std::size_t plus_cnt_update_vdb_ = 0;
    std::size_t minus_cnt_update_vdb_ = 0;

    // use multi thread
    std::thread update_tsdf_, update_vdb_;

    // LRU Cache for ESDF integrate
    bool init_capacity_ = false, init_cache_ = false;
    std::size_t CAPACITY = 10;
    std::mutex mutex_lru_cache_, mutex_lru_;;
    std::unordered_map<openvdb::Coord, typename std::list<std::pair<openvdb::Coord, float>>::iterator> hash_map_;
    std::list<std::pair<openvdb::Coord, float/* distance */>> lru_cache_;
    void addPoint(const openvdb::Coord& coord, const float& tsdf);
    bool setLRUCache(const tbb::concurrent_vector<std::pair<openvdb::Coord, float/* distance */>>& updates_esdf);
    bool setUpdateRange(const Eigen::Vector3f& origin);
    bool isInRange(const openvdb::Coord& coord);
    std::queue<std::pair<Eigen::Vector3f/* orign */, vdbblox_tools::PointCloud> /* points */> points_queue_;
    bool updateSlidingWindow(const vdbblox_tools::PointCloud& points_G, const Eigen::Vector3f& origin);

    // Free space TSDFMap voxel
    tbb::concurrent_vector<openvdb::Coord> freespace_tsdf_;
    std::mutex mtx_freespace_tsdf_;
    std::condition_variable cv_freespace_tsdf_;
    bool use_sync_ = false, sync_freespace_tsdf_ = false;

    // Visualization
    std::condition_variable cv_update_done_;
    bool update_done_ = false;

public:
    /// OpenVDB Grids modeling the signed distance field and the weight grid
    TSDFVDBGrid::Ptr pTSDFVDBGrid_;
    std::shared_ptr<TSDFVDBGrid::Accessor> pTSDFVDBGridAcc_;

    float min_x_ = 0, min_y_ = 0, min_z_ = 0;
    float max_x_ = 0, max_y_ = 0, max_z_ = 0;

    float min_orin_x_ = 0, min_orin_y_ = 0, min_orin_z_ = 0;
    float max_orin_x_ = 0, max_orin_y_ = 0, max_orin_z_ = 0;

    std::int32_t min_coord_x_ = 0, min_coord_y_ = 0, min_coord_z_ = 0;
    std::int32_t max_coord_x_ = 0, max_coord_y_ = 0, max_coord_z_ = 0;
};

}  // namespace vdbblox_core
