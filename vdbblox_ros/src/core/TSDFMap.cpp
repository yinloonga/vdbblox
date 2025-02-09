#include <geometry_msgs/TransformStamped.h>
#include <libmorton/morton.h>
#include <pcl/common/copy_point.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <ros/duration.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <tf/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <Eigen/Core>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <ostream>
#include <pcl/impl/point_types.hpp>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <future>

#include "cell/TSDFCell.hpp"
#include "ros/console.h"
#include "ros/node_handle.h"
#include "sensor_msgs/PointCloud2.h"
#include "tools/Timing.h"
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/partitioner.h>
#include <openvdb/Types.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/math/Transform.h>
#include <openvdb/openvdb.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tools/common.h>

#include "core/TSDFMap.h"

namespace {

inline float ComputeSDF(const Eigen::Vector3f& origin,
                 const Eigen::Vector3f& point,
                 const Eigen::Vector3f& voxel_center) {
    const Eigen::Vector3f v_voxel_origin = voxel_center - origin;
    const Eigen::Vector3f v_point_voxel = point - voxel_center;
    const double dist = v_point_voxel.norm();
    const double proj = v_voxel_origin.dot(v_point_voxel);

    if (proj == 0) return static_cast<float>(dist);

    // const double sign = proj / std::abs(proj);
    const double sign = proj > 0 ? 1 : -1;
    return static_cast<float>(sign * dist);
}

inline Eigen::Vector3f GetVoxelCenter(const openvdb::Coord& voxel, const openvdb::math::Transform& xform) {
    const float voxel_size = xform.voxelSize()[0];
    openvdb::math::Vec3d v_wf = xform.indexToWorld(voxel) + voxel_size / 2.0;
    return Eigen::Vector3f(v_wf.x(), v_wf.y(), v_wf.z());
}

}  // namespace

vdbblox_core::TSDFMap::TSDFMap(ros::NodeHandle& nh): nh_(nh) {
    pTf_ = std::make_shared<vdbblox_core::Transform>(nh_);
    pTransformer_ = std::make_shared<vdbblox_core::Transformer>(nh_);

    nh_.getParam("/use_multithread", tsdf_cfg_.use_multithread_);
    nh_.getParam("/space_carving", tsdf_cfg_.space_carving_);

    nh_.getParam("/voxel_size", tsdf_cfg_.voxel_size_);
    nh_.getParam("/sdf_trunc", tsdf_cfg_.sdf_trunc_);
    nh_.getParam("/downsample", tsdf_cfg_.downsample_);

    nh_.getParam("/tsdf_resize_factor", tsdf_cfg_.tsdf_resize_factor_);
    nh_.getParam("/esdf_resize_factor", tsdf_cfg_.esdf_resize_factor_);

    nh_.getParam("/preprocess", tsdf_cfg_.preprocess_);
    nh_.getParam("/apply_pose", tsdf_cfg_.apply_pose_);
    nh_.getParam("/min_range", tsdf_cfg_.min_range_);
    nh_.getParam("/max_range", tsdf_cfg_.max_range_);
    nh_.getParam("/offset_x", tsdf_cfg_.offset_x_);
    nh_.getParam("/offset_y", tsdf_cfg_.offset_y_);
    nh_.getParam("/offset_z", tsdf_cfg_.offset_z_);

    nh_.getParam("/enable_LRU", tsdf_cfg_.enable_LRU_);
    nh_.getParam("/update_local", tsdf_cfg_.update_local_);
    nh_.getParam("/update_range", tsdf_cfg_.update_range_);
    if (!nh_.getParam("/update_range_z", tsdf_cfg_.update_range_z_)) {
        ROS_WARN("update_range_z is deprecated, use update_range instead");
        tsdf_cfg_.update_range_z_ = tsdf_cfg_.update_range_;
    }
    nh_.getParam("/use_fpv", tsdf_cfg_.use_fpv);

    // TSFD Config
    tsdf_cfg_.bag_start_time_ = ros::Time(1473521338.62);
    tsdf_cfg_.bag_end_time_ = ros::Time(1473521481.40);
    double bag_start_time = 0, bag_end_time = 0;
    nh_.getParam("/bag_start_time", bag_start_time);
    nh_.getParam("/bag_end_time", bag_end_time);
    tsdf_cfg_.bag_start_time_ = ros::Time(bag_start_time);
    tsdf_cfg_.bag_end_time_ = ros::Time(bag_end_time);

    float kf_interval = 2.0f;
    nh_.getParam("/kf_interval", kf_interval);
    tsdf_cfg_.kf_interval_ = static_cast<std::size_t>(kf_interval);

    int32_t tol;
    nh_.getParam("/timestamp_tolerance_ns", tol);
    timestamp_tolerance_ = ros::Duration(0, tol);

    openvdb::initialize();
    pTSDFVDBGrid_ = TSDFVDBGrid::create(vdbblox_core::TSDFCell(0.0f));
    pTSDFVDBGrid_->setName("TSDFVDBGrid");
    pTSDFVDBGrid_->setTransform(openvdb::math::Transform::createLinearTransform(tsdf_cfg_.voxel_size_));
    pTSDFVDBGrid_->setGridClass(openvdb::GRID_LEVEL_SET);
    pTSDFVDBGridAcc_ = std::make_shared<TSDFVDBGrid::Accessor>(pTSDFVDBGrid_->getAccessor());

    // integrate_ = std::thread(&TSDFMap::Integrate, this);
    integrate_ = std::thread(&TSDFMap::processPointCloud, this);

    if (tsdf_cfg_.use_multithread_) {
        update_tsdf_ = std::thread(&TSDFMap::RunUpdateTSDF, this);
        update_tsdf_.detach();
        update_vdb_ = std::thread(&TSDFMap::RunUpdateVDB, this);
        update_vdb_.detach();
    }

    if (intensity_colormap_ == "rainbow") {
        pColorMap_ = std::make_shared<vdbblox_tools::RainbowColorMap>();
    } else if (intensity_colormap_ == "inverse_rainbow") {
        pColorMap_ = std::make_shared<vdbblox_tools::InverseRainbowColorMap>();
    } else if (intensity_colormap_ == "grayscale") {
        pColorMap_ = std::make_shared<vdbblox_tools::GrayscaleColorMap>();
    } else if (intensity_colormap_ == "inverse_grayscale") {
        pColorMap_ = std::make_shared<vdbblox_tools::InverseGrayscaleColorMap>();
    } else if (intensity_colormap_ == "ironbow") {
        pColorMap_ = std::make_shared<vdbblox_tools::IronbowColorMap>();
    } else {
        ROS_ERROR_STREAM("Invalid color map: " << intensity_colormap_);
    }
    pColorMap_->setMaxValue(intensity_max_value_);
    
    nh_.getParam("/invert_T_C_CH", tsdf_cfg_.invert_T_C_CH_);
    std::vector<double> matrix_data;
    nh_.getParam("/T_C_CH", matrix_data);
    if (matrix_data.size() == 16) {
        Eigen::Matrix4d T_C_CH;
        T_C_CH << matrix_data[0], matrix_data[1], matrix_data[2], matrix_data[3],
                  matrix_data[4], matrix_data[5], matrix_data[6], matrix_data[7],
                  matrix_data[8], matrix_data[9], matrix_data[10], matrix_data[11],
                  matrix_data[12], matrix_data[13], matrix_data[14], matrix_data[15];
        tsdf_cfg_.T_C_CH_ = T_C_CH;
    } else {
        ROS_ERROR("Invalid T_C_CH matrix size");
    }
    if (tsdf_cfg_.invert_T_C_CH_) {
        tsdf_cfg_.T_C_CH_ = tsdf_cfg_.T_C_CH_.inverse().eval();
    }

}

vdbblox_core::TSDFMap::~TSDFMap() { 

}

void vdbblox_core::TSDFMap::IntegrateROSInterval(const sensor_msgs::PointCloud2& pcd) {
    std::unique_lock<std::mutex> lock(mutex_integrate_ros_);

    std::size_t pc_size = pcd.width * pcd.height;
        
    // every 3 intervals, we update the TSDF
    if (cnt_queue_ % tsdf_cfg_.kf_interval_ == 0) {

        // voxel downsample
        if (tsdf_cfg_.downsample_) {
            vdbblox_tools::Timer PreProcessCloudTimer("PreProcessCloud");

            // convert ROS PointCloud2 to PCL PointCloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::fromROSMsg(pcd, *pcl_cloud);

            // voxel grid filter
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::VoxelGrid<pcl::PointXYZ> sor;
            sor.setInputCloud(pcl_cloud);
            sor.setLeafSize(tsdf_cfg_.voxel_size_, tsdf_cfg_.voxel_size_, tsdf_cfg_.voxel_size_);
            sor.filter(*filtered_cloud);

            // convert PCL PointCloud to ROS PointCloud2
            sensor_msgs::PointCloud2 pcd_downsampled;
            pcl::toROSMsg(*filtered_cloud, pcd_downsampled);
            pcd_downsampled.header = pcd.header;

            // Integrate the downsampled point cloud
            pushPCDIntoQueueBack(pcd_downsampled);
            PreProcessCloudTimer.Stop();
            std::size_t pc_downsampled_size = pcd_downsampled.width * pcd_downsampled.height;

        } else {
            pushPCDIntoQueueBack(pcd);
        }
    }
    cnt_queue_++;
    cv_integrate_.notify_one();

    ROS_WARN("sensor_msgs::PointCloud2 Qeueu: %d, Push: %d, Pop: %d, remain in queue: %d, pcd_queue size: %d", 
            static_cast<int>(cnt_queue_), static_cast<int>(cnt_push_), static_cast<int>(cnt_pop_), static_cast<int>(cnt_push_ - cnt_pop_), static_cast<int>(pcd_queue_.size()));
}

void vdbblox_core::TSDFMap::processPointCloud() {

    while (is_integrate_.load()) {
        std::unique_lock<std::mutex> lock(mutex_integrate_);
        cv_integrate_.wait(lock, [&] { return !isEmtpyPCDQueue() || !is_integrate_.load(); });
        if (!is_integrate_.load()) break;

        sensor_msgs::PointCloud2& pcd = getFrontPCDFromQueue();
        sensor_msgs::PointCloud2::Ptr pointcloud_msg = boost::make_shared<sensor_msgs::PointCloud2>(pcd);

        Eigen::Matrix4d tf_matrix = Eigen::Matrix4d::Identity();
        if (!pTransformer_->lookUpTransformfromSource(pointcloud_msg, tf_matrix)) {
            if (!init_tracking_) popPCDFromQueue();
            continue;
        }
        // check if the transformation matrix is valid
        if (!tf_matrix.allFinite()) ROS_ERROR("[vdbblox_core::TSDFMap::processPointCloud] Invalid transformation matrix");

        // convert ROS PointCloud2 to PCL PointCloud
        vdbblox_tools::PointCloud points_C; vdbblox_tools::Colors colors;
        vdbblox_tools::Timer processPointCloud("TSDF-processPointCloud");
        {
            bool color_pointcloud = false, has_intensity = false, has_label = false;
            for (size_t d = 0; d < pointcloud_msg->fields.size(); ++d) {
                if (pointcloud_msg->fields[d].name == std::string("rgb")) {
                    pointcloud_msg->fields[d].datatype = sensor_msgs::PointField::FLOAT32;
                    color_pointcloud = true;
                } else if (pointcloud_msg->fields[d].name == std::string("intensity")) {
                    has_intensity = true;
                } else if (pointcloud_msg->fields[d].name == std::string("label")) {
                    has_label = true;
                }
            }
            // pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            if (has_label) {
                pcl::PointCloud<pcl::PointXYZRGBL> pointcloud_pcl;
                pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
                convertPointcloud<pcl::PointXYZRGBL>(pointcloud_pcl, pColorMap_, points_C, colors);
                // pcl::copyPointCloud(pointcloud_pcl, *pcl_cloud);
            } else if (color_pointcloud) {
                pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
                pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
                convertPointcloud<pcl::PointXYZRGB>(pointcloud_pcl, pColorMap_, points_C, colors);
                // pcl::copyPointCloud(pointcloud_pcl, *pcl_cloud);
            } else if (has_intensity) {
                pcl::PointCloud<pcl::PointXYZI> pointcloud_pcl;
                pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
                if (tsdf_cfg_.use_fpv) {
                    Eigen::Matrix4d T_C_G = tf_matrix.inverse();
                    if (!T_C_G.allFinite()) ROS_ERROR("Invalid T_C_G matrix");
                    pcl::PointCloud<pcl::PointXYZI> pointcloud_pcl_transformed;
                    pcl::transformPointCloud(pointcloud_pcl, pointcloud_pcl_transformed, T_C_G.cast<float>());
                    convertPointcloud<pcl::PointXYZI>(pointcloud_pcl_transformed, pColorMap_, points_C, colors);
                } else {
                    convertPointcloud<pcl::PointXYZI>(pointcloud_pcl, pColorMap_, points_C, colors);
                }
                // pcl::copyPointCloud(pointcloud_pcl, *pcl_cloud);
            } else {
                pcl::PointCloud<pcl::PointXYZ> pointcloud_pcl;
                pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
                convertPointcloud<pcl::PointXYZ>(pointcloud_pcl, pColorMap_, points_C, colors);
                // pcl::copyPointCloud(pointcloud_pcl, *pcl_cloud);
            }
            // std::string clouds_size = "PointCloud: " + std::to_string(points_C.size()) + ", Color: " + std::to_string(colors.size());
            assert(points_C.size() == colors.size());
            if (points_C.size() == 0 || colors.size() == 0) ROS_ERROR("Empty point cloud or colors");
        }

        // combine the point cloud and colors
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_pcl = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        {
            pointcloud_pcl->resize(points_C.size());
            // Convert the PCL pointcloud into our awesome format.
            #pragma omp parallel for num_threads(std::max(1, vdbblox_tools::getMaxNumIdealThreads()))
            for (std::size_t i = 0; i < points_C.size(); ++i) {
                pcl::PointXYZRGB point;
                point.x = points_C[i].x();
                point.y = points_C[i].y();
                point.z = points_C[i].z();
                point.r = colors[i].r;
                point.g = colors[i].g;
                point.b = colors[i].b;
                pointcloud_pcl->points[i] = point;
            }
            assert(pointcloud_pcl->size() == points_C.size());
        }
        processPointCloud.Stop();

        // estimate normals
        pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        {
            vdbblox_tools::Timer ExtractNormalTimer("TSDF-ExtractNormal");
            ne.setNumberOfThreads(std::max(1, vdbblox_tools::getMaxNumIdealThreads()));
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
            Eigen::Vector3d origin = tf_matrix.block<3, 1>(0, 3).cast<double>();
            ne.setSearchMethod(tree);
            ne.setKSearch(10);
            ne.setViewPoint(origin.x(), origin.y(), origin.z());
            ne.setInputCloud(pointcloud_pcl);
            // ne.setRadiusSearch(0.05);
            ne.compute(*cloud_normals);
            ExtractNormalTimer.Stop();
        }

        // combine point cloud and normals
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        {
            vdbblox_tools::Timer combinePointCloudAndNormalTimer("TSDF-combinePointCloudAndNormal");
            cloud_with_normals->points.resize(pointcloud_pcl->points.size());
            std::size_t parallel_sum = 0;
            #pragma omp parallel for reduction(+:parallel_sum)        
            for (std::size_t i = 0; i < pointcloud_pcl->points.size(); ++i) {
                // check is point finite
                const pcl::PointXYZRGB& pt = pointcloud_pcl->points[i];
                if (!vdbblox_tools::isPointFinite(pt)) continue;
                const pcl::Normal& nm = cloud_normals->points[i];
                if (!std::isfinite(nm.normal_x) || !std::isfinite(nm.normal_y) || !std::isfinite(nm.normal_z)) continue;

                parallel_sum += 1;
                pcl::PointXYZRGBNormal& p = cloud_with_normals->points[i];
                p.x = pt.x; p.y = pt.y; p.z = pt.z;
                p.rgb = pt.rgb;
                p.normal_x = nm.normal_x; p.normal_y = nm.normal_y; p.normal_z = nm.normal_z;
            }
            cloud_with_normals->points.resize(parallel_sum);
            combinePointCloudAndNormalTimer.Stop();
        }

        // use voxel grid filter to downsample the point cloud
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals_filtered(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        {
            vdbblox_tools::Timer downsamplePointCloudTimer("TSDF-DownsamplePointCloud");
            pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor_;
            sor_.setLeafSize(tsdf_cfg_.voxel_size_, tsdf_cfg_.voxel_size_, tsdf_cfg_.voxel_size_);
            sor_.setDownsampleAllData(true);  // average of all points (x, y, z, rgb, normal_x, normal_y, normal_z) if false only x, y, z
            // sor_.setInputCloud(cloud);
            sor_.setInputCloud(cloud_with_normals);
            sor_.filter(*cloud_with_normals_filtered);
            downsamplePointCloudTimer.Stop();
        }

        // get the origin of the point cloud
        Eigen::Vector3f origin = tf_matrix.block<3, 1>(0, 3).cast<float>();
        vdbblox_tools::PointCloud normals_C;
        vdbblox_tools::PointCloud points_G = std::move(points_C);  points_C.clear();
        {
            vdbblox_tools::Timer ExtractPointCloudAndNormalTimer("TSDF-ExtractPointCloudAndNormal");
            std::size_t resize = cloud_with_normals_filtered->points.size();
            points_G.resize(resize);  // point is world coordinate
            normals_C.resize(resize);
            colors.resize(resize);
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals_filtered_transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::transformPointCloud(*cloud_with_normals_filtered, *cloud_with_normals_filtered_transformed, tf_matrix);
            std::size_t parallel_sum = 0;
            #pragma omp parallel for reduction(+:parallel_sum)
            for (std::size_t i = 0; i < resize; ++i) {
                parallel_sum += 1;
                const pcl::PointXYZRGBNormal& pt_nm = cloud_with_normals_filtered_transformed->points[i];
                points_G[i] = Eigen::Vector3f(pt_nm.x+tsdf_cfg_.offset_x_, pt_nm.y+tsdf_cfg_.offset_y_, pt_nm.z+tsdf_cfg_.offset_z_);
                if ((points_G[i].array() <= 0).any() ) ROS_ERROR("point is less than 0. Point: [%f, %f, %f]", points_G[i].x(), points_G[i].y(), points_G[i].z());

                normals_C[i] = Eigen::Vector3f(pt_nm.normal_x, pt_nm.normal_y, pt_nm.normal_z);
                colors[i] = vdbblox_tools::Color(pt_nm.r, pt_nm.g, pt_nm.b);
            }
            points_G.resize(parallel_sum);
            normals_C.resize(parallel_sum);
            colors.resize(parallel_sum);

            origin += Eigen::Vector3f(tsdf_cfg_.offset_x_, tsdf_cfg_.offset_y_, tsdf_cfg_.offset_z_);
            if ((origin.array() <= 0).any() ) ROS_ERROR("origin is less than 0. Origin: [%f, %f, %f]", origin.x(), origin.y(), origin.z());
            ExtractPointCloudAndNormalTimer.Stop();
        }

        {
            pTransformer_->pushPose(origin);
            // init the lru cache capacity
            if (!init_capacity_) {
                CAPACITY = points_G.size() * 2;
                init_capacity_ = true;
            }

            vdbblox_tools::Transformation T_G_C;
            vdbblox_tools::Eigen2KindrT(tf_matrix, T_G_C);
            UpdateTSDFCallback(T_G_C, origin, normals_C, points_G, colors, [](float /*unused*/) { return 1.0; });
            popPCDFromQueue();
            init_tracking_ = true;
        }

        {  // clear the point cloud
            points_C.clear();
            colors.clear();
            normals_C.clear();
        }

        std::string out_str = "";
        {
            ros::Time curr_time = pointcloud_msg->header.stamp;
            float bgTimePercent = static_cast<float>(curr_time.toSec() - tsdf_cfg_.bag_start_time_.toSec()) / (tsdf_cfg_.bag_end_time_.toSec() - tsdf_cfg_.bag_start_time_.toSec());
            out_str = std::to_string(curr_time.toSec());
            if (tsdf_cfg_.use_fpv) bgTimePercent = 0.0f;
            out_str += (std::stringstream() << " [" << std::fixed << std::setprecision(1) << bgTimePercent * 100 << "%]").str();
        }
        vdbblox_tools::Timing::Print(std::cout, out_str);

        ROS_WARN("sensor_msgs::PointCloud2 Qeueu: %d, Push: %d, Pop: %d, remain in queue: %d, pcd_queue size: %d", static_cast<int>(cnt_queue_), 
                    static_cast<int>(cnt_push_), static_cast<int>(cnt_pop_), static_cast<int>(cnt_push_ - cnt_pop_), static_cast<int>(pcd_queue_.size()));

    }

}

void vdbblox_core::TSDFMap::Integrate() {
    while (is_integrate_.load()) {

        std::unique_lock<std::mutex> lock(mutex_integrate_);
        cv_integrate_.wait(lock, [&] { return !isEmtpyPCDQueue() || !is_integrate_.load(); });
        if (!is_integrate_.load()) break;
        // sensor_msgs::PointCloud2&& pcd = getPCDFromQueue();
        sensor_msgs::PointCloud2& pcd = getFrontPCDFromQueue();
        geometry_msgs::TransformStamped transform;

        if (pTf_->lookUpTransform(pcd.header.stamp, timestamp_tolerance_, transform)) {
            ROS_INFO("Transform available");
            vdbblox_tools::Timer integrate("Integrate");

            std::vector<Eigen::Vector3f> scan;
            scan.clear();
            std::atomic<std::size_t> valid_cnt(0);

            if (tsdf_cfg_.apply_pose_) {
                Eigen::Transform<float, 3, Eigen::Isometry> tf =
                    Eigen::Translation3f(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z) *
                    Eigen::Quaternion<float>(transform.transform.rotation.w, transform.transform.rotation.x,
                                             transform.transform.rotation.y, transform.transform.rotation.z);
                // PointCloud2 convert to PointCloud
                int x_idx = sensor_msgs::getPointCloud2FieldIndex(pcd, "x");
                int y_idx = sensor_msgs::getPointCloud2FieldIndex(pcd, "y");
                int z_idx = sensor_msgs::getPointCloud2FieldIndex(pcd, "z");

                if (x_idx == -1 || y_idx == -1 || z_idx == -1) {
                    std::cerr << "x/y/z coordinates not found! Cannot convert to sensor_msgs::PointCloud!" << std::endl;
                    popPCDFromQueue();
                    return;
                }

                // points
                int x_offset = pcd.fields[x_idx].offset; int y_offset = pcd.fields[y_idx].offset; int z_offset = pcd.fields[z_idx].offset;
                uint8_t x_datatype = pcd.fields[x_idx].datatype; uint8_t y_datatype = pcd.fields[y_idx].datatype; uint8_t z_datatype = pcd.fields[z_idx].datatype;

                int pcd_size = pcd.width * pcd.height;
                scan.resize(pcd_size);

                tbb::parallel_for(
                    tbb::blocked_range<std::size_t>(0, pcd_size),
                    [&](const tbb::blocked_range<std::size_t>& r) {
                        for (std::size_t idx = r.begin(); idx != r.end(); ++idx) {
                            Eigen::Vector3f point_tmp = tf * 
                                        Eigen::Vector3f(sensor_msgs::readPointCloud2BufferValue<float>(&pcd.data[idx * pcd.point_step + x_offset], x_datatype),
                                                        sensor_msgs::readPointCloud2BufferValue<float>(&pcd.data[idx * pcd.point_step + y_offset], y_datatype),
                                                        sensor_msgs::readPointCloud2BufferValue<float>(&pcd.data[idx * pcd.point_step + z_offset], z_datatype));
                            if (std::isnan(point_tmp.x()) || std::isnan(point_tmp.y()) || std::isnan(point_tmp.z())) continue;
                            float point_norm = std::sqrt(point_tmp.x() * point_tmp.x() + point_tmp.y() * point_tmp.y() + point_tmp.z() * point_tmp.z());
                            if (point_norm > tsdf_cfg_.max_range_ || point_norm < tsdf_cfg_.min_range_) continue;

                            std::size_t pos = valid_cnt.fetch_add(1);
                            float point_x = point_tmp.x() + tsdf_cfg_.offset_x_;
                            float point_y = point_tmp.y() + tsdf_cfg_.offset_y_;
                            float point_z = point_tmp.z() + tsdf_cfg_.offset_z_;

                            if (point_x <= 0 || point_y <= 0 || point_z <= 0) {
                               ROS_ERROR("Point < 0: [%f, %f, %f]", point_x, point_y, point_z); 
                            }

                            scan[pos] = Eigen::Vector3f(point_x, point_y, point_z);
                        }
                    });
            }
            scan.resize(valid_cnt.load());

            const float& x = transform.transform.translation.x + tsdf_cfg_.offset_x_;
            const float& y = transform.transform.translation.y + tsdf_cfg_.offset_y_;
            const float& z = transform.transform.translation.z + tsdf_cfg_.offset_z_;

            if (x <= 0 || y <= 0 || z <= 0) {
                ROS_ERROR("Origin < 0: [%f, %f, %f]", x, y, z);
            }

            Eigen::Vector3f origin = Eigen::Vector3f(x, y, z);
            pTf_->pushPose(origin);

            if (!init_capacity_) {
                CAPACITY = scan.size() * 2;
                init_capacity_ = true;
            }

            if (tsdf_cfg_.use_multithread_) {
                this->pushPCLAndOrin(origin, scan);
            } else {
                // UpdateTSDFCallback(scan, origin, [](float /*unused*/) { return 1.0; });
            }

            popPCDFromQueue();
            init_tracking_ = true;

            integrate.Stop();
            vdbblox_tools::Timing::Print(std::cout);
        } else {
            if (!init_tracking_) popPCDFromQueue();
        }
    }
}

void vdbblox_core::TSDFMap::RunUpdateTSDF() {
    while (true) {
        if (isEmtpyPCLQueue()) continue;
        std::pair<Eigen::Vector3f, std::vector<Eigen::Vector3f>>&& orinAndPCL = this->getOrinAndPCL();
        std::function<float(float)> weighting_function = [](float /*unused*/) { return 1.0; };
        const Eigen::Vector3f& origin = orinAndPCL.first;
        const std::vector<Eigen::Vector3f>& points = orinAndPCL.second;

        if (points.empty()) {
            std::cerr << "PointCloud provided is empty\n";
            return;
        }
        vdbblox_tools::Timer run_update_tsdf("RunUpdateTSDF");
        // Get some variables that are common to all rays
        const openvdb::math::Transform& xform = pTSDFVDBGrid_->transform();
        const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());

        std::vector<std::tuple<openvdb::Coord, float, float>> updates;
        updates.resize(points.size());

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, points.size()),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t idx = r.begin(); idx != r.end(); ++idx) {
                    // Get the direction from the sensor origin to the point and normalize it
                    const Eigen::Vector3f direction = points[idx] - origin;
                    openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
                    dir.normalize();

                    const auto depth = static_cast<float>(direction.norm());
                    const float t0 = tsdf_cfg_.space_carving_ ? 0.0f : depth - tsdf_cfg_.sdf_trunc_;
                    const float t1 = depth + tsdf_cfg_.sdf_trunc_ + tsdf_cfg_.voxel_size_;

                    // Create one DDA per ray(per thread), the ray must operate on voxel grid coordinates.
                    const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*pTSDFVDBGrid_);
                    openvdb::math::DDA<decltype(ray)> dda(ray);
                    do {
                        const auto voxel = dda.voxel();
                        const auto voxel_center = GetVoxelCenter(voxel, xform);
                        const auto sdf = ComputeSDF(origin, points[idx], voxel_center);
                        if (sdf > -tsdf_cfg_.sdf_trunc_) {
                            const float tsdf = std::min(tsdf_cfg_.sdf_trunc_, sdf);
                            const float weight = weighting_function(sdf);
                            updates[idx] = std::make_tuple(voxel, tsdf, weight);
                        }
                    } while (dda.step());
                }
            });

        this->pushUpdates(updates);
        this->popOrinAndPCL();
        run_update_tsdf.Stop();
    }
}

void vdbblox_core::TSDFMap::RunUpdateVDB() {
    while (true) {
        if (isEmtpyUpdatesQueue()) continue;

        std::unique_lock<std::shared_mutex> lock(mutex_update_tsdf_);
        vdbblox_tools::Timer run_update_vdb("RunUpdateVDB");
        std::vector<std::tuple<openvdb::Coord, float, float>>&& updates = this->getUpdates();

        auto pTSDFVDBGridAcc = pTSDFVDBGrid_->getAccessor();
        for (const auto& [voxel, tsdf, weight] : updates) {
            const vdbblox_core::TSDFCell& last_value = pTSDFVDBGridAcc.getValue(voxel);
            const float new_weight = weight + last_value.weight;
            const float new_tsdf =
                (last_value.tsdf * last_value.weight + tsdf * weight) / (new_weight);
            vdbblox_core::TSDFCell new_tsdf_cell;
            new_tsdf_cell.tsdf = new_tsdf;
            new_tsdf_cell.weight = new_weight;
            pTSDFVDBGridAcc.setValue(voxel, new_tsdf_cell);
        }

        if (!tsdf_cfg_.init_mapping_) tsdf_cfg_.init_mapping_ = true;
        this->popUpdates();
        run_update_vdb.Stop();
    }
}

void vdbblox_core::TSDFMap::pushPCLAndOrin(const Eigen::Vector3f& origin,
                                           std::vector<Eigen::Vector3f>& pcl) {
    std::lock_guard<std::mutex> lock(mutex_pcl_);
    pcl_queue_.emplace(std::make_pair(origin, pcl));
    this->cnt_update_tsdf_++;
    this->plus_cnt_update_tsdf_++;
}

std::pair<Eigen::Vector3f, std::vector<Eigen::Vector3f>>&& vdbblox_core::TSDFMap::getOrinAndPCL() {
    std::lock_guard<std::mutex> lock(mutex_pcl_);
    return std::move(pcl_queue_.front());
}

void vdbblox_core::TSDFMap::popOrinAndPCL() {
    std::lock_guard<std::mutex> lock(mutex_pcl_);
    pcl_queue_.pop();
    this->cnt_update_tsdf_--;
    this->minus_cnt_update_tsdf_++;
}

bool vdbblox_core::TSDFMap::isEmtpyPCLQueue() {
    std::lock_guard<std::mutex> lock(mutex_pcl_);

    if (pcl_queue_.size() != 0)
        return false;
    else
        return true;
}

// UpdateVDB
void vdbblox_core::TSDFMap::pushUpdates(
    std::vector<std::tuple<openvdb::Coord, float, float>>& updates) {
    std::lock_guard<std::mutex> lock(mutex_updates_);
    updates_queue_.emplace(updates);
    this->cnt_update_vdb_++;
    this->plus_cnt_update_vdb_++;
}

std::vector<std::tuple<openvdb::Coord, float, float>>&& vdbblox_core::TSDFMap::getUpdates() {
    std::lock_guard<std::mutex> lock(mutex_updates_);
    return std::move(updates_queue_.front());
}

void vdbblox_core::TSDFMap::popUpdates() {
    std::lock_guard<std::mutex> lock(mutex_updates_);
    updates_queue_.pop();
    this->cnt_update_vdb_--;
    this->minus_cnt_update_vdb_++;
}

bool vdbblox_core::TSDFMap::isEmtpyUpdatesQueue() {
    std::lock_guard<std::mutex> lock(mutex_updates_);

    if (updates_queue_.size() != 0)
        return false;
    else
        return true;
}

void vdbblox_core::TSDFMap::pushPCDIntoQueueBack(const sensor_msgs::PointCloud2& pcd) {
    std::lock_guard<std::mutex> lock(mutex_pcd_queue_);
    pcd_queue_.emplace_back(pcd);
    cnt_push_++;
}

void vdbblox_core::TSDFMap::pushPCDIntoQueueFront(const sensor_msgs::PointCloud2& pcd) {
    std::lock_guard<std::mutex> lock(mutex_pcd_queue_);
    pcd_queue_.emplace_front(pcd);
    cnt_push_++;
}

sensor_msgs::PointCloud2&& vdbblox_core::TSDFMap::getPCDFromQueue() {
    std::lock_guard<std::mutex> lock(mutex_pcd_queue_);
    return std::move(pcd_queue_.front());
}

sensor_msgs::PointCloud2& vdbblox_core::TSDFMap::getFrontPCDFromQueue() {
    std::lock_guard<std::mutex> lock(mutex_pcd_queue_);
    return pcd_queue_.front();
}

void vdbblox_core::TSDFMap::popPCDFromQueue() {
    std::lock_guard<std::mutex> lock(mutex_pcd_queue_);
    pcd_queue_.pop_front();
    cnt_pop_++;
}

bool vdbblox_core::TSDFMap::isEmtpyPCDQueue() {
    std::lock_guard<std::mutex> lock(mutex_pcd_queue_);

    if (pcd_queue_.size() != 0)
        return false;
    else
        return true;
}


std::pair<float, float> vdbblox_core::TSDFMap::computeVoxelWeightRatio(const vdbblox_tools::Transformation& T_C_G, const vdbblox_tools::Point& normal_C, 
                                            const vdbblox_tools::Point& point_C, const openvdb::Coord& coord) {
    float lambda = 1.0f, normal_ratio = 1.0f;
    TSDFCell cell = pTSDFVDBGrid_->getAccessor().getValue(coord);

    if (cell.gradient.norm() > vdbblox_tools::kFloatEpsilon) {
        Eigen::Vector3f gradient_C = (T_C_G.getRotationMatrix() * cell.gradient.cast<double>()).cast<float>();
        if (tsdf_cfg_.curve_assumption && normal_C.norm() > vdbblox_tools::kFloatEpsilon) {
            // condition 1: curve assumption
            float cos_theta = std::abs(gradient_C.dot(point_C)/ point_C.norm());
            float cos_beta = std::abs(normal_C.dot(point_C) / (normal_C.norm() * point_C.norm()));
            float sin_theta = std::sqrt(1 - cos_theta * cos_theta);
            float sin_beta = std::sqrt(1 - cos_beta * cos_beta);
            lambda = std::abs(cos_theta * cos_beta + sin_theta * sin_beta);
            normal_ratio = cos_theta;
        } else {
            // condition 2: flat surface assumption
            float cos_theta = std::abs(gradient_C.dot(point_C) / point_C.norm());
            lambda = cos_theta;
            normal_ratio = cos_theta;
        }

    } else {
        // gradient not exist, only consider the normal
        if (normal_C.norm() > vdbblox_tools::kFloatEpsilon) {
            float cos_beta = std::abs(normal_C.dot(point_C) / (normal_C.norm() * point_C.norm()));
            lambda = cos_beta;
        }
    }
    return {lambda, normal_ratio};
}

float vdbblox_core::TSDFMap::updateVoxelWeight(const vdbblox_tools::Point& Point_C, const float sdf, const bool with_init_weight, const float init_weight) const {
    float weight = 1.0f;

    if (with_init_weight) weight = init_weight; 
    
    // Part 1. Weight reduction with distance
    // (according to sensor noise models).
    // Also Independent of sdf
    if (!tsdf_cfg_.use_const_weight) {
        weight /= std::pow(Point_C.norm(), tsdf_cfg_.weight_reduction_exp);
    }

    // Part 2. weight drop-off
    // Apply weight drop-off if appropriate.
    const float sdf_trunc = tsdf_cfg_.sdf_trunc_;
    if (tsdf_cfg_.use_weight_dropoff) {
        const float dropoff_epsilon = tsdf_cfg_.weight_dropoff_epsilon > 0.0f ? tsdf_cfg_.weight_dropoff_epsilon : tsdf_cfg_.weight_dropoff_epsilon * tsdf_cfg_.voxel_size_;
        // for example, weight_dropoff_epsilon = -1.0
        // --> dropoff_epsilon = voxel_size
        if (sdf < -dropoff_epsilon) {
            weight *= (sdf_trunc + sdf) /(sdf_trunc - dropoff_epsilon);
            weight = std::max(weight, 0.0f);
        }
    }

    // Part 3. deal with sparse point cloud
    // Compute the updated weight in case we compensate for sparsity. By
    // multiplicating the weight of occupied areas (|sdf| < truncation distance)
    // by a factor, we prevent to easily fade out these areas with the free
    // space parts of other rays which pass through the corresponding voxels.
    // This can be useful for creating a TSDF map from sparse sensor data (e.g.
    // visual features from a SLAM system). By default, this option is disabled.
    if (tsdf_cfg_.use_sparsity_compensation_factor) {
        if (std::abs(sdf) < sdf_trunc) weight *= tsdf_cfg_.sparsity_compensation_factor;
    }

    return weight;
}

Eigen::Vector3f vdbblox_core::TSDFMap::updateVoxelGradient(const openvdb::Coord& voxel , const Eigen::Vector3f& normal_G, float weight ) {
    vdbblox_core::TSDFCell cell;
    if (pTSDFVDBGridAcc_->isValueOn(voxel)) cell = pTSDFVDBGridAcc_->getValue(voxel);

    const float new_weight = cell.weight + weight;
    // it is possible to have weights very close to zero, due to the limited
    // precision of floating points dividing by this small value can cause nans
     if (new_weight < vdbblox_tools::kFloatEpsilon) {
        return Eigen::Vector3f::Zero();
     }

    if (cell.gradient.norm() > vdbblox_tools::kFloatEpsilon) {
        cell.gradient = ((cell.gradient * cell.weight + normal_G * weight) / new_weight).normalized();
    } else {
        // newly assigned, originally zero vector
        cell.gradient = normal_G.normalized();
    }
    return cell.gradient;
}

void vdbblox_core::TSDFMap::UpdateTSDFCallback( const vdbblox_tools::Transformation& T_G_C, const Eigen::Vector3f& origin,
                                                const vdbblox_tools::PointCloud& normals_C, const vdbblox_tools::PointCloud& points_G, const vdbblox_tools::Colors& colors,
                                                const std::function<float(float)>& weighting_function) {
    assert(normals_C.size() == points_G.size() && points_G.size() == colors.size());

    std::future<bool> future_prune;
    if (tsdf_cfg_.update_local_) {
        future_prune = std::async(std::launch::async, &vdbblox_core::TSDFMap::updateSlidingWindow, this, std::cref(points_G), std::cref(origin));
    }

    const vdbblox_tools::Transformation T_C_G = T_G_C.inverse();
    // check the transformation matrix is valid
    if (!T_C_G.getRotationMatrix().allFinite()) ROS_ERROR("[vdbblox_core::TSDFMap::UpdateTSDFCallback] Transformation matrix is invalid!");

    const openvdb::math::Transform& xform = pTSDFVDBGrid_->transform();
    const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());

    std::size_t ratio = static_cast<std::size_t>(std::round(tsdf_cfg_.sdf_trunc_ / tsdf_cfg_.voxel_size_));
    std::size_t ratio_size = points_G.size() * ratio;
    std::size_t band_resize = ratio_size * tsdf_cfg_.tsdf_resize_factor_;
    // check need to update local range
    if (tsdf_cfg_.update_local_) {
        // update the local range
        setUpdateRange(origin);
    }

    if (tsdf_cfg_.update_local_) future_prune.get();

    // Band TSDFMap voxel
    vdbblox_core::BandVoxelVector band_tsdf;
    band_tsdf.reserve(band_resize);

    // freespace TSDFMap voxel
    tbb::concurrent_vector<openvdb::Coord> freespace_tsdf;
    std::size_t freespace_resize = static_cast<std::size_t>(points_G.size() * (tsdf_cfg_.max_range_ / tsdf_cfg_.voxel_size_));
    if (tsdf_cfg_.space_carving_) freespace_tsdf.reserve(freespace_resize);

    std::atomic<std::size_t> cnt_unreliable_normal(0), cnt_unreliable_lambda(0), cnt_band_esdf(0);  // debug to watch the unreliable normal
    {
        vdbblox_tools::Timer update_timer("TSDF-UpdateTSDF");
        std::shared_lock<std::shared_mutex> lock(mutex_update_tsdf_);
        tbb::global_control control(tbb::global_control::max_allowed_parallelism, std::max(1, vdbblox_tools::getMaxNumIdealThreads()));
        tbb::parallel_for( tbb::blocked_range<std::size_t>(0, points_G.size()), [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t idx = r.begin(); idx != r.end(); ++idx) {

                // Transform the point to the camera frame
                const Eigen::Vector3f point_C = (T_C_G * points_G[idx].cast<double>()).cast<float>();
                if (!point_C.allFinite()) ROS_ERROR("Point_C = T_C_G * points_G[idx] is invalid!");
                // Transform the normal to the world frame
                const Eigen::Vector3f normal_G = (T_G_C.getRotationMatrix() * normals_C[idx].cast<double>()).cast<float>();
                if (!normal_G.allFinite()) ROS_ERROR("Normal_G = T_G_C.getRotationMatrix() * normals_C[idx] is invalid!");

                // Get the direction from the sensor origin to the point and normalize it
                const Eigen::Vector3f direction = points_G[idx] - origin;
                openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
                dir.normalize();

                // Truncate the Ray before and after the source unless space_carving_ is specified.
                const auto depth = static_cast<float>(direction.norm());
                const float t0 = tsdf_cfg_.space_carving_ ? 0.0f : depth - tsdf_cfg_.sdf_trunc_;
                const float t1 = depth + tsdf_cfg_.sdf_trunc_ + tsdf_cfg_.voxel_size_;

                openvdb::Vec3f trunc_voxel = (eye + dir * (depth - tsdf_cfg_.sdf_trunc_)) / tsdf_cfg_.voxel_size_;
                openvdb::Coord trunc_coord(trunc_voxel.x(), trunc_voxel.y(), trunc_voxel.z());
                bool is_trunc = false;

                // Create one DDA per ray(per thread), the ray must operate on voxel grid
                const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*pTSDFVDBGrid_);
                openvdb::math::DDA<decltype(ray)> dda(ray);
                do {
                    const auto voxel = dda.voxel();
                    if ((voxel.x() != trunc_coord.x()) && (!is_trunc)) {
                        if (tsdf_cfg_.space_carving_ && tsdf_cfg_.update_local_ && isInRange(voxel)) {
                            // update only the local range
                            freespace_tsdf.emplace_back(voxel);
                            continue;
                        } else if (tsdf_cfg_.space_carving_ && !tsdf_cfg_.update_local_) {
                            // update the whole range
                            freespace_tsdf.emplace_back(voxel);
                            continue;
                        } else continue;
                    } else is_trunc = true;

                    const auto voxel_center = GetVoxelCenter(voxel, xform);
                    float sdf = ComputeSDF(origin, points_G[idx], voxel_center);
                    if (sdf < tsdf_cfg_.sdf_trunc_ || is_trunc) {

                        float lambda = 1.0f, normal_ratio = 1.0f;
                        std::tie(lambda, normal_ratio) = computeVoxelWeightRatio(T_C_G, normals_C[idx], point_C, voxel);
                        
                        // check the lambda and normal_ratio is reliable
                        if (lambda < tsdf_cfg_.reliable_lambda_thre) {
                            cnt_unreliable_lambda.fetch_add(1);
                            continue;
                        }
                        if (normal_ratio <= tsdf_cfg_.reliable_normal_ratio_thre) {
                            cnt_unreliable_normal.fetch_add(1);
                            continue;   
                        }

                        float weight = updateVoxelWeight(point_C, sdf, true, lambda);
                        if (!std::isfinite(weight)) ROS_ERROR("Weight is invalid!");
                        sdf *= normal_ratio;

                        // update the voxel gradient
                        Eigen::Vector3f gradient = Eigen::Vector3f::Zero();
                        if (tsdf_cfg_.normal_available) {
                            gradient = updateVoxelGradient(voxel, normal_G, weight);
                            if (!gradient.allFinite()) ROS_ERROR("Gradient is invalid!");
                        }

                        vdbblox_core::Voxel update_voxel;
                        update_voxel.coord = voxel;
                        update_voxel.tsdf = sdf;
                        update_voxel.weight = weight /* weight */;
                        update_voxel.gradient = gradient;
                        band_tsdf.emplace_back(update_voxel);

                    } 
                } while (dda.step());
            }
        }, tbb::simple_partitioner());

        update_timer.Stop();

    }
    std::string msg = "TSDFMap ==> [Band TSDF] size: " + std::to_string(band_tsdf.size()) + ", unreliable lambda: " + std::to_string(cnt_unreliable_lambda.load()) 
                        + ", unreliable normal: " + std::to_string(cnt_unreliable_normal.load()) + "\n";
    write(STDOUT_FILENO, msg.c_str(), msg.size());

    // Freespace TSDFMap update
    std::size_t freespace_size = freespace_tsdf.size();
    if (tsdf_cfg_.space_carving_) {
        {
            std::lock_guard<std::mutex> lock(mtx_freespace_tsdf_);
            freespace_tsdf_= std::move(freespace_tsdf);
            sync_freespace_tsdf_ = false;
            cv_freespace_tsdf_.notify_one();
        }
    }

    ROS_INFO("TSDF ==> [Band TSDF] resize: %d, actual size: %d, [Freespace TSDF] resize: %d, actual size: %d", static_cast<int>(band_resize), static_cast<int>(band_tsdf.size()), static_cast<int>(freespace_resize), static_cast<int>(freespace_size));

    {
        vdbblox_tools::Timer update_tsdf("TSDF-UpdateVDB");
        std::unique_lock<std::shared_mutex> lock(mutex_update_tsdf_);
        TSDFVDBGrid::Accessor pTSDFVDBGridAcc = pTSDFVDBGrid_->getAccessor();
        // ESDFMap updat, count in (-voxel_size_, voxel_size_) range for each voxel
        tbb::concurrent_vector<std::pair<openvdb::Coord, float/* distance */>> updates_esdf;
        band_resize = band_tsdf.size();  updates_esdf.resize(band_resize);
        std::atomic<std::size_t> cnt_esdf_in_band(0);

        // const float weight = 1.0f;
        for (const auto& update_voxel : band_tsdf) {
            vdbblox_core::TSDFCell cell;
            if (pTSDFVDBGridAcc.isValueOn(update_voxel.coord)) cell = pTSDFVDBGridAcc.getValue(update_voxel.coord);
            // update the voxel tsdf 
            const float new_weight = update_voxel.weight + cell.weight;
            float new_tsdf = (cell.tsdf * cell.weight + update_voxel.tsdf * update_voxel.weight) / (new_weight);
            cell.tsdf = new_tsdf;
            cell.tsdf = std::min(cell.tsdf, tsdf_cfg_.sdf_trunc_);
            // udpate the voxel weight
            cell.weight = new_weight;
            cell.weight = std::min(cell.weight, tsdf_cfg_.max_weight);
            // update the voxel gradient
            cell.gradient = update_voxel.gradient;
            // update the voxel
            pTSDFVDBGridAcc.setValue(update_voxel.coord, cell);
            // Update LRU Cache
            if (tsdf_cfg_.voxel_size_ > new_tsdf && new_tsdf >= 0) {
                const std::size_t curr_esdf_idx = cnt_esdf_in_band.fetch_add(1);
                if (curr_esdf_idx >= band_resize) {
                    ROS_ERROR("ESDFMap update index out of range: %d, resize: %d", static_cast<int>(curr_esdf_idx), static_cast<int>(band_resize));
                }
                updates_esdf[curr_esdf_idx] = std::make_pair(update_voxel.coord, new_tsdf);
            } 
        }
        band_tsdf.clear();
        updates_esdf.resize(cnt_esdf_in_band.load());
        if (tsdf_cfg_.enable_LRU_) setLRUCache(updates_esdf);
        // ROS_WARN("TSDF ==> [LRU ESDF] size: %d, actual LRU cache: %d", static_cast<int>(updates_esdf.size()), static_cast<int>(lru_cache_.size()));
        updates_esdf.clear();
        update_tsdf.Stop();
    }
    if (!tsdf_cfg_.init_mapping_) tsdf_cfg_.init_mapping_ = true;
    this->cv_update_done_.notify_one();
    this->update_done_ = true;
    if (!use_sync_) return;
    {
        // wait for the freespace TSDFMap update
        while (true) {
            std::lock_guard<std::mutex> lock(mtx_freespace_tsdf_);
            if (sync_freespace_tsdf_) break;
        }
    }
}

bool vdbblox_core::TSDFMap::setLRUCache(const tbb::concurrent_vector<std::pair<openvdb::Coord, float/* distance */>>& updates_esdf) {
    std::lock_guard<std::mutex> lock(mutex_lru_);
    vdbblox_tools::Timer setLRUCacheTimer("TSDF-setLRUCache");
    for (const auto& [coord, tsdf] : updates_esdf) {
        addPoint(coord, tsdf);
        min_coord_x_ = std::min(min_coord_x_, coord.x());  max_coord_x_ = std::max(max_coord_x_, coord.x());
        min_coord_y_ = std::min(min_coord_y_, coord.y());  max_coord_y_ = std::max(max_coord_y_, coord.y());
        min_coord_z_ = std::min(min_coord_z_, coord.z());  max_coord_z_ = std::max(max_coord_z_, coord.z());
    }
    setLRUCacheTimer.Stop();
    return true;
}


void vdbblox_core::TSDFMap::addPoint(const openvdb::Coord& coord, const float& tsdf) {
    std::lock_guard<std::mutex> lock(mutex_lru_cache_);
    // check if the cache is need re-initialized (in ESDFMap)
    if (init_cache_) {
        lru_cache_.clear();
        hash_map_.clear();
        init_cache_ = false;
    }
    auto iter = hash_map_.find(coord);
    if (iter == hash_map_.end()) {
        // key does no exist, add to cache
        lru_cache_.emplace_front(coord, tsdf);
        hash_map_.emplace(coord, lru_cache_.begin());

        // check if the cache is full
        if (hash_map_.size() > CAPACITY) {
            hash_map_.erase(lru_cache_.back().first);
            lru_cache_.pop_back();
        }
    } else {
        // key exists, update the value
        lru_cache_.splice(lru_cache_.begin(), lru_cache_, iter->second);
        lru_cache_.begin()->second = tsdf;
        hash_map_[coord] = lru_cache_.begin();
    }
    
}

bool vdbblox_core::TSDFMap::setUpdateRange(const Eigen::Vector3f& origin) {
    // udpate the left-bottom corner
    tsdf_cfg_.left_bottom_ = origin - Eigen::Vector3f(tsdf_cfg_.update_range_, tsdf_cfg_.update_range_, tsdf_cfg_.update_range_z_);
    tsdf_cfg_.left_bottom_ /= tsdf_cfg_.voxel_size_;
    tsdf_cfg_.left_bottom_ = (tsdf_cfg_.left_bottom_.array() < 0).select(0, tsdf_cfg_.left_bottom_);
    // update the right-top corner
    tsdf_cfg_.right_top_ = origin + Eigen::Vector3f(tsdf_cfg_.update_range_, tsdf_cfg_.update_range_, tsdf_cfg_.update_range_z_);
    tsdf_cfg_.right_top_ /= tsdf_cfg_.voxel_size_;
    tsdf_cfg_.right_top_ = (tsdf_cfg_.right_top_.array() < 0).select(0, tsdf_cfg_.right_top_);

    return true;
}

bool vdbblox_core::TSDFMap::isInRange(const openvdb::Coord& coord) {
    return (coord.x() >= tsdf_cfg_.left_bottom_.x() && coord.x() <= tsdf_cfg_.right_top_.x() &&
            coord.y() >= tsdf_cfg_.left_bottom_.y() && coord.y() <= tsdf_cfg_.right_top_.y() &&
            coord.z() >= tsdf_cfg_.left_bottom_.z() && coord.z() <= tsdf_cfg_.right_top_.z());
}

template <typename PCLPoint>
void vdbblox_core::TSDFMap::convertPointcloud(const typename pcl::PointCloud<PCLPoint>& pointcloud_pcl, const std::shared_ptr<vdbblox_tools::ColorMap>& color_map, 
                            vdbblox_tools::PointCloud& points_C, vdbblox_tools::Colors& colors) {
    
    points_C.reserve(pointcloud_pcl.size());
    colors.reserve(pointcloud_pcl.size());

    // use tbb
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, std::max(1, vdbblox_tools::getMaxNumIdealThreads()));
    tbb::parallel_for( tbb::blocked_range<std::size_t>(0, pointcloud_pcl.size()), [&](const tbb::blocked_range<std::size_t>& r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            PCLPoint point = pointcloud_pcl.points[i];

            // check isPointFinite
            if (!vdbblox_tools::isPointFinite(point)) continue;
            // check isPointInRange
            float point_norm = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (point_norm > tsdf_cfg_.max_range_ || point_norm < tsdf_cfg_.min_range_) continue;

            points_C.emplace_back(vdbblox_tools::Point(point.x, point.y, point.z));
            colors.emplace_back(vdbblox_tools::convertColor<PCLPoint>(point, color_map));
        }
    }, tbb::simple_partitioner());

}

bool vdbblox_core::TSDFMap::updateSlidingWindow(const vdbblox_tools::PointCloud& points_G, const Eigen::Vector3f& pose) {

    vdbblox_tools::Timer run_update("TSDF-UpdateSlidingWindow");
    if (points_queue_.size() >= 5) {
        std::unique_lock<std::shared_mutex> lock(mutex_update_tsdf_);
        // remove the front element
        Eigen::Vector3f origin;  vdbblox_tools::PointCloud points_front;
        std::tie(origin, points_front) = std::move(points_queue_.front());
        std::size_t raw_size = points_front.size();

        std::size_t before_active_voxels = pTSDFVDBGrid_->activeVoxelCount();
        std::size_t before_active_mem = pTSDFVDBGrid_->memUsage();

        const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());
        const openvdb::math::Transform& xform = pTSDFVDBGrid_->transform();
        std::atomic<std::size_t> cnt_invalid(0);
        tbb::global_control control(tbb::global_control::max_allowed_parallelism, std::max(1, vdbblox_tools::getMaxNumIdealThreads()));
        tbb::parallel_for( tbb::blocked_range<std::size_t>(0, points_front.size()), [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t idx = r.begin(); idx != r.end(); ++idx) {
                const Eigen::Vector3f direction = points_front[idx] - origin;
                openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
                dir.normalize();

                const auto depth = static_cast<float>(direction.norm());
                const float t0 = depth - tsdf_cfg_.sdf_trunc_;
                const float t1 = depth + tsdf_cfg_.sdf_trunc_ + tsdf_cfg_.voxel_size_;

                openvdb::Vec3f trunc_voxel = (eye + dir * (depth - tsdf_cfg_.sdf_trunc_)) / tsdf_cfg_.voxel_size_;
                openvdb::Coord trunc_coord(trunc_voxel.x(), trunc_voxel.y(), trunc_voxel.z());
                bool is_trunc = false;

                const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*pTSDFVDBGrid_);
                openvdb::math::DDA<decltype(ray)> dda(ray);
                do {
                    const auto voxel = dda.voxel(); 
                    if ((voxel.x() != trunc_coord.x()) && (!is_trunc)) {
                        continue;
                    } else is_trunc = true;

                    // deactive the voxel
                    if (pTSDFVDBGridAcc_->isValueOn(voxel)) {
                        pTSDFVDBGridAcc_->setValueOff(voxel);
                        cnt_invalid.fetch_add(1);
                    }

                } while (dda.step());
            
            }
        }, tbb::simple_partitioner());

        // prune the grid
        pTSDFVDBGrid_->pruneGrid();

        std::size_t after_active_voxels = pTSDFVDBGrid_->activeVoxelCount();
        std::size_t after_active_mem = pTSDFVDBGrid_->memUsage();
        ROS_INFO("TSDF ==> [Sliding Window] Points: %ld, Invalid: %ld, before active voxels: %d, after active voxels: %d, before active memory: %d, after active memory: %d", raw_size, static_cast<std::size_t>(cnt_invalid.load()),
                    static_cast<int>(before_active_voxels), static_cast<int>(after_active_voxels), static_cast<int>(before_active_mem), static_cast<int>(after_active_mem));

        points_queue_.pop();
        points_queue_.emplace(pose, points_G);
    } else {
        // add the points into the queue
        points_queue_.emplace(pose, points_G);
    }

    run_update.Stop();
    return true;
}