#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <utility>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/parallel_for.h>
#include <openvdb/math/Transform.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <openvdb/openvdb.h>
#include "cell/TSDFCell.hpp"
#include "core/ESDFMap.h"
#include "igl/write_triangle_mesh.h"

#include "VDBbloxNode.h"
#include "core/TSDFMap.h"
#include "core/Transform.h"
#include "ros/console.h"
#include "ros/node_handle.h"
#include "tools/Visualization.h"
#include <tf2/convert.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_field_conversion.h>

vdbblox_node::VDBbloxNode::VDBbloxNode(ros::NodeHandle& nh) : nh_(nh) {
    // Initialize Transform
    pTf_ = std::make_shared<vdbblox_core::Transform>(nh_);

    // Initialize TSDF Map
    pTSDFMap_ = std::make_shared<vdbblox_core::TSDFMap>(nh_);
    const int queue_size = 10;
    nh_.getParam("/pcl_topic", node_cfg_.pcl_topic_);
    // sub_ = nh_.subscribe(node_cfg_.pcl_topic_, queue_size, &vdbblox_core::TSDFMap::Integrate, pTSDFMap_.get(), ros::TransportHints().tcpNoDelay());
    sub_ = nh_.subscribe(node_cfg_.pcl_topic_, queue_size, &vdbblox_core::TSDFMap::IntegrateROSInterval, pTSDFMap_.get());

    // Initialize ESDF Map
    pESDFMap_ = std::make_shared<vdbblox_core::ESDFMap>(nh_, pTSDFMap_);
    float update_esdf_every_n_sec;
    nh_.getParam("/update_esdf_every_n_sec", update_esdf_every_n_sec);
    nh_.getParam("/updateESDF", node_cfg_.updateESDF_);
    if (update_esdf_every_n_sec > 0 && node_cfg_.updateESDF_) {
        update_esdf_timer_ = nh_.createTimer(ros::Duration(update_esdf_every_n_sec), &vdbblox_core::ESDFMap::Integrate, pESDFMap_.get());
    }

    // visualization
    pVisualizer_ = std::make_shared<vdbblox_tools::Visualization>(nh_, pTSDFMap_, pESDFMap_);
    float update_vis_every_n_sec;
    nh_.getParam("/update_vis_every_n_sec", update_vis_every_n_sec);
    bool bUpdateROS = false;
    if (update_vis_every_n_sec > 0 && bUpdateROS) {
        update_vis_timer_ = nh_.createTimer(ros::Duration(update_vis_every_n_sec), &vdbblox_tools::Visualization::Visualize, pVisualizer_.get());
    }

    nh_.getParam("/fill_holes", node_cfg_.fill_holes_);
    nh_.getParam("/min_weight", node_cfg_.min_weight_);

    srv_ = nh_.advertiseService("/save_vdb_volume", &vdbblox_node::VDBbloxNode::saveVDBVolume, this);

    ROS_INFO("Use '/save_vdb_volume' service to save the integrated volume");
}

vdbblox_node::VDBbloxNode::~VDBbloxNode() {
    ROS_INFO("Shutting down VDBbloxNode");
}

bool vdbblox_node::VDBbloxNode::saveVDBVolume(vdbblox_ros::save_vdb_volume::Request& path,
                                             vdbblox_ros::save_vdb_volume::Response& response) {
    ROS_INFO("Saving the mesh and VDB grid files ...");
    std::string volume_name = path.path;

    openvdb::FloatGrid::Ptr tsdf = openvdb::FloatGrid::create(pTSDFMap_->tsdf_cfg_.sdf_trunc_);
    tsdf->setName("TSDF");
    tsdf->setTransform(openvdb::math::Transform::createLinearTransform(pTSDFMap_->tsdf_cfg_.voxel_size_));
    tsdf->setGridClass(openvdb::GRID_LEVEL_SET);
    auto tsdf_acc = tsdf->getAccessor();

    for (vdbblox_core::TSDFVDBGrid::ValueOnCIter iter = pTSDFMap_->pTSDFVDBGrid_->cbeginValueOn(); iter.test(); ++iter) {
        if (!iter.isVoxelValue()) {
            continue;
        }
        const openvdb::Coord& coord = iter.getCoord();
        const vdbblox_core::TSDFCell& val = iter.getValue();
        tsdf_acc.setValue(coord, val.tsdf);
    }

    openvdb::io::File(volume_name + "_tsdf.vdb").write({tsdf});

    // Run marching cubes and save a .ply file
    auto [vertices, triangles] =
        this->pTSDFMap_->ExtractTriangleMesh(node_cfg_.fill_holes_, node_cfg_.min_weight_);

    Eigen::MatrixXf V(vertices.size(), 3);
    for (size_t i = 0; i < vertices.size(); i++) {
        V.row(i) = Eigen::VectorXf::Map(&vertices[i][0], vertices[i].size());
    }

    Eigen::MatrixXi F(triangles.size(), 3);
    for (size_t i = 0; i < triangles.size(); i++) {
        F.row(i) = Eigen::VectorXi::Map(&triangles[i][0], triangles[i].size());
    }
    igl::write_triangle_mesh(volume_name + "_mesh.ply", V, F, igl::FileEncoding::Binary);
    ROS_INFO("Done saving the mesh and VDB grid files");
    return true;
}

