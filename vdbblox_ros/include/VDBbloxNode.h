#pragma once

#include <ros/publisher.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/ESDFMap.h"
#include "core/TSDFMap.h"
#include "core/Transform.h"
#include "ros/node_handle.h"
#include "ros/timer.h"
#include "tools/Visualization.h"
#include "vdbblox_ros/save_vdb_volume.h"

namespace vdbblox_node {

struct NodeConfig {
    std::string pcl_topic_;
    bool fill_holes_;
    float min_weight_;

    // UpdateESDF   
    bool updateESDF_ = false;
    bool shutdown_ = false;
};

class VDBbloxNode {
public:
    VDBbloxNode(ros::NodeHandle& nh);
    ~VDBbloxNode();

private:
    void Integrate(const sensor_msgs::PointCloud2& pcd);
    bool saveVDBVolume(vdbblox_ros::save_vdb_volume::Request& path,
                       vdbblox_ros::save_vdb_volume::Response& response);

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::ServiceServer srv_;
    ros::Duration timestamp_tolerance_;

    std::vector<Eigen::Vector3d> points_;

public:
    std::shared_ptr<vdbblox_core::Transform> pTf_;
    std::shared_ptr<vdbblox_core::TSDFMap> pTSDFMap_;

    std::shared_ptr<vdbblox_core::ESDFMap> pESDFMap_;

    NodeConfig node_cfg_;

    // Visualization
    std::shared_ptr<vdbblox_tools::Visualization> pVisualizer_;
    bool shutdown_ = false;
    void Visualization();
    std::mutex mutex_data_;

    // ESDF
    ros::Timer update_esdf_timer_;
    ros::Timer update_vis_timer_;
};
}  // namespace vdbblox_node
