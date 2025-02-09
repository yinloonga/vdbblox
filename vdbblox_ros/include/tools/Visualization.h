#pragma once

#include <openvdb/Types.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include "core/ESDFMap.h"
#include "core/TSDFMap.h"
#include "ros/forwards.h"
#include "ros/node_handle.h"
#include "ros/publisher.h"
#include "sensor_msgs/PointCloud.h"
#include "std_msgs/ColorRGBA.h"
#include "visualization_msgs/MarkerArray.h"

namespace vdbblox_tools {

class NormalizedRGB  : public std_msgs::ColorRGBA {
public:
    NormalizedRGB () : std_msgs::ColorRGBA() {}

    NormalizedRGB (int hex_color) {
        int _r = (hex_color >> 16) & 0xFF;
        int _g = (hex_color >> 8) & 0xFF;
        int _b = hex_color & 0xFF;
        r = static_cast<double>(_r) / 255.0;
        g = static_cast<double>(_g) / 255.0;
        b = static_cast<double>(_b) / 255.0;
    }

    NormalizedRGB (NormalizedRGB  c, double alpha) {
        r = c.r;
        g = c.g;
        b = c.b;
        a = alpha;
    }

    NormalizedRGB (double red, double green, double blue) : NormalizedRGB (red, green, blue, 1.0) {
        r = red > 1.0 ? red / 255.0 : red;
        g = green > 1.0 ? green / 255.0 : green;
        b = blue > 1.0 ? blue / 255.0 : blue;
    }

    NormalizedRGB (double red, double green, double blue, double alpha) : NormalizedRGB () {
        r = red > 1.0 ? red / 255.0 : red;
        g = green > 1.0 ? green / 255.0 : green;
        b = blue > 1.0 ? blue / 255.0 : blue;
        a = alpha;
    }

    static const NormalizedRGB  White() { return NormalizedRGB (1.0, 1.0, 1.0); }

    static const NormalizedRGB  Black() { return NormalizedRGB (0.0, 0.0, 0.0); }

    static const NormalizedRGB  Gray() { return NormalizedRGB (0.5, 0.5, 0.5); }

    static const NormalizedRGB  Red() { return NormalizedRGB (1.0, 0.0, 0.0); }

    static const NormalizedRGB  Green() { return NormalizedRGB (0.0, 0.96, 0.0); }

    static const NormalizedRGB  Blue() { return NormalizedRGB (0.0, 0.0, 1.0); }

    static const NormalizedRGB  SteelBlue() { return NormalizedRGB (0.4, 0.7, 1.0); }

    static const NormalizedRGB  Yellow() { return NormalizedRGB (1.0, 1.0, 0.0); }

    static NormalizedRGB  Orange() { return NormalizedRGB (1.0, 0.5, 0.0); }

    static const NormalizedRGB  Purple() { return NormalizedRGB (0.5, 0.0, 1.0); }

    static const NormalizedRGB  Chartreuse() { return NormalizedRGB (0.5, 1.0, 0.0); }

    static const NormalizedRGB  Teal() { return NormalizedRGB (0.0, 1.0, 1.0); }

    static const NormalizedRGB  Pink() { return NormalizedRGB (1.0, 0.0, 0.5); }
};

struct VisConfig {

    // visualization range
    float max_range_ = 5.0f;
    float min_range_ = 0.5f;
    float update_range_ = 0.5f, update_range_z_ = 0.5f;
    float sdf_trunc_ = 0.6f;

    double min_slice_z_ = 0.0f, max_slice_z_ = 0.0f;
    float slice_vis_level_ = 4.5;
    double minx_dist_ = 0.0f, max_dist_ = 0.0f;

    float slice_vis_max_dist_ = 2.0;

    float voxel_size_ = 0.02;

    float offset_x_, offset_y_, offset_z_;

    // visualization esdf slice
    bool updateESDF_ = false;

    // Default distance set for unknown values and values > max_distance_m.
    float infinity_distance_m = 10.0; // infinity_
    bool use_fpv = false;

};

class Visualization {
public:
    Visualization(ros::NodeHandle& nh, std::shared_ptr<vdbblox_core::TSDFMap> pTSDFMap, std::shared_ptr<vdbblox_core::ESDFMap> pESDFMap);
    ~Visualization();
    void Visualize(const ros::TimerEvent& event);
    void publishTSDFPointCloud();
    bool publishESDFPointCloud();
    void publishESDFMapSlice();
    std_msgs::ColorRGBA RainbowColorMap(float h);

public:
    ros::NodeHandle nh_;
    std::shared_ptr<vdbblox_core::TSDFMap> pTSDFMap_;
    std::shared_ptr<vdbblox_core::ESDFMap> pESDFMap_;

    // global pcds
    std::mutex mutex_tsdf_pub_, mutex_esdf_pub_, mutex_slice_pub_;
    ros::Publisher tsdf_pub_, esdf_pub_, slice_pub_, pose_pub_, marker_pub_;
    sensor_msgs::PointCloud pcds_;

    VisConfig vis_cfg_;

    // TSDF Grid (deep copy) for visualization (map)
    vdbblox_core::TSDFVDBGrid::Ptr pTSDFVDBGridCopy_;
    std::shared_ptr<vdbblox_core::TSDFVDBGrid::Accessor> pTSDFVDBGridAccCopy_;

    // ESDF Grid (deep copy) for visualization (map)
    vdbblox_core::ESDFVDBGrid::Ptr pESDFVDBGridCopy_;
    std::shared_ptr<vdbblox_core::ESDFVDBGrid::Accessor> pESDFVDBGridAccCopy_;
    // ESDF Grid (deep copy) for visualization (slice)
    vdbblox_core::ESDFVDBGrid::Ptr pESDFVDBGridSlice_;
    std::shared_ptr<vdbblox_core::ESDFVDBGrid::Accessor> pESDFVDBGridAccSlice_;

    // update TSDFMap PointCloud
    std::atomic<bool> should_exit_tsdf_ = false;
    std::thread thread_update_tsdf_;

    // update ESDFMap Slice
    std::thread thread_update_esdf_slice_;
    std::atomic<bool> should_exit_esdf_ = false;
    std::size_t cnt_update_slice_ = 0;

public:
        static void visualizeBoundingBox(visualization_msgs::MarkerArray &mkrarr,
                                     const Eigen::Vector3f &box_min,
                                     const Eigen::Vector3f &box_max,
                                     const std::string &ns,
                                     const NormalizedRGB &color,
                                     const double &size_x = 0.1,
                                     const double &alpha = 1.0,
                                     const bool &print_ns = true) {
        Eigen::Vector3f size = (box_max - box_min) / 2;
        Eigen::Vector3f vis_pos_world = (box_min + box_max) / 2;
        double width = size.x();
        double length = size.y();
        double hight = size.z();

//Publish Bounding box
        int id = 0;
        visualization_msgs::Marker line_strip;
        line_strip.header.stamp = ros::Time::now();
        line_strip.header.frame_id = "world";
        line_strip.action = visualization_msgs::Marker::ADD;
        line_strip.ns = ns;
        line_strip.pose.orientation.w = 1.0;
        line_strip.id = id++; //unique id, useful when multiple markers exist.
        line_strip.type = visualization_msgs::Marker::LINE_STRIP; //marker type
        line_strip.scale.x = size_x;


        line_strip.color = color;
        line_strip.color.a = alpha; //不透明度，设0则全透明
        geometry_msgs::Point p[8];

        //vis_pos_world是目标物的坐标
        p[0].x = vis_pos_world(0) - width;
        p[0].y = vis_pos_world(1) + length;
        p[0].z = vis_pos_world(2) + hight;
        p[1].x = vis_pos_world(0) - width;
        p[1].y = vis_pos_world(1) - length;
        p[1].z = vis_pos_world(2) + hight;
        p[2].x = vis_pos_world(0) - width;
        p[2].y = vis_pos_world(1) - length;
        p[2].z = vis_pos_world(2) - hight;
        p[3].x = vis_pos_world(0) - width;
        p[3].y = vis_pos_world(1) + length;
        p[3].z = vis_pos_world(2) - hight;
        p[4].x = vis_pos_world(0) + width;
        p[4].y = vis_pos_world(1) + length;
        p[4].z = vis_pos_world(2) - hight;
        p[5].x = vis_pos_world(0) + width;
        p[5].y = vis_pos_world(1) - length;
        p[5].z = vis_pos_world(2) - hight;
        p[6].x = vis_pos_world(0) + width;
        p[6].y = vis_pos_world(1) - length;
        p[6].z = vis_pos_world(2) + hight;
        p[7].x = vis_pos_world(0) + width;
        p[7].y = vis_pos_world(1) + length;
        p[7].z = vis_pos_world(2) + hight;
        //LINE_STRIP类型仅仅将line_strip.points中相邻的两个点相连，如0和1，1和2，2和3
        for (int i = 0; i < 8; i++) {
            line_strip.points.push_back(p[i]);
        }
        //为了保证矩形框的八条边都存在：
        line_strip.points.push_back(p[0]);
        line_strip.points.push_back(p[3]);
        line_strip.points.push_back(p[2]);
        line_strip.points.push_back(p[5]);
        line_strip.points.push_back(p[6]);
        line_strip.points.push_back(p[1]);
        line_strip.points.push_back(p[0]);
        line_strip.points.push_back(p[7]);
        line_strip.points.push_back(p[4]);
        mkrarr.markers.push_back(line_strip);
    }

    /* Type B: Add marker to given marker_arr for later publish */
    static void visualizeText(visualization_msgs::MarkerArray &mkr_arr,
                              const std::string &ns,
                              const std::string &text,
                              const Eigen::Vector3f &position,
                              const NormalizedRGB &c = NormalizedRGB::White(),
                              const double &size = 0.6,
                              const int &id = -1) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.ns = ns.c_str();
        if (id >= 0) {
            marker.id = id;
        } else {
            static int id = 0;
            marker.id = id++;
        }
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        marker.scale.z = size;
        marker.color = c;
        marker.text = text;
        marker.pose.position.x = position.x();
        marker.pose.position.y = position.y();
        marker.pose.position.z = position.z();
        marker.pose.orientation.w = 1.0;
        mkr_arr.markers.push_back(marker);
    };
};
}  // namespace vdbblox_tools