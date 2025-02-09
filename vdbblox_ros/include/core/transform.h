/**
 * Copyright (C) 2022, IADC, Hong Kong University of Science and Technology
 * MIT License
 * Author: Kin ZHANG (https://kin-zhang.github.io/)
 * Date: 2022-11-05
 * Description: callback on the transformer, save the pose according to the pt
 * time Reference: mainly from the voxblox with slightly modification.
 */

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <cmath>

#include <Eigen/Core>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_listener.h>

#include "kindr/minimal/quat-transformation.h"

namespace vdbblox_core {

typedef kindr::minimal::QuatTransformationTemplate<double> kindrQuatT;
using Transformation = kindrQuatT;
typedef kindr::minimal::RotationQuaternionTemplate<double> kindrRotaT;
using Rotation = kindrRotaT;
typedef std::pair<ros::Time, std::vector<double>> pairTP;
inline void transform2Eigen(Eigen::Matrix4d& res, double tx, double ty, double tz, double w, double x, double y, double z) {
    Eigen::Vector3d t(tx, ty, tz);
    Eigen::Quaterniond q(w, x, y, z);

    res.block<3, 3>(0, 0) = q.toRotationMatrix();
    res.block<3, 1>(0, 3) = t;
};
inline void SixVector2Eigen(Eigen::Matrix4d& res, double tx, double ty, double tz, double roll, double pitch, double yaw) {
    Eigen::Vector3d t(tx, ty, tz);
    Eigen::Quaterniond q;
    q = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    res.block<3, 3>(0, 0) = q.toRotationMatrix();
    res.block<3, 1>(0, 3) = t;
};

inline void kindrT2Eigen(Eigen::Matrix4d& res, kindrQuatT& input) {
    res.block<3, 3>(0, 0) = input.getRotationMatrix();
    res.block<3, 1>(0, 3) = input.getPosition();
};

inline void Eigen2KindrT(Eigen::Matrix4d& input, Transformation& res) {
    Eigen::Matrix3d rotation_matrix = input.block<3, 3>(0, 0);
    res = Transformation(Rotation(rotation_matrix), Eigen::Matrix<double, 3, 1>(input.block<3, 1>(0, 3)));
};

class Transformer {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit Transformer(const ros::NodeHandle& nh);

    ~Transformer() = default;

    bool lookUpTransformfromSource(const sensor_msgs::PointCloud2::ConstPtr& input, Eigen::Matrix4d& tf_matrix);
    inline std::string getWorldframe() { return world_frame_; };

   private:
    bool tfTreeRead(const ros::Time& timestamp, const ros::Duration& tolerance, geometry_msgs::TransformStamped& transform);

    void tfCallback(const geometry_msgs::TransformStamped& input);

    void odomCallback(const nav_msgs::Odometry::ConstPtr& input);

public:
    void pushPose(const Eigen::Vector3f& pose);
    Eigen::Vector3f& getFrontPoseFromQueue();
    Eigen::Vector3f& getBackPoseFromQueue();
    Eigen::Vector3f& atPoseFromQueue(const std::size_t& idx);
    void popPoseFromQueueFront();
    void popPoseFromQueueBack();
    bool isPoseQeueEmpty();
    std::size_t getPoseQueueSize();

    std::mutex mutex_pose_queue_;
    std::deque<Eigen::Vector3f/* origin */> pose_queue_;

    ros::NodeHandle nh_;

    tf::TransformListener tf_listener_;
    tf2_ros::Buffer buffer_;
    tf2_ros::TransformListener tf_;

    ros::Subscriber tf_sub_, odom_sub_;

    // save all pose queue
    std::vector<std::pair<ros::Time, std::vector<double>>> TimeWithPose_queue;
    kindr::minimal::QuatTransformationTemplate<double> static_tf;

    int64_t tolerance_ns_;
    float pose_source_ = 0;  // [0:tf_tree, 1:tf_topic, 2:odom_topic]

    std::string world_frame_ = "map";
    std::string child_frame_ = "lidar";
};

}  // namespace vdbblox_core

#endif  // TRANSFORM_H