// MIT License
//
// # Copyright (c) 2022 Saurabh Gupta, Ignacio Vizzo, Cyrill Stachniss, University of Bonn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <geometry_msgs/Transform.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Core>
#include <deque>
#include <mutex>

#include "ros/time.h"
#include "sophus/se3.hpp"

inline Sophus::SE3d TransformToSE3(const geometry_msgs::Transform& tf) {
    return {Eigen::Quaterniond{tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z},
            Eigen::Vector3d{tf.translation.x, tf.translation.y, tf.translation.z}};
}

inline geometry_msgs::Transform SE3ToTransform(const Sophus::SE3d T) {
    auto t = T.translation();
    auto q = T.unit_quaternion();

    geometry_msgs::Transform tf;
    tf.translation.x = t.x();
    tf.translation.y = t.y();
    tf.translation.z = t.z();
    tf.rotation.w = q.w();
    tf.rotation.x = q.x();
    tf.rotation.y = q.y();
    tf.rotation.z = q.z();
    return tf;
}

namespace vdbblox_core {
class Transform {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit Transform(ros::NodeHandle& nh);

    bool lookUpTransform(const ros::Time& timestamp,
                         const ros::Duration& tolerance,
                         geometry_msgs::TransformStamped& transform);

private:
    bool lookUpTransformTF2(const std::string& parent_frame,
                            const std::string& child_frame,
                            const ros::Time& timestamp,
                            const ros::Duration& tolerance,
                            geometry_msgs::TransformStamped& transform);

    bool lookUpTransformQ(const ros::Time& timestamp,
                          const ros::Duration& tolerance,
                          geometry_msgs::TransformStamped& transform);

    void tfCallback(const geometry_msgs::TransformStamped& transform_msg);

public:
    ros::Time getLatestTransformTime();

public:
    // Pose queue
    void pushPose(const Eigen::Vector3f& pose);
    Eigen::Vector3f& getFrontPoseFromQueue();
    Eigen::Vector3f& getBackPoseFromQueue();
    Eigen::Vector3f& atPoseFromQueue(const std::size_t& idx);
    void popPoseFromQueueFront();
    void popPoseFromQueueBack();

private:
    tf2_ros::Buffer buffer_;
    tf2_ros::TransformListener tf_;
    std::string parent_frame_;
    std::string child_frame_;

    bool use_tf2_;
    ros::Subscriber tf_sub_;
    std::deque<geometry_msgs::TransformStamped,
               Eigen::aligned_allocator<geometry_msgs::TransformStamped>>
        tf_queue_;
    geometry_msgs::TransformStamped static_tf_;

    std::mutex mutex_data_;

    std::mutex mutex_pose_queue_;
    std::deque<Eigen::Vector3f> pose_queue_;
};
}  // namespace vdbblox_core
