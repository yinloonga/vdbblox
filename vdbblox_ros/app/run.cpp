#include <tools/common.h>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include "VDBbloxNode.h"
#include "ros/node_handle.h"
#include "ros/spinner.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "vdbblox_rosnode");
    ros::NodeHandle nh("~");

    std::shared_ptr<vdbblox_node::VDBbloxNode> pVDBbloxNode = std::make_shared<vdbblox_node::VDBbloxNode>(nh);

    ros::AsyncSpinner spinner(std::max(1, vdbblox_tools::getMaxNumIdealThreads()));
    spinner.start();
    ros::waitForShutdown();
}