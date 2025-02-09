#include "core/ESDFMap.h"
#include <libmorton/morton.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <openvdb/Types.h>
#include <tools/Visualization.h>
#include <tools/common.h>
#include <unistd.h>
#include <visualization_msgs/MarkerArray.h>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <ostream>
#include <shared_mutex>
#include <string>
#include <utility>
#include "cell/ESDFCell.hpp"
#include "core/TSDFMap.h"
#include "ros/console.h"
#include "ros/forwards.h"
#include "ros/node_handle.h"
#include "tools/Timing.h"

vdbblox_core::ESDFMap::ESDFMap(ros::NodeHandle& nh, std::shared_ptr<TSDFMap> pTSDFMap) : nh_(nh), pTSDFMap_(pTSDFMap) {
    // Constructor
    ROS_INFO("ESDFMap Constructor");

    esdf_cfg_.voxel_size_ = pTSDFMap_->tsdf_cfg_.voxel_size_;
    esdf_cfg_.sdf_trunc_ = pTSDFMap_->tsdf_cfg_.sdf_trunc_;

    esdf_cfg_.offset_x_ = pTSDFMap_->tsdf_cfg_.offset_x_;
    esdf_cfg_.offset_y_ = pTSDFMap_->tsdf_cfg_.offset_y_;
    esdf_cfg_.offset_z_ = pTSDFMap_->tsdf_cfg_.offset_z_;

    esdf_cfg_.update_range_ = pTSDFMap_->tsdf_cfg_.update_range_;
    esdf_cfg_.update_range_z_ = pTSDFMap_->tsdf_cfg_.update_range_z_;
    
    esdf_cfg_.space_carving_ = pTSDFMap_->tsdf_cfg_.space_carving_;
    esdf_cfg_.update_local_ = pTSDFMap_->tsdf_cfg_.update_local_;

    // ROS Parameters
    nh_.getParam("/infinity_distance_m", esdf_cfg_.infinity_distance_m);

    // ESDF VDB
    pESDFVDBGrid_ = ESDFVDBGrid::create(vdbblox_core::ESDFCell(false));
    pESDFVDBGrid_->setName("ESDFVDBGrid");
    pESDFVDBGrid_->setTransform(openvdb::math::Transform::createLinearTransform(pTSDFMap_->tsdf_cfg_.voxel_size_));
    pESDFVDBGrid_->setGridClass(openvdb::GRID_LEVEL_SET);
    pESDFVDBGridAcc_ = std::make_shared<ESDFVDBGrid::Accessor>(pESDFVDBGrid_->getAccessor());

    // Default ESDF Cell
    default_esdf_cell_ = vdbblox_core::ESDFCell(false);

    if (esdf_cfg_.space_carving_) thread_update_free_space_ = std::thread(&ESDFMap::updateFreeSpace, this);

    if (esdf_cfg_.update_local_) {
        Eigen::Vector3f left_bottom(-esdf_cfg_.update_range_, -esdf_cfg_.update_range_, -esdf_cfg_.update_range_z_);
        left_bottom /= esdf_cfg_.voxel_size_;
        Eigen::Vector3f right_top(esdf_cfg_.update_range_, esdf_cfg_.update_range_, esdf_cfg_.update_range_z_);
        right_top /= esdf_cfg_.voxel_size_;
        std::size_t range_num_voxels = static_cast<std::size_t>((right_top.x() - left_bottom.x() + 1) *
                                                                 (right_top.y() - left_bottom.y() + 1) *
                                                                 (right_top.z() - left_bottom.z() + 1));
        range_num_voxles_ = range_num_voxels;
    }
}

vdbblox_core::ESDFMap::~ESDFMap() {

}

bool vdbblox_core::ESDFMap::setUpdateRange() {
    // const Eigen::Vector3f& cur_pose = pTSDFMap_->pTf_->getBackPoseFromQueue();  // get the current pose
    if (pTSDFMap_->pTransformer_->isPoseQeueEmpty()) {
        ROS_ERROR("[vdbblox_core::ESDFMap::setUpdateRange] Pose Queue is empty!");
        return false;
    }

    const Eigen::Vector3f& cur_pose = pTSDFMap_->pTransformer_->getBackPoseFromQueue();  // get the latest tf
    if (!cur_pose.allFinite()) {
        ROS_ERROR("[vdbblox_core::ESDFMap::setUpdateRange] Transformation matrix is invalid!");
        return false;
    }

    // udpate the left-bottom corner 
    esdf_cfg_.left_bottom_ = cur_pose - Eigen::Vector3f(esdf_cfg_.update_range_, esdf_cfg_.update_range_, esdf_cfg_.update_range_z_);
    esdf_cfg_.left_bottom_.z() = esdf_cfg_.offset_z_;
    esdf_cfg_.left_bottom_ /= esdf_cfg_.voxel_size_;
    esdf_cfg_.left_bottom_ = (esdf_cfg_.left_bottom_.array() < 0).select(0, esdf_cfg_.left_bottom_);

    // update the right-top corner
    esdf_cfg_.right_top_ = cur_pose + Eigen::Vector3f(esdf_cfg_.update_range_, esdf_cfg_.update_range_, esdf_cfg_.update_range_z_);
    esdf_cfg_.right_top_ /= esdf_cfg_.voxel_size_;
    esdf_cfg_.right_top_ = (esdf_cfg_.right_top_.array() < 0).select(0, esdf_cfg_.right_top_);

    return true;
}

bool vdbblox_core::ESDFMap::isInRange(const openvdb::Coord& coord) {
    if (esdf_cfg_.update_local_) {
        return (coord.x() >= esdf_cfg_.left_bottom_.x() && coord.x() <= esdf_cfg_.right_top_.x() &&
                coord.y() >= esdf_cfg_.left_bottom_.y() && coord.y() <= esdf_cfg_.right_top_.y() &&
                coord.z() >= esdf_cfg_.left_bottom_.z() && coord.z() <= esdf_cfg_.right_top_.z());
    } else {
        return false;
    }
}

void vdbblox_core::ESDFMap::updateFreeSpace() {
    // update the free space
    while (!should_exit_.load()) {

        tbb::concurrent_vector<openvdb::Coord> freespace_tsdf;        
        std::unique_lock<std::mutex> lock(pTSDFMap_->mtx_freespace_tsdf_);
        pTSDFMap_->cv_freespace_tsdf_.wait(lock, [&] { return !pTSDFMap_->freespace_tsdf_.empty() || should_exit_.load(); });
        if (should_exit_.load()) break;

        vdbblox_tools::Timer updateFreeSpace("updateESDFMap-updateFreeSpace");
        if (esdf_cfg_.update_local_) freespace_tsdf.reserve(pTSDFMap_->freespace_tsdf_.size() + range_num_voxles_);
        freespace_tsdf = std::move(pTSDFMap_->freespace_tsdf_);
        pTSDFMap_->freespace_tsdf_.clear();
        // update the local range free space
        std::atomic<std::size_t> cnt(0);
        if (esdf_cfg_.update_local_) {
            if (pTSDFMap_->pTransformer_->isPoseQeueEmpty()) {
                ROS_ERROR("[vdbblox_core::ESDFMap::updateFreeSpace] Pose Queue is empty!");
                continue;
            }
            const Eigen::Vector3f cur_pose = pTSDFMap_->pTransformer_->getBackPoseFromQueue();  // get the latest tf
            // update the left-bottom corner
            Eigen::Vector3f left_bottom = cur_pose - Eigen::Vector3f(esdf_cfg_.update_range_, esdf_cfg_.update_range_, esdf_cfg_.update_range_z_);
            // fix the z-axis
            left_bottom.z() = esdf_cfg_.offset_z_;
            left_bottom /= esdf_cfg_.voxel_size_;
            left_bottom = (left_bottom.array() < 0).select(0, left_bottom);
            // update the right-top corner
            Eigen::Vector3f right_top = cur_pose + Eigen::Vector3f(esdf_cfg_.update_range_, esdf_cfg_.update_range_, esdf_cfg_.update_range_z_);
            right_top /= esdf_cfg_.voxel_size_;
            right_top = (right_top.array() < 0).select(0, right_top);
            std::size_t left_x = static_cast<std::size_t>(left_bottom.x()), left_y = static_cast<std::size_t>(left_bottom.y()), left_z = static_cast<std::size_t>(left_bottom.z());
            std::size_t right_x = static_cast<std::size_t>(right_top.x()), right_y = static_cast<std::size_t>(right_top.y()), right_z = static_cast<std::size_t>(right_top.z());
            // update the voxel from left_bottom to right_top
            for (std::size_t x = left_x; x <= right_x; ++x) {
                for (std::size_t y = left_y; y <= right_y; ++y) {
                    for (std::size_t z = left_z; z <= right_z; ++z) {
                        freespace_tsdf.emplace_back(openvdb::Coord(x, y, z));
                        cnt.fetch_add(1);
                    }
                }
            }
        }

        std::atomic<std::size_t> cnt_set(0);
        {
            std::unique_lock<std::shared_mutex> lock_update_esdf(mutex_update_esdf_);
            for (const auto& coord : freespace_tsdf) {
                if (pESDFVDBGridAcc_->isValueOn(coord)) continue;
                pESDFVDBGridAcc_->setValue(coord, vdbblox_core::ESDFCell(false));
                vdbblox_core::ESDFCell& esdf_cell = const_cast<vdbblox_core::ESDFCell&>(pESDFVDBGridAcc_->getValue(coord));
                esdf_cell.self_idx = libmorton::morton3D_32_encode(coord.x(), coord.y(), coord.z());
                esdf_cell.raw_distance = esdf_cfg_.infinity_distance_m;
                cnt_set.fetch_add(1);
            }
        }
        freespace_tsdf.clear();
        updateFreeSpace.Stop();

        if (!pTSDFMap_->use_sync_) continue;
        {
            std::lock_guard<std::mutex> lock(pTSDFMap_->mtx_freespace_tsdf_);
            pTSDFMap_->sync_freespace_tsdf_ = true;
        }
    }
}


void vdbblox_core::ESDFMap::Integrate(const ros::TimerEvent& event) {
    if (!pTSDFMap_->tsdf_cfg_.init_mapping_) return;
    std::lock_guard<std::mutex> lock(this->mtx_integrate_);

    std::list<std::pair<openvdb::Coord, float/* distance */>> lru_cache;
    {
        std::lock_guard<std::mutex> lock_lru_cache(pTSDFMap_->mutex_lru_);
        lru_cache = std::move(pTSDFMap_->lru_cache_);
        pTSDFMap_->init_cache_ = true;
    }
    if (lru_cache.empty()) return;  // no new data

    vdbblox_tools::Timer Integrate("updateESDFMap-Integrate"); 
    std::size_t cnt_newly = 0, cnt_observed = 0;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_update_esdf_);
        for (const auto& [coord, tsdf] : lru_cache) {
            
            if (!pESDFVDBGridAcc_->isValueOn(coord)) pESDFVDBGridAcc_->setValue(coord, vdbblox_core::ESDFCell());
            vdbblox_core::ESDFCell& esdf_cell = const_cast<vdbblox_core::ESDFCell&>(pESDFVDBGridAcc_->getValue(coord));
            bool curr_occupied = false;

            if (esdf_cfg_.finer_esdf_on) {
                // determine occupancy state with tsdf and gradient (not thread-safe)
                // curr_occupied = isOccupied(tsdf_cell.tsdf, tsdf_cell.gradient);
            } else {
                // determine occupancy state with only the tsdf
                curr_occupied = isOccupied(tsdf);
            }

            if (esdf_cell.self_idx == UINT_MAX) {  // not yet initialized
                cnt_newly++;

                esdf_cell.observed = true;
                esdf_cell.newly = true;
                esdf_cell.self_idx = libmorton::morton3D_32_encode(coord.x(), coord.y(), coord.z());
                esdf_cell.raw_distance = esdf_cfg_.infinity_distance_m;

                // Newly found occupied --> insert_list
                if (curr_occupied) insert_list_.emplace_back(coord);
                // pESDFVDBGridAcc_->setActiveState(coord, true);

            } else {  // already initialized
                cnt_observed++;
                esdf_cell.newly = false;

                if (curr_occupied != esdf_cell.occupied) {
                    if (curr_occupied) {  // Originally not occupied but occupied now
                        insert_list_.emplace_back(coord);
                        // pESDFVDBGridAcc_->setActiveState(coord, true);
                    } else {  // Originally occupied but not occupied now
                        delete_list_.emplace_back(coord);
                        // pESDFVDBGridAcc_->setActiveState(coord, true);
                    }
                } else {  // No change in occupancy
                    // pESDFVDBGridAcc_->setActiveState(coord, curr_occupied);
                    // pESDFVDBGridAcc_->setActiveState(coord, true);
                }

            }

            esdf_cell.distance = tsdf;
            esdf_cell.occupied = curr_occupied;
            // pESDFVDBGridAcc_->setValueOnly(coord, esdf_cell);
            pESDFVDBGridAcc_->setValue(coord, esdf_cell);

        }
    }

    Integrate.Stop();
    ROS_INFO("[ESDFMapIntegrate]==> LRU Cache: %u, Insert List: %u, Delete List: %u, Newly: %u, Observed: %u", static_cast<int>(lru_cache.size()), static_cast<int>(insert_list_.size()), 
                static_cast<int>(delete_list_.size()), static_cast<int>(cnt_newly), static_cast<int>(cnt_observed));
    
    updateESDFMap();
}


void vdbblox_core::ESDFMap::updateESDFMap() {
    if (esdf_cfg_.update_local_) {
        bool bUpateRange = setUpdateRange();
        if (!bUpateRange) {
            ROS_ERROR("[vdbblox_core::ESDFMap::updateESDFMap] Update Range is invalid!");
            return;
        }
    }
    processInsertList();
    processDeleteList();
    processUpdateList();
    {
        this->cv_update_done_.notify_one();
        this->update_done_ = true;
    }
}

bool vdbblox_core::ESDFMap::processInsertList() {
    std::lock_guard<std::mutex> lock(mutex_process_insert_);
    vdbblox_tools::Timer ProcessInsertList("updateESDFMap-ProcessInsertList");

    std::size_t times_insert = 0;
    while (!insert_list_.empty()) {
        times_insert++;

        openvdb::Coord cur_coord = *insert_list_.begin();
        insert_list_.pop_front();
        auto& cur_cell = CoordDecoder(cur_coord, true, true);
        if (!cur_cell.occupied) continue;

        if (cur_cell.coc_idx != UINT_MAX) {
            auto [coc_cur_coord, coc_cur_cell] = IndexDecoder(cur_cell.coc_idx, true, true);
            deleteFromList(coc_cur_cell, cur_cell);
        }
        cur_cell.raw_distance = 0.0f;
        cur_cell.coc_idx = cur_cell.self_idx;
        insertIntoList(cur_cell, cur_cell);

        update_tsdfmap_.emplace_back(cur_coord, cur_cell.raw_distance);
        update_list_.emplace(cur_cell.raw_distance/* priority (distance) */, cur_coord/* coord */);
    }
    ProcessInsertList.Stop();
    return true;
}

bool vdbblox_core::ESDFMap::processDeleteList() {
    std::lock_guard<std::mutex> lock(mutex_process_delete_);
    vdbblox_tools::Timer ProcessDeleteList("updateESDFMap-ProcessDeleteList");
    std::size_t times_delete = 0, init_delete_list_size = delete_list_.size(), times_update = 0;

    std::cerr << "[delete_list_] size: " << delete_list_.size() << ", curr cnt: " << std::flush;
    while (!delete_list_.empty()) {
        if (times_delete++ %1000 == 0) std::cerr <<  std::to_string(times_delete) + " ";

        openvdb::Coord cur_coord = *delete_list_.begin();
        delete_list_.pop_front();
        auto& cur_cell = CoordDecoder(cur_coord, true, true);
        if (cur_cell.occupied) continue;

        std::uint32_t next_tmp_idx = UINT_MAX; bool change = false;  std::uint32_t cnt_cnt_delete = 0;
        for (std::uint32_t tmp_idx = cur_cell.head_idx; tmp_idx != UINT_MAX; tmp_idx = next_tmp_idx) {  // check the in DDL voxle the current voxel

            if (cnt_cnt_delete++ > 100000) {
                ROS_ERROR("[Delete] cnt_cnt_delete: %u", cnt_cnt_delete);
                std::exit(0);
            }

            auto [tmp_coord, tmp_cell] = IndexDecoder(tmp_idx, true, true);
            tmp_cell.coc_idx = UINT_MAX;

            float distance = esdf_cfg_.infinity_distance_m;
            for (const openvdb::Coord& dir: dirs_) {  // update neighbor voxels in DDL of delete_list
                openvdb::Coord tmp_nbr_coord = tmp_coord + dir;
                // if (!isInRange(tmp_nbr_coord)) continue;
                auto& tmp_nbr_cell = CoordDecoder(tmp_nbr_coord, true, true);

                if (tmp_nbr_cell.coc_idx == UINT_MAX) continue;
                auto [tmp_coc_nbr_coord, tmp_coc_nbr_cell] = IndexDecoder(tmp_nbr_cell.coc_idx, true, true);
                if (!tmp_coc_nbr_cell.occupied) continue;

                float tmp_dist = getDistance(tmp_coord, tmp_coc_nbr_coord);
                if (tmp_dist < distance) {
                    distance = tmp_dist;
                    tmp_cell.coc_idx = tmp_nbr_cell.coc_idx;
                    change = true;
                }
                break;
                    
            }
            // destroy the linked-list
            next_tmp_idx = tmp_cell.next_idx;
            tmp_cell.prev_idx = tmp_cell.next_idx = UINT_MAX;
            tmp_cell.raw_distance = distance;

            if (distance < esdf_cfg_.infinity_distance_m) {
                update_tsdfmap_.emplace_back(tmp_coord, tmp_cell.raw_distance);
                update_list_.emplace(tmp_cell.raw_distance, tmp_coord);
                times_update++;
            }

            if (tmp_cell.coc_idx == UINT_MAX) continue;
            auto [tmp_coc_coord, tmp_coc_cell] = IndexDecoder(tmp_cell.coc_idx, true, true);
            insertIntoList(tmp_coc_cell, tmp_cell);

        }
        cur_cell.head_idx = UINT_MAX;
    }
    std::cerr << std::endl;
    ProcessDeleteList.Stop();
    return true;
}

bool vdbblox_core::ESDFMap::processUpdateList() {
    std::lock_guard<std::mutex> lock(mutex_process_update_);
    vdbblox_tools::Timer ProcessUpdateList("updateESDFMap-ProcessUpdateList");
    std::cerr << "[update_list_] size: " << update_list_.size() << ", curr cnt: " << std::flush;

    std::size_t times_update = 0, init_update_list_size = update_list_.size(), add_new_cur = 0, add_new_coc = 0;
    while (!update_list_.empty()) {
        openvdb::Coord cur_coord = update_list_.begin()->second;
        auto& cur_cell = CoordDecoder(cur_coord, true, true);
        assert(cur_cell.coc_idx != UINT_MAX);
        float cur_dist = update_list_.begin()->first;
        update_list_.erase(update_list_.begin());
        if (cur_cell.raw_distance != cur_dist) continue;

        if (times_update++ % 1000 == 0) std::cerr << std::to_string(times_update) << " " << std::flush; 
        // update the current voxel in update_list from neighbor voxels
        bool change = false;
        for (int i = 0; i < num_dirs_; ++i) {
            openvdb::Coord nbr_cur_coord = cur_coord + dirs_[i];
            // if (!isInRange(nbr_cur_coord)) continue;
            auto& nbr_cur_cell = CoordDecoder(nbr_cur_coord, true, true);
            if (nbr_cur_cell.coc_idx == UINT_MAX) continue;
            auto [coc_nbr_coord, coc_nbr_cell] = IndexDecoder(nbr_cur_cell.coc_idx, true, true);

            float tmp_dist = getDistance(cur_coord, coc_nbr_coord);
            if (tmp_dist < cur_cell.raw_distance) {

                if (cur_cell.coc_idx != UINT_MAX) { 
                    auto [coc_cur_coord, coc_cur_cell] = IndexDecoder(cur_cell.coc_idx, true, true);
                    if (coc_cur_cell.occupied) deleteFromList(coc_cur_cell, cur_cell);
                }
                if (!coc_nbr_cell.occupied) continue;
                insertIntoList(coc_nbr_cell, cur_cell);

                cur_cell.raw_distance = tmp_dist;
                cur_cell.coc_idx = nbr_cur_cell.coc_idx;

                change = true;
                update_list_.emplace(cur_cell.raw_distance, cur_coord);
                update_tsdfmap_.emplace_back(cur_coord, cur_cell.raw_distance);
                add_new_cur++;
            }
        }

        if (change) continue;

        auto [coc_cur_coord, coc_cur_cell] = IndexDecoder(cur_cell.coc_idx, true, true);
        // update neighbor voxels from the current voxel in update_list 
        for (const auto& dir : dirs_) {
            openvdb::Coord nbr_cur_coord = cur_coord + dir;
            bool is_InRange = isInRange(nbr_cur_coord);
            if (!is_InRange) continue;

            auto& nbr_cur_cell = CoordEncoder(nbr_cur_coord);
            if (nbr_cur_cell.raw_distance < 0) {
                ROS_ERROR("[Update] nbr_cur_cell.raw_distance: %f", nbr_cur_cell.raw_distance);
                // continue;
            }

            float tmp_dist = getDistance(nbr_cur_coord, coc_cur_coord);
            if (tmp_dist < nbr_cur_cell.raw_distance) {

                if (nbr_cur_cell.coc_idx != UINT_MAX) { 
                    auto [coc_nbr_coord, coc_nbr_cell] = IndexDecoder(nbr_cur_cell.coc_idx, true, true);
                    if (coc_nbr_cell.occupied) deleteFromList(coc_nbr_cell, nbr_cur_cell);
                }
                if (!coc_cur_cell.occupied) continue;
                insertIntoList(coc_cur_cell, nbr_cur_cell);

                nbr_cur_cell.raw_distance = tmp_dist;
                nbr_cur_cell.coc_idx = cur_cell.coc_idx;
                update_list_.emplace(nbr_cur_cell.raw_distance, nbr_cur_coord);
                update_tsdfmap_.emplace_back(nbr_cur_coord, nbr_cur_cell.raw_distance);
                add_new_coc++;
            }
        }
    }
    std::cerr << std::endl;
    ProcessUpdateList.Stop();
    return true;
}

void vdbblox_core::ESDFMap::deleteFromList(vdbblox_core::ESDFCell& occ_cell/* old obstacle of cur cell */, vdbblox_core::ESDFCell& cur_cell/* needed update */){
    if (cur_cell.prev_idx != UINT_MAX) {
        auto [prev_coord, cur_prev_cell] = IndexDecoder(cur_cell.prev_idx, true, true);
        cur_prev_cell.next_idx = cur_cell.next_idx;
    } else {
        occ_cell.head_idx = cur_cell.next_idx;
    }

    if (cur_cell.next_idx != UINT_MAX) {
        auto [next_coord, cur_next_cell] = IndexDecoder(cur_cell.next_idx, true, true);
        cur_next_cell.prev_idx = cur_cell.prev_idx;
    }
    // reset the cur_cell
    cur_cell.prev_idx = cur_cell.next_idx = UINT_MAX;
}

void vdbblox_core::ESDFMap::insertIntoList(vdbblox_core::ESDFCell& occ_cell/* new obstacle of cur cell */, vdbblox_core::ESDFCell& cur_cell/* needed update */) {
    if (occ_cell.head_idx == UINT_MAX) {
        occ_cell.head_idx = cur_cell.self_idx;
    } else {
        auto [occ_head_coord, occ_head_cell] = IndexDecoder(occ_cell.head_idx, true, true);
        occ_head_cell.prev_idx = cur_cell.self_idx;  // <-
        cur_cell.next_idx = occ_cell.head_idx;  // ->
        occ_cell.head_idx = cur_cell.self_idx;
    }
}
