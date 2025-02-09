#pragma once

#include <Eigen/Core>
#include <vector>
#include <queue>

namespace vdbblox_tools {
/**
 * Bucketed priority queue, mostly following L. Yatziv et al in
 * O(N) Implementation of the Fast Marching Algorithm, though skipping the
 * circular aspect (don't care about a bit more memory used for this).
 */
template <typename T>
class BucketQueue {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BucketQueue() : last_bucket_index_(0) {}
    explicit BucketQueue(std::uint32_t num_buckets, float max_val)
        : num_buckets_(num_buckets), max_val_(max_val), last_bucket_index_(0), num_elements_(0) {
        buckets_.resize(num_buckets_);
    }

    /// WARNING: will CLEAR THE QUEUE!
    void setNumBuckets(std::uint32_t num_buckets, float max_val) {
        max_val_ = max_val;
        num_buckets_ = num_buckets;
        buckets_.clear();
        buckets_.resize(num_buckets_);
        num_elements_ = 0;
    }

    void push(const T& key, float value) {
        assert(num_buckets_ != 0);
        if (value > max_val_) {
            value = max_val_;
        }
        std::uint32_t bucket_index = std::floor(std::abs(value) / max_val_ * (num_buckets_ - 1));
        if (bucket_index >= num_buckets_) {
            bucket_index = num_buckets_ - 1;
        }
        if (bucket_index < last_bucket_index_) {
            last_bucket_index_ = bucket_index;
        }
        buckets_[bucket_index].push(key);
        num_elements_++;
    }

    void pop() {
        if (empty()) {
            return;
        }
        while (buckets_[last_bucket_index_].empty() && last_bucket_index_ < num_buckets_) {
            last_bucket_index_++;
        }
        if (last_bucket_index_ < num_buckets_) {
            buckets_[last_bucket_index_].pop();
            num_elements_--;
        }
    }

    T front() {
        assert(num_buckets_ != 0);
        assert(!empty());
        while (buckets_[last_bucket_index_].empty() && last_bucket_index_ < num_buckets_) {
            last_bucket_index_++;
        }
        return buckets_[last_bucket_index_].front();
    }

    bool empty() { return num_elements_ == 0; }

    std::uint32_t size() { return num_elements_; }

    void clear() {
        buckets_.clear();
        buckets_.resize(num_buckets_);
        last_bucket_index_ = 0;
        num_elements_ = 0;
    }

private:
    std::uint32_t num_buckets_;
    float max_val_;
    std::vector<std::queue<T>> buckets_;

    /// Speeds up retrivals.
    std::uint32_t last_bucket_index_;
    /// This is also to speed up empty checks.
    std::uint32_t num_elements_;
};
}  // namespace vdbblox_tools
