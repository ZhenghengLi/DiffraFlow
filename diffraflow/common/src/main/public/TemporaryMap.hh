#ifndef __TemporaryMap_H__
#define __TemporaryMap_H__

#include <vector>
#include <map>
#include <list>
#include <cassert>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>

using std::atomic_bool;
using std::map;
using std::list;
using std::pair;
using std::make_pair;
using std::thread;
using std::mutex;
using std::condition_variable;
using std::lock_guard;
using std::unique_lock;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::system_clock;
using std::milli;

namespace diffraflow {
    template <typename K, typename V>
    class TemporaryMap {
    public:
        TemporaryMap(size_t capacity = 100, double ttl = 1000 /* millisecond */);
        TemporaryMap(const TemporaryMap& tmap);
        virtual ~TemporaryMap();

        virtual void set(K key, V value);
        bool get(K key, V* value);

        void clear();
        size_t size();
        bool empty();

    private:
        void do_cleaning_();

    private:
        map<K, V> data_;
        size_t capacity_;
        double ttl_;
        list<pair<K, double>> key_list_;

        thread* cleaner_;
        atomic_bool is_running_;
        mutex data_mtx_;
        condition_variable clean_cv_;
    };

    template <typename K, typename V>
    TemporaryMap<K, V>::TemporaryMap(size_t capacity, double ttl) {
        capacity_ = (capacity > 100 ? capacity : 100);
        ttl_ = (ttl > 100 ? ttl : 100);
        is_running_ = true;
        cleaner_ = new thread([this]() {
            while (is_running_) {
                do_cleaning_();
            }
        });
    }

    template <typename K, typename V>
    TemporaryMap<K, V>::TemporaryMap(const TemporaryMap<K, V>& tmap) {
        // cannot be copied
        assert(false);
    }

    template <typename K, typename V>
    TemporaryMap<K, V>::~TemporaryMap() {
        is_running_ = false;
        clean_cv_.notify_all();
        cleaner_->join();
        delete cleaner_;
    }

    template <typename K, typename V>
    void TemporaryMap<K, V>::set(K key, V value) {
        lock_guard<mutex> lg(data_mtx_);
        // with new key
        if (data_.find(key) == data_.end()) {
            while (data_.size() >= capacity_) {
                pair<K, double> oldest = key_list_.front();
                data_.erase(oldest.first);
                key_list_.pop_front();
            }
        } else { // key already exists
            key_list_.remove_if([&key](pair<K, double> el) { return el.first == key; });
        }
        // add or update data with current time
        duration<double, milli> current_time = system_clock::now().time_since_epoch();
        key_list_.push_back(make_pair(key, current_time.count()));
        data_[key] = value;
        clean_cv_.notify_all();
    }

    template <typename K, typename V>
    bool TemporaryMap<K, V>::get(K key, V* value) {
        lock_guard<mutex> lg(data_mtx_);
        typename map<K, V>::iterator iter = data_.find(key);
        if (iter == data_.end()) {
            return false;
        } else {
            if (value != nullptr) {
                *value = iter->second;
            }
            return true;
        }
    }

    template <typename K, typename V>
    void TemporaryMap<K, V>::clear() {
        lock_guard<mutex> lg(data_mtx_);
        data_.clear();
        key_list_.clear();
    }

    template <typename K, typename V>
    bool TemporaryMap<K, V>::empty() {
        lock_guard<mutex> lg(data_mtx_);
        return key_list_.empty();
    }

    template <typename K, typename V>
    size_t TemporaryMap<K, V>::size() {
        lock_guard<mutex> lg(data_mtx_);
        return key_list_.size();
    }

    template <typename K, typename V>
    void TemporaryMap<K, V>::do_cleaning_() {
        unique_lock<mutex> ulk(data_mtx_);

        if (key_list_.empty()) {
            clean_cv_.wait(ulk, [this]() { return !is_running_; });
        }

        if (!is_running_) return;

        pair<K, double> oldest = key_list_.front();
        duration<double, milli> current_time = system_clock::now().time_since_epoch();
        double longest_live_time = current_time.count() - oldest.second;
        if (longest_live_time >= ttl_) {
            data_.erase(oldest.first);
            key_list_.pop_front();
        } else {
            clean_cv_.wait_for(ulk, milliseconds(static_cast<int>(ttl_ - longest_live_time)));
        }
    }

} // namespace diffraflow

#endif