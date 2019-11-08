#ifndef BlockingQueue_H
#define BlockingQueue_H

#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <condition_variable>

using std::queue;
using std::mutex;
using std::lock_guard;
using std::unique_lock;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;

namespace shine {
    template <typename E>
    class BlockingQueue {
    private:
        mutex mtx_;
        condition_variable cv_push_;
        condition_variable cv_take_;
        queue<E>* internal_queue_;
        atomic_bool aborted_;
        atomic<size_t> max_size_;
    public:
        BlockingQueue(size_t ms = 100);
        ~BlockingQueue();

        // synchronized methods
        void set_maxsize(size_t ms);
        size_t maxsize();

        size_t size();
        bool empty();

        // blocking push until have space, return false when aborted
        bool push(const E& el);
        // blocking push with timeout, return false when timeout or aborted
        bool push(const E& el, int timeout);
        // blocking take until have element, return false when aborted
        bool take(E& el);
        // blocking take with timeout, return false when timeout or aborted
        bool take(E& el, int timeout);
        // push only when have space now, otherwise directly return false
        bool offer(const E& el);
        // take only when have element now, otherwise directly return false
        bool get(E& el);
        // non-blocking push only when get lock and have space
        bool try_offer(const E& el);
        // non-blocking take only when get lock and have element
        bool try_get(E& el);

        void abort();
        bool aborted();
        void resume();
    };

    template <typename E>
    BlockingQueue<E>::BlockingQueue(size_t ms) {
        max_size_ = ms;
        internal_queue_ = new queue<E>();
        aborted_ = false;
    }

    template <typename E>
    BlockingQueue<E>::~BlockingQueue() {
        delete internal_queue_;
        internal_queue_ = nullptr;
    }

    template <typename E>
    void BlockingQueue<E>::set_maxsize(size_t ms) {
        max_size_ = ms;
    }

    template <typename E>
    size_t BlockingQueue<E>::maxsize() {
        return max_size_;
    }

    template <typename E>
    size_t BlockingQueue<E>::size() {
        lock_guard<mutex> lk(mtx_);
        return internal_queue_->size();
    }

    template <typename E>
    bool BlockingQueue<E>::empty() {
        lock_guard<mutex> lk(mtx_);
        return internal_queue_->empty();
    }

    template <typename E>
    bool BlockingQueue<E>::push(const E& el) {
        unique_lock<mutex> lk(mtx_);
        cv_push_.wait(lk, [&]() {return aborted_ || internal_queue_->size() < max_size_;});
        if (aborted_) {
            return false;
        }
        internal_queue_->push(el);
        cv_take_.notify_one();
        return true;
    }

    template <typename E>
    bool BlockingQueue<E>::push(const E& el, int timeout) {
        if (timeout < 0) return false;
        unique_lock<mutex> lk(mtx_);
        if (cv_push_.wait_for(lk, std::chrono::milliseconds(timeout),
            [&]() {return aborted_ || internal_queue_->size() < max_size_;})) {
            if (aborted_) {
                return false;
            }
            internal_queue_->push(el);
            cv_take_.notify_one();
            return true;
        } else {
            // timeout
            return false;
        }
    }

    template <typename E>
    bool BlockingQueue<E>::take(E& el) {
        unique_lock<mutex> lk(mtx_);
        cv_take_.wait(lk, [&]() {return aborted_ || !internal_queue_->empty();});
        if (aborted_) {
            return false;
        }
        el = internal_queue_->front();
        internal_queue_->pop();
        cv_push_.notify_one();
        return true;
    }

    template <typename E>
    bool BlockingQueue<E>::take(E& el, int timeout) {
        if (timeout < 0) return false;
        unique_lock<mutex> lk(mtx_);
        if (cv_take_.wait_for(lk, std::chrono::milliseconds(timeout),
            [&]() {return aborted_ || !internal_queue_->empty();})) {
            if (aborted_) {
                return false;
            }
            el = internal_queue_->front();
            internal_queue_->pop();
            cv_push_.notify_one();
            return true;
        } else {
            // timeout
            return false;
        }
    }

    template <typename E>
    bool BlockingQueue<E>::offer(const E& el) {
        lock_guard<mutex> lk(mtx_);
        if (internal_queue_->size() < max_size_) {
            internal_queue_->push(el);
            return true;
        } else {
            return false;
        }
    }

    template <typename E>
    bool BlockingQueue<E>::get(E& el) {
        lock_guard<mutex> lk(mtx_);
        if (internal_queue_->empty()) {
            return false;
        } else {
            el = internal_queue_->front();
            internal_queue_->pop();
        }
    }

    template <typename E>
    bool BlockingQueue<E>::try_offer(const E& el) {
        unique_lock<mutex> lk(mtx_, std::try_to_lock);
        if (!lk.owns_lock()) return false;
        if (internal_queue_->size() < max_size_) {
            internal_queue_->push(el);
            return true;
        } else {
            return false;
        }
    }

    template <typename E>
    bool BlockingQueue<E>::try_get(E& el) {
        unique_lock<mutex> lk(mtx_, std::try_to_lock);
        if (!lk.owns_lock()) return false;
        if (internal_queue_->empty()) {
            return false;
        } else {
            el = internal_queue_->front();
            internal_queue_->pop();
        }
    }

    template <typename E>
    void BlockingQueue<E>::abort() {
        aborted_ = true;
        cv_push_.notify_all();
        cv_take_.notify_all();
    }

    template <typename E>
    bool BlockingQueue<E>::aborted() {
        return aborted_;
    }

    template <typename E>
    void BlockingQueue<E>::resume() {
        aborted_ = false;
    }

}

#endif