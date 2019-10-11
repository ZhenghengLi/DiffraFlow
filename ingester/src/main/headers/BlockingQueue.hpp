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

namespace shine {
    template <typename E>
    class BlockingQueue {
    private:
        mutex mtx_;
        condition_variable cv_push_;
        condition_variable cv_take_;
        queue<E>* internal_queue_;
        bool aborted_;
        size_t max_size_;
    public:
        BlockingQueue(size_t ms = 100);
        ~BlockingQueue();

        // synchronized methods
        void set_maxsize(size_t ms);
        size_t size();
        bool   empty();
        bool   push(const E& el);
        bool   push(const E& el, int timeout);
        bool   take(E& el);
        bool   take(E& el, int timeout);
        bool   offer(const E& el);
        bool   get(E& el);
        void   abort();
        bool   aborted();
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
        cv_push_.wait(lk, [&]() {return  aborted_ || internal_queue_->size() < max_size_;});
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
            [&]() {return  aborted_ || internal_queue_->size() < max_size_;})) {
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
        cv_take_.wait(lk, [&]() {return  aborted_ || !internal_queue_->empty();});
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
            [&]() {return  aborted_ || !internal_queue_->empty();})) {
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
    void BlockingQueue<E>::abort() {
        lock_guard<mutex> lk(mtx_);
        aborted_ = true;
        cv_push_.notify_all();
        cv_take_.notify_all();
    }

    template <typename E>
    bool BlockingQueue<E>::aborted() {
        lock_guard<mutex> lk(mtx_);
        return aborted_;
    }

}

#endif