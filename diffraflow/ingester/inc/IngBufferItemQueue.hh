#ifndef __IngBufferItemQueue_H__
#define __IngBufferItemQueue_H__

#include "BlockingQueue.hh"
#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

namespace diffraflow {

    struct IngBufferItem {
        explicit IngBufferItem(int idx = -1) : index(idx), rawdata(nullptr), save(false) {}
        shared_ptr<vector<char>> rawdata;
        int index;
        bool save;
    };

    typedef BlockingQueue<shared_ptr<IngBufferItem>> IngBufferItemQueue;

} // namespace diffraflow

#endif