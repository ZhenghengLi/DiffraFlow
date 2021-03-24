#ifndef __IngBufferItemQueue_H__
#define __IngBufferItemQueue_H__

#include "BlockingQueue.hh"
#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

namespace diffraflow {

    struct IngBufferItem {
        shared_ptr<vector<char>> rawdata;
        int index;
    };

    typedef BlockingQueue<IngBufferItem> IngBufferItemQueue;

} // namespace diffraflow

#endif