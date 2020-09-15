#ifndef __ImageFrameDgram_H__
#define __ImageFrameDgram_H__

// FRAME_SIZE = HEAD_SIZE + BODY_SIZE * BODY_COUNT
// HEAD_SIZE < DGRAM_MSIZE
// BODY_SIZE < DGRAM_MSIZE

#define FRAME_SIZE 131096
#define DGRAM_MSIZE 8210
#define HEAD_SIZE 8096
#define BODY_SIZE 8200
#define BODY_COUNT 15

#endif