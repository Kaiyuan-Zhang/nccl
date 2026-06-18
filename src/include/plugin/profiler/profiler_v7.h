/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#ifndef PROFILER_V7_H_
#define PROFILER_V7_H_

#include "profiler_v6.h"

// V7 event structures and descriptors inherit directly from V6
typedef ncclProfilerEventDescr_v6_t ncclProfilerEventDescr_v7_t;
typedef ncclProfilerEventStateArgs_v6_t ncclProfilerEventStateArgs_v7_t;
typedef ncclProfilerEventState_v6_t ncclProfilerEventState_v7_t;

typedef struct {
  const char* name;

  // init - initialize the profiler plugin
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes, int nranks, int rank, ncclDebugLogger_t logfn);

  // startEvent - initialize and start a new event
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v7_t* eDescr);

  // stopEvent - stop/finalize an event
  ncclResult_t (*stopEvent)(void* eHandle);

  // recordEventState - record event state transitions and updates
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v7_t eState, ncclProfilerEventStateArgs_v7_t* eStateArgs);

  // finalize - finalize the profiler plugin
  ncclResult_t (*finalize)(void* context);

  // createDeviceFifo - create device FIFO that will be used for device to profiler
  // communication
  // Input
  //  - num_fifo: number of FIFOs to allocate
  // Output
  //  - fifo_descriptor_dev: descriptor of FIFO that will be passed to device
  //  - fifo_handle: handle of FIFO used when destroying the FIFO
  ncclResult_t (*createDeviceFifo)(int num_fifo, struct ncclDeviceFifo* fifo_descriptor_dev, void** fifo_handle);

  // destroyDeviceFifo - destroy device FIFO that will be used for device to profiler
  // communication
  // Input
  //  - num_fifo: number of FIFOs to destroy
  //  - fifo_handle: fifo handle
  ncclResult_t (*destroyDeviceFifo)(int num_fifo, void** fifo_handle);
} ncclProfiler_v7_t;

#endif // PROFILER_V7_H_
