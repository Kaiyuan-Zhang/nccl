/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "checks.h"
#include "os.h"

static ncclProfiler_t ncclProfiler;
static ncclProfiler_v6_t* ncclProfiler_v6;

static ncclResult_t ncclProfiler_startEvent(void* ctx, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  return ncclProfiler_v6->startEvent(ctx, eHandle, (ncclProfilerEventDescr_v6_t *)eDescr);
}

static ncclResult_t ncclProfiler_recordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) {
  return ncclProfiler_v6->recordEventState(eHandle, (ncclProfilerEventState_v6_t)eState, (ncclProfilerEventStateArgs_v6_t *)eStateArgs);
}

static ncclResult_t ncclProfiler_init(void** ctx, uint64_t commId, int* eActivationMask, const char* commName, int nNodes, int nRanks, int rank, ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclProfiler_v6->init(ctx, commId, eActivationMask, commName, nNodes, nRanks, rank, logfn));
  ncclProfiler.startEvent = ncclProfiler_startEvent;
  ncclProfiler.recordEventState = ncclProfiler_recordEventState;
  ncclProfiler.stopEvent = ncclProfiler_v6->stopEvent;
  ncclProfiler.finalize = ncclProfiler_v6->finalize;
  ncclProfiler.createDeviceFifo = nullptr;
  ncclProfiler.destroyDeviceFifo = nullptr;
  return ncclSuccess;
}

ncclProfiler_t* getNcclProfiler_v6(void* lib) {
  ncclProfiler_v6 = (ncclProfiler_v6_t*)ncclOsDlsym(lib, "ncclProfiler_v6");
  if (ncclProfiler_v6) {
    ncclProfiler.name = ncclProfiler_v6->name;
    ncclProfiler.init = ncclProfiler_init;
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v6)", ncclProfiler_v6->name);
    return &ncclProfiler;
  }
  return NULL;
}
