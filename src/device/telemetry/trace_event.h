#ifndef TELEMETRY_DEVICE_INCLUDE_TRACE_EVENT_H_
#define TELEMETRY_DEVICE_INCLUDE_TRACE_EVENT_H_

#include <cstdint>
#include "cuda_compat.h"

namespace telemetry {

template <typename T>
struct Span {
  uint64_t start_ts;
  uint64_t stop_ts;
  T metadata;
};

}  // namespace telemetry

#endif /* TELEMETRY_DEVICE_INCLUDE_TRACE_EVENT_H_ */
