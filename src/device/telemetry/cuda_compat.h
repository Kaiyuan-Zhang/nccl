#ifndef TELEMETRY_DEVICE_INCLUDE_CUDA_COMPAT_H_
#define TELEMETRY_DEVICE_INCLUDE_CUDA_COMPAT_H_

// If not compiled with NVCC need to define __host__ and __device__

#if !__NVCC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

#endif /* TELEMETRY_DEVICE_INCLUDE_CUDA_COMPAT_H_ */
