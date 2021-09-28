/*
 * Software License Agreement (BSD License)
 * Copyright (c) 2020, Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**********************************************
 * @file stream_managed.cuh
 * @author Manan Gandhi <mgandhi3@gatech.edu>
 * @date Aug 27, 2020
 * @copyright 2020 Georgia Institute of Technology
 * @brief Class to be inherited by classes passed
 * to CUDA kernels. Helps unify stream management.
 * Modified from managed.cuh used in MPPI-Generic.
 ***********************************************/

#ifndef STREAM_MANAGED_CUH_
#define STREAM_MANAGED_CUH_

#include <gpu_err_chk.h>

/**
 * @class Managed managed.cuh
 * @brief Class for setting the stream and managing GPU memory associated with
 * a class utilized in CUDA code.
 *
 * This class has one variable, which is the CUDA stream, and a function which
 * sets the stream. It is meant to be inherited by any class that will be used 
 * in a CUDA kernel function. In the past, this class
 * used unified memory (hence the managed name), so that classes could be passed
 * by reference to CUDA kernels. However, as of right now, the difficulties of
 * using unified memory with multi-threaded CPU programs make getting good
 * performance with unified memory difficult, so this has been removed. Future
 * implementations may bring back the unified memory feature.
 *
 * When GPU setup is called for this class, 
 */
class Managed {
 public:
  cudaStream_t stream_ = 0;  ///< The CUDA Stream that the class is bound too. 0
                             ///< is the default (NULL) stream.


  Managed(cudaStream_t stream = 0) { this->bindToStream(stream); }

  /**
  @brief Sets the stream and synchronizes the device.
  @param stream is the CUDA stream that the object is assigned too.
  */
  void bindToStream(cudaStream_t stream) {
    stream_ = stream;
    cudaDeviceSynchronize();
  }

  /**
  * @brief Get the current GPUMemstatus of the managed object
  *
  * We use a getter because only one function should be able to 
  * set GPUMemStatus_.
  */
  bool getGPUMemStatus() { return GPUMemStatus_; }
  bool GPUMemStatus_ = false;  // This variable is set to true when GPUSetup() is called.


  // REQUIRED: basic interface, make sure to implement in each class
  // GPUSetup - allocates the GPU object
  // freeCudaMem -> deallocates what is setup in GPUSetup
  // getParams, setParams -> gets and sets the parameters
  // paramsToDevice -> copies the parameters over to the GPU side

  // OPTIONAL:
  // printParams
  // other printing methods

 protected:
  template <class T>
  static T* GPUSetup(T* host_ptr) {
    // Allocate enough space on the GPU for the object
    T* device_ptr;
    cudaMalloc((void**)&device_ptr, sizeof(T));
    // Cudamemcpy
    HANDLE_ERROR(cudaMemcpyAsync(device_ptr, host_ptr, sizeof(T),
                                 cudaMemcpyHostToDevice, host_ptr->stream_));
    cudaDeviceSynchronize();
    host_ptr->GPUMemStatus_ = true;
    return device_ptr;
  }

  // TODO CRTP template this on the base class for allocation and dealloation
};

#endif /* STREAM_MANAGED_CUH_*/
