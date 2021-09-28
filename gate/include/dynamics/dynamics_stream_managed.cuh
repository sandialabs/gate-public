#ifndef DYNAMICS_STREAM_MANAGED_CUH_
#define DYNAMICS_STREAM_MANAGED_CUH_


#include <stdio.h>
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif  //_MSC_VER
#include <cuda_util/stream_managed.cuh>
#include <math.h>
#include <vector>
#include <array>
#include <cfloat>

// Individual Eigen includes are required to build with CUDA inside CPP kernels
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include "Eigen/geometry_wrapper.h"

namespace GATE_internal {
  template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
  class Dynamics : public Managed
      // class Guidance : public Managed -> This is the base class for all guidance objects
  {
  public:
    static const int STATE_DIM = S_DIM;
    static const int CONTROL_DIM = C_DIM;

    typedef CLASS_T DYN_T;
    typedef PARAMS_T DYN_PARAMS_T;
    typedef Eigen::Matrix<float, CONTROL_DIM, 1> control_array; // Control at a time t
    typedef Eigen::Matrix<float, STATE_DIM, 1> state_array; // State at a time t

  protected:
    /**
     * sets the default control ranges to -infinity and +infinity
     */
    Dynamics(cudaStream_t stream = 0) : Managed(stream) {
      for (int i = 0; i < C_DIM; i++) {
        control_rngs_[i].x = -FLT_MAX;
        control_rngs_[i].y = FLT_MAX;
      }
    }

    /**
     * sets the control ranges to the passed in value
     * @param control_rngs
     * @param stream
     */
    Dynamics(std::array<float2, C_DIM> control_rngs, cudaStream_t stream = 0) : Managed(stream) {
      for (int i = 0; i < C_DIM; i++) {
        control_rngs_[i].x = control_rngs[i].x;
        control_rngs_[i].y = control_rngs[i].y;
      }
    }
  public:


    /**
     * Destructor must be virtual so that children are properly
     * destroyed when called from a Dynamics reference
     */
    virtual ~Dynamics() {
      freeCudaMem();
    }

    /**
     * Allocates all of the GPU memory
     */
    void GPUSetup();

    std::array<float2, C_DIM> getControlRanges() {
      std::array<float2, C_DIM> result;
      for (int i = 0; i < C_DIM; i++) {
        result[i] = control_rngs_[i];
      }
      return result;
    }
    __host__ __device__ float2* getControlRangesRaw() {
      return control_rngs_;
    }

    void setParams(const PARAMS_T& params) {
      params_ = params;
      if (GPUMemStatus_) {
        CLASS_T& derived = static_cast<CLASS_T&>(*this);
        derived.paramsToDevice();
      }
    }

    __device__ __host__ PARAMS_T getParams() { return params_; }


    /*
     *
     */
    void freeCudaMem();

    /**
     *
     */
    void printState(float* state);


    /**
     *
     */
    void paramsToDevice();


    __host__ __device__ void computeStateDeriv(state_array& x, control_array& u, state_array& xdot) {};


    // control ranges [.x, .y]
    float2 control_rngs_[C_DIM];

    // device pointer, null on the device
    CLASS_T* model_d_ = nullptr;

  protected:
    // generic parameter structure
    PARAMS_T params_;
  };

#ifdef __CUDACC__
#include "dynamics_stream_managed.cu"
#endif

  template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
  const int Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::STATE_DIM;

  template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
  const int Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::CONTROL_DIM;

} // GATE
#endif // DYNAMICS_STREAM_MANAGED_CUH_