#ifndef GUIDANCE_STREAM_MANAGED_CUH_
#define GUIDANCE_STREAM_MANAGED_CUH_

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

	template<class CLASS_T, class PARAMS_T, class DYN_PARAMS_T, int S_DIM, int C_DIM>
	class Guidance : public Managed {
	public:
		static const int STATE_DIM = S_DIM;
		static const int CONTROL_DIM = C_DIM;

		typedef CLASS_T GUID_T;
		typedef PARAMS_T GUID_PARAMS_T;
		typedef Eigen::Matrix<float, CONTROL_DIM, 1> control_array;
		typedef Eigen::Matrix<float, STATE_DIM, 1> state_array;

	protected:
		Guidance(cudaStream_t stream) : Managed(stream) {}

	public:
		virtual ~Guidance() {
			freeCudaMem();
		}

		void GPUSetup(); 

		void freeCudaMem();

		void paramsToDevice();

		// setter for parameters
		void setGuidanceParams(const PARAMS_T& params) {
			guidance_params_ = params;
			if (GPUMemStatus_) {
				CLASS_T& derived = static_cast<CLASS_T&>(*this);
				derived.paramsToDevice();
			}
		}

		void setDynamicsParams(const DYN_PARAMS_T& params) {
			dynamics_params_ = params;
			if (GPUMemStatus_) {
				CLASS_T& derived = static_cast<CLASS_T&>(*this);
				derived.paramsToDevice();
			}
		}

		// Getter for dynamics and guidance parameters
		__device__ __host__ PARAMS_T getGuidanceParams() { return guidance_params_; }
		__device__ __host__ DYN_PARAMS_T getDynamicsParams() { return dynamics_params_; }
		__device__ __host__ void getControl(state_array* x, state_array* xdot, control_array* u_k, control_array* u_kp1) {}

		PARAMS_T guidance_params_;
		DYN_PARAMS_T dynamics_params_;  //TODO this object should be shared between the dynamics class and perturbation class?

		// Guidance device pointer
		CLASS_T* guidance_d_ = nullptr;
	};

} // end namespace GATE

// stream is used for kernel calls. ex:
// kernel<<<10,10,0,stream> 


#ifdef __CUDACC__
#include "guidance_stream_managed.cu"
#endif // __CUDACC__

#endif // GUIDANCE_STREAM_MANAGED_CUH_