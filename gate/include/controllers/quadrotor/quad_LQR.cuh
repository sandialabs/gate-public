#pragma once
#ifndef QUAD_LQR_CUH
#define QUAD_LQR_CUH
#include <dynamics/quadrotor/quad_dynamics.cuh>
#include <cuda_util/stream_managed.cuh>
#include <controllers/guidance_stream_managed.cuh>
#include <dynamics/dynamics_stream_managed.cuh>
#include <dynamics/util.cuh>

namespace quad_lqr {
	const int S_DIM = 12;
	const int C_DIM = 6;
}

struct QuadLQRParams {
	Eigen::Matrix<float, 4, 12> K;
	QuadLQRParams() {
		K << 0, 0, 0.1, 0, 0, 0.4583, 0, 0, 0, 0, 0, 0,
			0, -0.1000, 0, 0, -0.3160, 0, 4.4078, 0, 0, 3.2887, 0, 0,
			0.1, 0, 0, 0.316, 0, 0, 0, 4.4078, 0, 0, 3.2887, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 2.0;
	};
	~QuadLQRParams() = default;
};

class QuadLQR : public GATE_internal::Guidance<QuadLQR, QuadLQRParams, QuadParams, quad_lqr::S_DIM, quad_lqr::C_DIM> {
public:
	QuadLQR(cudaStream_t stream = nullptr) :
		GATE_internal::Guidance<QuadLQR, QuadLQRParams, QuadParams, quad_lqr::S_DIM, quad_lqr::C_DIM>(stream) {
		this->guidance_params_ = QuadLQRParams();
		this->dynamics_params_ = QuadParams();
	}
	~QuadLQR() = default;

	//__host__ __device__ void getControl(state_array* x, state_array* xdot, control_array* u_k, control_array* u_kp1) {
	template<class PER_T>
	__host__ __device__ void getControl(PER_T* pert, int timestep, int rollout, int num_rollouts, state_array* x, state_array* xdot, control_array* u_k, control_array* u_kp1) {

		(*u_k).segment(0, 2) << 0, 0;
		(*u_k).segment(2, 4) << -(this->guidance_params_.K) * (*x);
		(*u_k)(2) += 1 * 9.8 / (cos((*x)(6)) * cos((*x)(7)));  // nonlinear gravity offset;

		if (rollout > 0) {
			control_array input_pert;
			input_pert.block<quad_lqr::C_DIM, 1>(0, 0) << (*(pert->u_t_pert_d_)).block<quad_lqr::C_DIM, 1>(0, num_rollouts * timestep + rollout);
			(*u_k) += input_pert;
		}

		*u_kp1 = *u_k;
	}
};
#endif // MSD_PID_CUH 