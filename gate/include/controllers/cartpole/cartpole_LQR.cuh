#pragma once
#ifndef CARTPOLE_LQR_CUH
#define CARTPOLE_LQR_CUH
#include <dynamics/cartpole/cartpole_dynamics.cuh>
#include <perturbations/cartpole/cartpole_perturbations.cuh>
#include <cuda_util/stream_managed.cuh>
#include <controllers/guidance_stream_managed.cuh>
#include <dynamics/dynamics_stream_managed.cuh>
#include <dynamics/util.cuh>


namespace cartpole_lqr {
	const int S_DIM = 4;
	const int C_DIM = 1;
}

struct CartpoleLQRParams {
	Eigen::Matrix<float, 1, 4> K;
	//lqrK << -1., -2.55, 41.67, 11.68;

	//Vector4f K(-1.f, -2.55f, 41.67f, 11.68f);
	//v << 1, 2, 3;

	CartpoleLQRParams() {};
	~CartpoleLQRParams() = default;
};

class CartpoleLQR : public GATE_internal::Guidance<CartpoleLQR, CartpoleLQRParams, CartpoleParams, cartpole_lqr::S_DIM, cartpole_lqr::C_DIM> {
public:
	CartpoleLQR(cudaStream_t stream = nullptr) :
		GATE_internal::Guidance<CartpoleLQR, CartpoleLQRParams, CartpoleParams, cartpole_lqr::S_DIM, cartpole_lqr::C_DIM>(stream) {
			this->guidance_params_ = CartpoleLQRParams();
			this->dynamics_params_ = CartpoleParams();
		}
	~CartpoleLQR() = default;

	template<class PERT_T>
	__host__ __device__ void getControl(PERT_T* pert, int timestep, int rollout, int num_rollouts, state_array* x, state_array* xdot, control_array* u_k, control_array* u_kp1){

		Eigen::Matrix<float, 1, 4> K; //todo make this a param
		K << -1., -2.55, 41.67, 11.68;

		control_array input_pert;
		input_pert.block<cartpole_lqr::C_DIM, 1>(0, 0) << (*(pert->u_t_pert_d_)).block<cartpole_lqr::C_DIM, 1>(0, num_rollouts * timestep + rollout);

		(*u_k)(0) = -K.dot(*x);
		if (rollout > 0) {
			(*u_k)(0) += input_pert(0);
		}
		(*u_kp1)(0) = (*u_k)(0);
	}
}; 
#endif // CARTPOLE_LQR_CUH 