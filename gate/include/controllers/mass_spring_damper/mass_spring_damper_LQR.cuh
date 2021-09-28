#pragma once
#ifndef MSD_LQR_CUH
#define MSD_LQR_CUH
#include <dynamics/mass_spring_damper/mass_spring_damper_dynamics.cuh>
#include <perturbations/mass_spring_damper/mass_spring_damper_perturbations.cuh>
#include <cuda_util/stream_managed.cuh>
#include <controllers/guidance_stream_managed.cuh>
#include <dynamics/dynamics_stream_managed.cuh>
#include <dynamics/util.cuh>

namespace msd_lqr {
	const int S_DIM = 2;
	const int C_DIM = 1;
}

struct MsdLqrParams {
	
	// LQR weights
	Eigen::Matrix<float, 2, 2> Q;
	float R = 1;

	// assuming 5000 is the maximum number of timesteps we will have
	Eigen::Matrix<float, 2, 2> P[5000];

	float dt;
	float pos_desired;
	float integral_error = 0.f;
	float num_timesteps;
	MsdLqrParams() {
		pos_desired = 0.f;
		Q << 100, 0,
			0, 100;
		for (int i = 0; i < 5000; ++i) {
			P[i] << 0.f, 0.f,
					0.f, 0.f; 
		}
	}

	~MsdLqrParams() = default;

	void setNumTimesteps(int input_num_timesteps) {
		num_timesteps = input_num_timesteps;
		P[input_num_timesteps-1] << Q(0, 0), Q(0, 1),
									Q(1, 0), Q(1, 1);
	}

};

class MsdLQR : public GATE_internal::Guidance<MsdLQR, MsdLqrParams, MsdParams, msd_lqr::S_DIM, msd_lqr::C_DIM> {
public:
	MsdLQR(cudaStream_t stream = nullptr) :
		GATE_internal::Guidance<MsdLQR, MsdLqrParams, MsdParams, msd_lqr::S_DIM, msd_lqr::C_DIM>(stream) {
		this->guidance_params_ = MsdLqrParams();
		this->dynamics_params_ = MsdParams();
		}
	~MsdLQR() = default;

	__host__ void computeP(){
		Eigen::Matrix<float, 2, 2> P_kp1;
		Eigen::Matrix<float, 2, 2> A = this->dynamics_params_.A;
		Eigen::Matrix<float, 2, 1> B = this->dynamics_params_.B;
		Eigen::Matrix<float, 2, 2> negative_dp_dt;
		for (int i = this->guidance_params_.num_timesteps-2; i > -1; --i) {
			P_kp1 = this->guidance_params_.P[i + 1];
			negative_dp_dt = (A.transpose()) * P_kp1 + P_kp1 * A - (P_kp1 * B) * (1 / this->guidance_params_.R) * (B.transpose() * P_kp1) + this->guidance_params_.Q;
			this->guidance_params_.P[i] = P_kp1 + negative_dp_dt * this->guidance_params_.dt;
		}
		paramsToDevice();
	}

	template<class PER_T>
	__host__ __device__ void getControl(PER_T* pert, int timestep, int rollout, int num_rollouts, state_array* x, state_array* xdot, control_array* u_k, control_array* u_kp1) {
		
		Eigen::Matrix<float, 1, 2> K;
		Eigen::Matrix<float, 2, 2> P_k;

		P_k = this->guidance_params_.P[timestep];
		K = (1 / guidance_params_.R) * (this->dynamics_params_.B.transpose() * P_k);

		// Control commands
		(*u_k)(0) = -(K.transpose()).dot((*x));

		(*u_kp1)(0) = (*u_k)(0);

		// do not perturb first iteration
		if (rollout > 0) {

			control_array input_pert;
			input_pert.block<msd_lqr::C_DIM, 1>(0, 0) << (*(pert->u_t_pert_d_)).block<msd_lqr::C_DIM, 1>(0, num_rollouts * timestep + rollout);

			// add gaussian noise to input
			(*u_k)(0) = (*u_k)(0) + input_pert(0);

			(*u_kp1)(0) = (*u_k)(0);
		}
	
	}

}; 
#endif // MSD_LQR_CUH 