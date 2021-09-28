#pragma once
#ifndef MSD_PID_CUH
#define MSD_PID_CUH
#include <dynamics/mass_spring_damper/mass_spring_damper_dynamics.cuh>
#include <cuda_util/stream_managed.cuh>
#include <controllers/guidance_stream_managed.cuh>
#include <dynamics/dynamics_stream_managed.cuh>
#include <dynamics/util.cuh>

namespace msd_pid {
	const int S_DIM = 2;
	const int C_DIM = 1;
}

struct MsdPidParams {
	float p_gain = 5.0;
	float i_gain = 1.0;
	float d_gain = 0.f;
	float dt;
	float pos_desired;
	// assuming 5000 is the maximum number of timesteps we will have
	float integral_error[5000] = { 0 };
	MsdPidParams() {
		pos_desired = 0.f;
	}
	MsdPidParams(float input_p_gain, float input_i_gain, float input_d_gain, float input_dt, float input_pos_desired) { 
	p_gain = input_p_gain; 
	i_gain = input_i_gain; 
	d_gain = input_d_gain; 
	dt = input_dt; 
	pos_desired = input_pos_desired;
	}
	~MsdPidParams() = default;
};

class MsdPID : public GATE_internal::Guidance<MsdPID, MsdPidParams, MsdParams, msd_pid::S_DIM, msd_pid::C_DIM> {
public:
	MsdPID(cudaStream_t stream = nullptr) :
		GATE_internal::Guidance<MsdPID, MsdPidParams, MsdParams, msd_pid::S_DIM, msd_pid::C_DIM>(stream) {
		this->guidance_params_ = MsdPidParams();
		this->dynamics_params_ = MsdParams();
		}
	~MsdPID() = default;

	__host__ void computeP() {

	}

	template<class PER_T>
	__host__ __device__ void getControl(PER_T* pert, int timestep, int rollout, int num_rollouts, state_array* x, state_array* xdot, control_array* u_k, control_array* u_kp1){

		float errorPos = guidance_params_.pos_desired - (*x)(0);

		float dt = guidance_params_.dt;

		// memory of integral error is indexed by rollout #
		guidance_params_.integral_error[rollout] = guidance_params_.integral_error[rollout] + dt * errorPos;
		
		// Control commands
		(*u_k)(0) = guidance_params_.p_gain * (errorPos)+guidance_params_.i_gain * (guidance_params_.integral_error[rollout]);

		(*u_kp1)(0) = (*u_k)(0);

		// do not perturb first iteration
		if (rollout > 0) {

			control_array input_pert;
			input_pert.block<msd_pid::C_DIM, 1>(0, 0) << (*(pert->u_t_pert_d_)).block<msd_pid::C_DIM, 1>(0, num_rollouts * timestep + rollout);

			// add gaussian noise to input
			(*u_k)(0) = (*u_k)(0) + input_pert(0);

			(*u_kp1)(0) = (*u_k)(0);
		}
	} }; 
#endif // MSD_PID_CUH 