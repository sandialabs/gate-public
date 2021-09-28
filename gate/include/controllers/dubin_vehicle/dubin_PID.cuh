#pragma once
#ifndef DUBIN_PID_CUH
#define DUBIN_PID_CUH
#include <dynamics/dubin_vehicle/dubin_dynamics.cuh>
#include <perturbations/dubin_vehicle/dubin_perturbations.cuh>
#include <cuda_util/stream_managed.cuh>
#include <controllers/guidance_stream_managed.cuh>
#include <dynamics/dynamics_stream_managed.cuh>
#include <dynamics/util.cuh>

namespace dubin_pid {
	const int S_DIM = 3;
	const int C_DIM = 1;
}

struct DubinPidParams {
	float p_gain = 5.0;
	float i_gain = 1.0;
	float d_gain = 0.f;
	float dt;
	Eigen::Matrix<float, 2, 1> pos_desired;
	// assuming 5000 is the maximum number of timesteps we will have
	float integral_error[8192] = { 0 };
	DubinPidParams() {
		pos_desired << 0.f, 0.f; }; 
	DubinPidParams(float input_p_gain, float input_i_gain, float input_d_gain, float input_dt, Eigen::Matrix<float, 2, 1>& input_pos_desired) { 
		p_gain = input_p_gain; i_gain = input_i_gain; d_gain = input_d_gain; dt = input_dt; pos_desired = input_pos_desired;
	}
	~DubinPidParams() = default;
};

class DubinPID : public GATE_internal::Guidance<DubinPID, DubinPidParams, DubinParams, dubin_pid::S_DIM, dubin_pid::C_DIM> {
public:
	DubinPID(cudaStream_t stream = nullptr) :
		GATE_internal::Guidance<DubinPID, DubinPidParams, DubinParams, dubin_pid::S_DIM, dubin_pid::C_DIM>(stream) {
		this->guidance_params_ = DubinPidParams();
		this->dynamics_params_ = DubinParams();
		}
	~DubinPID() = default;

	template<class PERT_T>
	__host__ __device__ void getControl(PERT_T* pert, int timestep, int rollout, int num_rollouts, state_array* x, state_array* xdot, control_array* u_k, control_array* u_kp1) {

		// get positional vector pointing from vehicle to target
		float error_x = guidance_params_.pos_desired(0) - (*x)(0);
		float error_y = guidance_params_.pos_desired(1) - (*x)(1);

		float error_distance = sqrtf(powf(error_x, 2.f) + powf(error_y, 2.f));
		float idealTheta;
		if (error_distance < 1e-5){
			idealTheta = 0.f;
		}
		else {
			// find orientation of this vector
			idealTheta = atan2(error_y, error_x);
		}

		// idealTheta between 0 and 360
		if (idealTheta < 0){
			idealTheta = idealTheta + deg2rad(360.f);
		}

		// calculate error theta
		float eTheta = idealTheta - (*x)(2);
		// wrap eTheta to be between -pi and pi
		if (eTheta < deg2rad(-180)) {
			eTheta = eTheta + deg2rad(360);
		}
		if (eTheta > deg2rad(180)) {
			eTheta = eTheta - deg2rad(360);
		}

		float dt = guidance_params_.dt;

		guidance_params_.integral_error[rollout] = guidance_params_.integral_error[rollout] + dt * eTheta;
		
		// Control commands
		(*u_k)(0) = guidance_params_.p_gain * (eTheta) + guidance_params_.i_gain * (guidance_params_.integral_error[rollout]);

		(*u_kp1)(0) = (*u_k)(0);

		// do not perturb first iteration
		if (rollout > 0) {

			control_array input_pert;
			input_pert.block<dubin_pid::C_DIM, 1>(0, 0) << (*(pert->u_t_pert_d_)).block<dubin_pid::C_DIM, 1>(0, num_rollouts * timestep + rollout);

			// add gaussian noise to input
			(*u_k)(0) = (*u_k)(0) + input_pert(0);

			(*u_kp1)(0) = (*u_k)(0);
		}

	} }; 
#endif // DUBIN_PID_CUH 