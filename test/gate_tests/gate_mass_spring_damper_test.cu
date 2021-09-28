#include <gtest/gtest.h>
#include <trajectory_generation/GATE.cuh>
#include <dynamics/mass_spring_damper/mass_spring_damper_dynamics.cuh>
#include <controllers/mass_spring_damper/mass_spring_damper_PID.cuh>
#include <controllers/mass_spring_damper/mass_spring_damper_LQR.cuh>
#include <perturbations/mass_spring_damper/mass_spring_damper_perturbations.cuh>
#include <dynamics/integrators_eigen.cuh>
#include <memory>
#include <chrono>
#include <iostream>


template <class T>
__global__ void testx0Kernel(T::all_states_at_t* x0_noise_device) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	printf("Idx: %i, Mean: %f\n", idx, (*x0_noise_device).row(idx).mean());
}

TEST(GATE_MSD, PropagateTrajectories) {
	// simulation timestep
	float dt = 0.01;

	// Setup rollouts
	const int num_timesteps = 5000; // 2500;
	const int num_rollouts = 500; // 500;
	const int bdim_x = 1;
	
	// set params
	std::shared_ptr<MsdDynamics> dyn = std::make_shared<MsdDynamics>();
	
	// PID Controller
	std::shared_ptr<MsdPID> ctrl = std::make_shared<MsdPID>();
	auto default_guid_params = ctrl->getGuidanceParams();
	default_guid_params.dt = dt;
	default_guid_params.p_gain =30;
	default_guid_params.i_gain = 0.3f;


	// LQR Controller
	//std::shared_ptr<MsdLQR> ctrl = std::make_shared<MsdLQR>();
	//auto default_guid_params = ctrl->getGuidanceParams();
	//default_guid_params.dt = dt;
	//default_guid_params.setNumTimesteps(num_timesteps);


	auto default_dyn_params = ctrl->getDynamicsParams();
	ctrl->setGuidanceParams(default_guid_params);
	ctrl->setDynamicsParams(default_dyn_params);
	dyn->setParams(default_dyn_params);
	//****Include if doing LQR controller****
	//ctrl->computeP();

	// perturbations
	MsdPertParams::state_array x0_mean, x0_std;
	MsdPertParams::control_array u_std;
	x0_mean << 1.f, 0.1f;
	x0_std << 1.f, 1.f;
	u_std << .1;
	MsdPertParams params = MsdPertParams(x0_mean, x0_std, u_std);
	std::shared_ptr<MsdPert<num_timesteps, num_rollouts>> pert = std::make_shared<MsdPert<num_timesteps, num_rollouts>>(params); 
	pert->initializeX0andControlPerturbations();
	pert->initPerturbations();

	// TODO: ultimately want to get rid of this
	std::cout << "x0_mean: " << x0_mean.transpose() << std::endl;
	std::cout << "x0_std: " << x0_std.transpose() << std::endl;

	// Create the GATE object
	auto RTE = new GATE<MsdDynamics, MsdPID, MsdPert<num_timesteps, num_rollouts>, 
		num_timesteps, num_rollouts, bdim_x>(dyn.get(), ctrl.get(), pert.get(), x0_mean, dt);

	RTE->computeTrajectories();
	
	//std::cout << "Trajectory 0: " << std::endl;
	//int idx = 0;
	//float tmp;
	//int sim_idx = 0;
	//for (int i = 0; i < 2*num_timesteps; ++i) {
	//	std::cout << sim_idx << "    ";
	//	for (int j = 0; j < MsdDynamics::STATE_DIM; ++j) {
	//		tmp = RTE->state_trajectories_host[idx];
	//		std::cout << tmp << "    ";
	//		idx++;
	//	};
	//	sim_idx++;
	//	std::cout << "" << std::endl;
	//	if (i == num_timesteps - 1) {
	//		std::cout << "" << std::endl;
	//	}
	//}

	// Assert that there are no NaN's
	for (int i = 0; i < num_rollouts; ++i) {
		for (int j = 0; j < MsdDynamics::STATE_DIM; ++j) {
			ASSERT_FALSE(isnan(
				RTE->state_trajectories_host[num_timesteps * MsdDynamics::STATE_DIM * i // Rollout
				+ (num_timesteps - 1) * MsdDynamics::STATE_DIM //Timestep
				+ j])) << "Rollout: " << i << ", State: " << j << std::endl; // State
		}
	}

	delete(RTE);
}

template<class PERT_T>
__global__ void testKernelStateDeriv(MsdDynamics* dyn, PERT_T* pert) {
	
	int timestep = 5;
	int rollout = 6;
	Eigen::Matrix<float, 2, 1> x;
	Eigen::Matrix<float, 2, 1> x_kp1;
	Eigen::Matrix<float, 1, 1> u;
	Eigen::Matrix<float, 1, 1> u_kp1;
	Eigen::Matrix<float, 2, 1> xdot;

	x << 0, 0;
	x_kp1 << 0, 0; 
	u << 0;
	xdot << 0, 0;

	dyn->computeStateDeriv(pert, timestep, rollout, x, u, xdot);

	float dt = 0.1; 

	integrators_eigen::rk4<MsdDynamics, PERT_T>(dyn, pert, timestep, rollout, x, u, dt, u_kp1, xdot, x_kp1);
}

TEST(RTE_GATE_Test, PerturbationsClass1) {
	// simulation timestep
	float dt = 0.01;

	// Setup rollouts

	const int num_timesteps = 2000;
	const int num_rollouts = 1;
	const int bdim_x = 1;

	// set params
	auto dyn = MsdDynamics();
	//std::shared_ptr<MsdDynamics> dyn = std::make_shared<MsdDynamics>();

	// PID Controller
	auto ctrl = MsdPID();
	//std::shared_ptr<MsdPID> ctrl = std::make_shared<MsdPID>();;
	auto default_guid_params = ctrl.getGuidanceParams();
	default_guid_params.dt = dt;
	default_guid_params.p_gain =30;
	default_guid_params.i_gain = 0.3f;

	auto default_dyn_params = ctrl.getDynamicsParams();
	ctrl.setGuidanceParams(default_guid_params);
	ctrl.setDynamicsParams(default_dyn_params);
	dyn.setParams(default_dyn_params);
	dyn.GPUSetup();

	MsdPertParams::state_array x0_mean, x0_std;
	MsdPertParams::control_array u_std;
	x0_mean << 1.3, 1.23;
	x0_std << 3, 1.4;
	u_std << 2.1;

	MsdPertParams params = MsdPertParams(x0_mean, x0_std, u_std);

	std::shared_ptr<MsdPert<num_timesteps, num_rollouts>> pert = std::make_shared<MsdPert<num_timesteps, num_rollouts>>(params);
	pert->initializeX0andControlPerturbations();
	pert->GPUSetup();
	pert->initPerturbations();

	testKernelStateDeriv << <1, 1 >> > ((&dyn)->model_d_, pert->perturbations_d_);
	CudaCheckError();
	// If the test makes it to this point without failing then we are good!

}