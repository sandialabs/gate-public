#include <trajectory_export.h>
#include <trajectory_generation/GATE.cuh>
#include <dynamics/quadrotor/quad_dynamics.cuh>
#include <controllers/quadrotor/quad_LQR.cuh>
#include <perturbations/quadrotor/quad_perturbations.cuh>
#include <dynamics/integrators_eigen.cuh>
#include <memory>
#include <iostream>

#include <dynamics/dynamics_stream_managed.cuh>
#include <perturbations/perturbations_stream_managed.cuh>
#include <perturbations/quadrotor/quad_perturbations.cuh>


int main() {

	// NOTE: When saving trajectories we assume that the binary is located in
	// */gate-public/build/[BUILD-CONFIGURATION]/bin
	// [BUILD-CONFIGURATION] examples are Debug, Release, etc.
	// This assumption will be true if we use CMake to setup the build system for this project.
	std::string system_name = "quad";

	// simulation timestep
	float dt = 0.01;

	// Setup rollouts
	const int num_timesteps = 1500;
	const int num_rollouts =  1024;
	const int bdim_x = 64;

	// set params
	std::shared_ptr<QuadDynamics> dyn = std::make_shared<QuadDynamics>();

	// LQR Controller
	std::shared_ptr<QuadLQR> ctrl = std::make_shared<QuadLQR>();
	auto default_guid_params = ctrl->getGuidanceParams();
	//default_guid_params.dt = dt; //todo is this necessary?
	//default_guid_params.setNumTimesteps(num_timesteps);
	
	
	auto default_dyn_params = ctrl->getDynamicsParams();
	ctrl->setGuidanceParams(default_guid_params);
	ctrl->setDynamicsParams(default_dyn_params);
	dyn->setParams(default_dyn_params);

	
	// perturbations
	QuadPertParams::state_array x0_mean, x0_std;
	QuadPertParams::control_array u_std;
	x0_mean << -2.f, -2.f, -2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f;
	x0_std << 0.1f, 0.1f, 0.1f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f;
	//u_std << .0f, .0f, .0f, .0f;
	u_std << 0, 0, .25f, .25f, .25f, .25f; // First two inputs are force in x and y
	QuadPertParams params = QuadPertParams(x0_mean, x0_std, u_std);
	std::shared_ptr<QuadPert<num_timesteps, num_rollouts>> pert = std::make_shared<QuadPert<num_timesteps, num_rollouts>>(params);
	pert->initializeX0andControlPerturbations();
	pert->initPerturbations();

	
	// Create the GATE object
	typedef GATE<QuadDynamics, QuadLQR, QuadPert<num_timesteps, num_rollouts>, num_timesteps, num_rollouts, bdim_x> Quad_ROTE;
	std::shared_ptr<Quad_ROTE> RTE = std::make_shared<Quad_ROTE>(dyn.get(), ctrl.get(), pert.get(), x0_mean, dt);

	RTE->computeTrajectories();
	
	GATE_internal::save_traj_bundle(QuadDynamics::STATE_DIM, num_timesteps, num_rollouts, RTE->state_trajectories_host, system_name);
	
	return 0;
}