#include <trajectory_export.h>
#include <trajectory_generation/GATE.cuh>
#include <dynamics/dubin_vehicle/dubin_dynamics.cuh>
#include <controllers/dubin_vehicle/dubin_PID.cuh>
#include <perturbations/dubin_vehicle/dubin_perturbations.cuh>
#include <dynamics/integrators_eigen.cuh>
#include <memory>
#include <iostream>

int main() {
	// NOTE: When saving trajectories we assume that the binary is located in
	// */gate-public/build/[BUILD-CONFIGURATION]/bin
	// [BUILD-CONFIGURATION] examples are Debug, Release, etc.
	// This assumption will be true if we use CMake to setup the build system for this project.
	std::string system_name = "dubins";

	// simulation timestep
	float dt = 0.1;

	// Setup rollouts
	const int num_timesteps = 500; // 2500;
	const int num_rollouts = 8192; // 500;
	const int bdim_x = 64;

	std::shared_ptr<DubinDynamics> dyn = std::make_shared<DubinDynamics>();
	std::shared_ptr<DubinPID> ctrl = std::make_shared<DubinPID>();
	// set params
	auto default_guid_params = ctrl->getGuidanceParams();
	auto default_dyn_params = ctrl->getDynamicsParams();

	default_guid_params.dt = dt;
	default_guid_params.p_gain = 0.85;
	default_guid_params.i_gain = 0.f;
	default_guid_params.pos_desired << 10.f, 10.f;

	ctrl->setGuidanceParams(default_guid_params);
	ctrl->setDynamicsParams(default_dyn_params);
	dyn->setParams(default_dyn_params);

	// perturbations
	DubinPertParams::state_array x0_mean, x0_std;
	DubinPertParams::control_array u_std;
	x0_mean << -5.f, 3.f, 0.f;
	x0_std << 1.f, 1.f, 1.f; // 0, 0, 0;
	u_std << .1;
	DubinPertParams params = DubinPertParams(x0_mean, x0_std, u_std);
	std::shared_ptr<DubinPert<num_timesteps, num_rollouts>> pert = std::make_shared<DubinPert<num_timesteps, num_rollouts>>(params);
	pert->initializeX0andControlPerturbations();
	pert->initPerturbations();

	// Create the GATE object
	typedef GATE<DubinDynamics, DubinPID, DubinPert<num_timesteps, num_rollouts>, num_timesteps, num_rollouts, bdim_x> Dubin_ROTE;
	std::shared_ptr<Dubin_ROTE> RTE = std::make_shared<Dubin_ROTE>(dyn.get(), ctrl.get(), pert.get(), x0_mean, dt);

	RTE->computeTrajectories();

	GATE_internal::save_traj_bundle(DubinDynamics::STATE_DIM, num_timesteps, num_rollouts, RTE->state_trajectories_host, system_name);

	return 0;
}