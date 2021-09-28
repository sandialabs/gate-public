#ifndef QUAD_DYNAMICS_CUH_
#define QUAD_DYNAMICS_CUH_

#include <dynamics/dynamics_stream_managed.cuh>
#include <perturbations/perturbations_stream_managed.cuh>
#include <perturbations/quadrotor/quad_perturbations.cuh>

using namespace Eigen;

namespace quad {
	const int S_DIM = 12;
	const int C_DIM = 6;
}

struct QuadParams {
	float m = 1.0;
	float g = 9.8;
	std::array<float2, 1> ctrl_limits;

	Matrix<float, 3, 3> I3;
	Matrix<float, 3, 3> invI3;
	QuadParams() {
		I3 = MatrixXf::Identity(3, 3);
		invI3 = MatrixXf::Identity(3, 3);
	};
	~QuadParams() = default;
};

class QuadDynamics : public GATE_internal::Dynamics<QuadDynamics, QuadParams, quad::S_DIM, quad::C_DIM> {
public:
	QuadDynamics(cudaStream_t stream = nullptr) :
		GATE_internal::Dynamics<QuadDynamics, QuadParams, quad::S_DIM, quad::C_DIM>(stream) {
			this->params_ = QuadParams();
		}
	~QuadDynamics() = default;

	__device__ __host__ void rotate_thrust(state_array& x, control_array& force_torque) {

		float roll = x(6);
		float pitch = x(7);
		float yaw = x(8);

		Matrix3f Rx;
		Rx << 1., 0., 0.,
			0., cos(roll), -sin(roll),
			0., sin(roll), cos(roll);

		Matrix3f Ry;
		Ry << cos(pitch), 0., sin(pitch),
			0., 1., 0.,
			-sin(pitch), 0., cos(pitch);

		Matrix3f Rz;
		Rz << cos(yaw), -sin(yaw), 0.,
			sin(yaw), cos(yaw), 0.,
			0., 0., 1.;

		force_torque.segment(0, 3) = Rz * Ry * Rx * force_torque.segment(0, 3);
	}

	__device__ __host__ void rot_w_to_inertial(float roll, float pitch, Matrix3f& R) {
		// rotate body angular rates to the inertial frame
		R << 1.0, sin(roll)* tan(pitch), cos(roll)* tan(pitch),
			0.0, cos(roll), -sin(roll),
			0.0, sin(roll) / cos(pitch), cos(roll) / cos(pitch);
	}

	template<class PER_T>
	__device__ __host__ void quad_dynamics(PER_T* pert, int rollout, state_array& x, control_array& u, state_array& x_dot) {

		float m = this->params_.m;
		//float g = this->params_.g;
		//Matrix<float, 3, 3> I3 = this->params_.I3;
		//Matrix<float, 3, 3> invI3 = this->params_.invI3;

		// todo replace m, g, I3 with variables?
		Matrix<float, 3, 3> I3; 
		I3 = MatrixXf::Identity(3, 3);
		Matrix<float, 3, 3> invI3;
		invI3 = MatrixXf::Identity(3, 3);

		// do not perturb first rollout
		if (rollout > 0) {
			// mass perturbation
			m = m + abs((*(pert->m_pert_d_))(rollout));

			//todo calculating every timestep.. I thought they changed every timestep
			int n = 0;
			I3.block<1, 1>(n, n) << abs(I3.coeff(n, n) + (*(pert->ixx_pert_d_))(rollout));
			invI3.block<1, 1>(n, n) << 1.0f / I3.coeff(n, n);
			
			n = 1;
			I3.block<1, 1>(n, n) << abs(I3.coeff(n, n) + (*(pert->iyy_pert_d_))(rollout));
			invI3.block<1, 1>(n, n) << 1.0f / I3.coeff(n, n);
			
			n = 2;
			I3.block<1, 1>(n, n) << abs(I3.coeff(n, n) + (*(pert->izz_pert_d_))(rollout));
			invI3.block<1, 1>(n, n) << 1.0f / I3.coeff(n, n);
		}



		// input
		Vector3f F = u.segment(0, 3);  // force in inertial frame
		F(2) -= m * 9.81;  // gravity assuming unit mass
		Vector3f tau = u.segment(3, 6);  // torque in body frame

		// substates
		//Vector3f x1 = x.segment(0, 2);  // position in inertial frame
		Vector3f x2 = x.segment(3, 6);  // velocity in inertial frame
		Vector3f x3 = x.segment(6, 9);  // roll pitch yaw in inertial frame
		Vector3f x4 = x.segment(9, 12);  // angular velocity in body frame

		// sub state derivatives
		Vector3f x1_dot = x2;  // d(pos) / dt = vel
		Vector3f x2_dot = F / m;  // F = ma->a = F / m
		// x3_dot
		Matrix3f R;
		rot_w_to_inertial(x3(0), x3(1), R);
		Vector3f x3_dot = R * x4;
		// x4_dotf
		Vector3f w_cross_Iw = x4.cross(I3 * x4);
		Vector3f x4_dot = invI3 * (tau - w_cross_Iw);

		//VectorXf x_dot(12);
		x_dot << x1_dot, x2_dot, x3_dot, x4_dot;

	}

	//__host__ __device__ void computeStateDeriv(state_array &x, control_array &u, state_array &xdot) {
	template<class PER_T>
	__host__ __device__ void computeStateDeriv(PER_T* pert, int timestep, int rollout, state_array& x_k, control_array& u, state_array& xdot) {

		rotate_thrust(x_k, u);
		quad_dynamics(pert, rollout, x_k, u, xdot);
	}
};
#endif // QUAD_DYNAMICS_CUH_