#include "openmc/physics.h"

#include "openmc/math_functions.h"
#include "openmc/random_lcg.h"

namespace openmc {

void
inelastic_scatter(const double& awr, Reaction* rxn, Particle* p)
{
  // Copy energy of neutron
  double E_in = p->E;

  // Sample outgoing energy and scattering cosine
  double E;
  double mu;
  reaction_product_sample(rxn, 1, E_in, &E, &mu);

  // If the scattering system is center-of-mass, transfer the cosine of the
  // scattering angle and outgoing energy from the CM to LAB frame
  if (rxn->scatter_in_cm_) {
    double E_cm = E;

    // determine the outgoing energy in LAB
    E = E_cm + (E_in + 2. * mu * (awr + 1.) * std::sqrt(E_in * E_cm)) /
         ((awr + 1.) * (awr + 1.));

    // determine outgoing angle in lab
    mu = mu * std::sqrt(E_cm / E) + std::sqrt(E_in / E) / (awr + 1.);
  }

  // Because of floating-point roundoff, it may be possible for mu to be
  // outside the range of [-1,1). If so, we just set mu to exactly -1 or 1
  if (std::abs(mu) > 1.) mu = std::copysign(1., mu);

  // Set outgoing energy and scattering angle
  p->E = E;
  p->mu = mu;

  // Change the direction of the particle
  rotate_angle_c(p->coord[0].uvw, mu, nullptr);

  // Evaluate the yield
  double yield = reaction_product_yield(rxn, 1, E_in);
  if (std::floor(yield) == yield) {
    // if the yield is an integer, create exactly that many secondary particles
    for (int i = 0; i < static_cast<int>(std::round(yield)) - 1; i++) {
      p->create_secondary(p->coord[0].uvw, p->E,
                          static_cast<int>(ParticleType::neutron), true);
    }
  } else {
    p->wgt = yield * p->wgt;
  }
}

//TODO: Replace uvw[3], v_target[3] with Direction
void
sample_cxs_target_velocity(const double& awr, const double& E,
     const double uvw[3], const double& kT, double v_target[3])
{
  double beta_vn = std::sqrt(awr * E / kT);
  double alpha = 1. / (1. + 0.5 * std::sqrt(PI) * beta_vn);

  double beta_vt_sq;
  double mu;
  double accept_prob;
  do {
    // Sample 2 random numbers
    double r1 = prn();
    double r2 = prn();

    if (prn() < alpha) {
      // With probability alpha, we sample the distribution p(y)=y*exp(-y).
      // This can be done with sampling scheme C45 from the Monte Carlo sampler
      beta_vt_sq = -std::log(r1 * r2);
    } else {
      // With probability 1-alpha, we sample the distributio p(y)=y^2*exp(-y).
      // This can be done with sampling scheme C61 from the Monte Carlo sampler
      double c = std::cos(0.5 * PI * prn());
      beta_vt_sq = -std::log(r1) - std::log(r2) * c * c;
    }

    // Determine beta * vt
    double beta_vt = std::sqrt(beta_vt_sq);

    // Sample the cosine of the angle between the particle and target velocity
    mu = 2. * prn() - 1.;

    // Determine the rejection probability
    accept_prob =
         std::sqrt(beta_vn * beta_vn + beta_vt_sq -
                   2. * beta_vn * beta_vt * mu) /  (beta_vn + beta_vt);

    // Perform rejection sampling on vt and mu
  } while(prn() >= accept_prob);

  // Determine the speed of the target nucleus
  double vt = std::sqrt(beta_vt_sq * kT / awr);

  // Determine the velocity vector of the target nucleus based on the neutron's
  // velocity and the sampled angle between them
  Direction uvw_t {uvw[0], uvw[1], uvw[2]};
  Direction v_target_temp = rotate_angle(uvw_t, mu, nullptr);
  v_target[0] = vt * v_target_temp.x;
  v_target[1] = vt * v_target_temp.y;
  v_target[2] = vt * v_target_temp.z;
}

} // namespace openmc