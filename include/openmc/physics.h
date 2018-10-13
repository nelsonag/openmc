//! \file physics.h
//! Methods needed to perform the collision physics for continuous-energy mode

#ifndef OPENMC_PHYSICS_H
#define OPENMC_PHYSICS_H

#include "openmc/capi.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/reaction.h"

namespace openmc {

//! \brief ! handle all reactions with a single secondary neutron (other
//! than fission), i.e. level scattering, (n,np), (n,na), etc.
extern "C" void
inelastic_scatter(const double& awr, Reaction* rxn, Particle* p);

//! \brief Samples a target velocity based on the free gas scattering
//! formulation, used by most Monte Carlo codes, in which cross section is
//! assumed to be constant in energy. Excellent documentation for this method
//! can be found in FRA-TM-123.

extern "C" void
sample_cxs_target_velocity(const double& awr, const double& E,
     const double uvw[3], const double& kT, double v_target[3]);

} // namespace openmc

#endif // OPENMC_PHYSICS_H
