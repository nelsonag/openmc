#include "openmc/tallies/derivative.h"

#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/settings.h"
#include "openmc/tallies/tally.h"
#include "openmc/xml_interface.h"

#include <sstream>

template class std::vector<openmc::TallyDerivative>;

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace model {
  std::vector<TallyDerivative> tally_derivs;
  std::unordered_map<int, int> tally_deriv_map;
}

//==============================================================================
// TallyDerivative implementation
//==============================================================================

TallyDerivative::TallyDerivative(pugi::xml_node node)
{
  if (check_for_node(node, "id")) {
    id = std::stoi(get_node_value(node, "id"));
  } else {
    fatal_error("Must specify an ID for <derivative> elements in the tally "
                "XML file");
  }

  if (id <= 0)
    fatal_error("<derivative> IDs must be an integer greater than zero");

  std::string variable_str = get_node_value(node, "variable");

  if (variable_str == "density") {
    variable = DIFF_DENSITY;

  } else if (variable_str == "nuclide_density") {
    variable = DIFF_NUCLIDE_DENSITY;

    std::string nuclide_name = get_node_value(node, "nuclide");
    bool found = false;
    for (auto i = 0; i < data::nuclides.size(); ++i) {
      if (data::nuclides[i]->name_ == nuclide_name) {
        found = true;
        //TODO: off-by-one
        diff_nuclide = i + 1;
      }
    }
    if (!found) {
      std::stringstream out;
      out << "Could not find the nuclide \"" << nuclide_name
          << "\" specified in derivative " << id << " in any material.";
      fatal_error(out);
    }

  } else if (variable_str == "temperature") {
    variable = DIFF_TEMPERATURE;

  } else {
    std::stringstream out;
    out << "Unrecognized variable \"" << variable_str
        << "\" on derivative " << id;
    fatal_error(out);
  }

  diff_material = std::stoi(get_node_value(node, "material"));
}

//==============================================================================
// Non-method functions
//==============================================================================

extern "C" void
read_tally_derivatives(pugi::xml_node* node)
{
  // Populate the derivatives array.  This must be done in parallel because
  // the derivatives are threadprivate.
  #pragma omp parallel
  {
    for (auto deriv_node : node->children("derivative"))
      model::tally_derivs.emplace_back(deriv_node);
  }

  // Fill the derivative map.
  for (auto i = 0; i < model::tally_derivs.size(); ++i) {
    auto id = model::tally_derivs[i].id;
    auto search = model::tally_deriv_map.find(id);
    if (search == model::tally_deriv_map.end()) {
      model::tally_deriv_map[id] = i;
    } else {
      fatal_error("Two or more derivatives use the same unique ID: "
                  + std::to_string(id));
    }
  }

  // Make sure derivatives were not requested for an MG run.
  if (!settings::run_CE && !model::tally_derivs.empty())
    fatal_error("Differential tallies not supported in multi-group mode");
}

void
apply_derivative_to_score(const Particle* p, int i_tally, int i_nuclide,
  double atom_density, int score_bin, double& score)
{
  const Tally& tally {*model::tallies[i_tally]};

  if (score == 0.0) return;

  // If our score was previously c then the new score is
  // c * (1/f * d_f/d_p + 1/c * d_c/d_p)
  // where (1/f * d_f/d_p) is the (logarithmic) flux derivative and p is the
  // perturbated variable.

  const auto& deriv {model::tally_derivs[tally.deriv_]};
  auto flux_deriv = deriv.flux_deriv;

  // Handle special cases where we know that d_c/d_p must be zero.
  if (score_bin == SCORE_FLUX) {
    score *= flux_deriv;
    return;
  } else if (p->material == MATERIAL_VOID) {
    score *= flux_deriv;
    return;
  }
  //TODO: off-by-one
  const Material& material {*model::materials[p->material-1]};
  if (material.id_ != deriv.diff_material) {
    score *= flux_deriv;
    return;
  }

  switch (deriv.variable) {

  //============================================================================
  // Density derivative:
  // c = Sigma_MT
  // c = sigma_MT * N
  // c = sigma_MT * rho * const
  // d_c / d_rho = sigma_MT * const
  // (1 / c) * (d_c / d_rho) = 1 / rho

  case DIFF_DENSITY:
    switch (tally.estimator_) {

    case ESTIMATOR_ANALOG:
    case ESTIMATOR_COLLISION:
      switch (score_bin) {

      case SCORE_TOTAL:
      case SCORE_SCATTER:
      case SCORE_ABSORPTION:
      case SCORE_FISSION:
      case SCORE_NU_FISSION:
        score *= flux_deriv + 1. / material.density_gpcc_;
        break;

      default:
        fatal_error("Tally derivative not defined for a score on tally "
          + std::to_string(tally.id_));
      }
      break;

    default:
      fatal_error("Differential tallies are only implemented for analog and "
        "collision estimators.");
    }
    break;

  //============================================================================
  // Nuclide density derivative:
  // If we are scoring a reaction rate for a single nuclide then
  // c = Sigma_MT_i
  // c = sigma_MT_i * N_i
  // d_c / d_N_i = sigma_MT_i
  // (1 / c) * (d_c / d_N_i) = 1 / N_i
  // If the score is for the total material (i_nuclide = -1)
  // c = Sum_i(Sigma_MT_i)
  // d_c / d_N_i = sigma_MT_i
  // (1 / c) * (d_c / d_N) = sigma_MT_i / Sigma_MT
  // where i is the perturbed nuclide.

  case DIFF_NUCLIDE_DENSITY:
    //TODO: off-by-one throughout on diff_nuclide
    switch (tally.estimator_) {

    case ESTIMATOR_ANALOG:
      if (p->event_nuclide != deriv.diff_nuclide) {
        score *= flux_deriv;
        return;
      }

      switch (score_bin) {

      case SCORE_TOTAL:
      case SCORE_SCATTER:
      case SCORE_ABSORPTION:
      case SCORE_FISSION:
      case SCORE_NU_FISSION:
        {
          // Find the index of the perturbed nuclide.
          int i;
          for (i = 0; i < material.nuclide_.size(); ++i)
            if (material.nuclide_[i] == deriv.diff_nuclide - 1) break;
          score *= flux_deriv + 1. / material.atom_density_(i);
        }
        break;

      default:
        fatal_error("Tally derivative not defined for a score on tally "
          + std::to_string(tally.id_));
      }
      break;

    case ESTIMATOR_COLLISION:
      switch (score_bin) {

      case SCORE_TOTAL:
        if (i_nuclide == -1 && simulation::material_xs.total > 0.0) {
          score *= flux_deriv
            + simulation::micro_xs[deriv.diff_nuclide-1].total
            / simulation::material_xs.total;
        } else if (i_nuclide == deriv.diff_nuclide-1
                   && simulation::micro_xs[i_nuclide].total) {
          score *= flux_deriv + 1. / atom_density;
        } else {
          score *= flux_deriv;
        }
        break;

      case SCORE_SCATTER:
        if (i_nuclide == -1 && (simulation::material_xs.total
                                - simulation::material_xs.absorption) > 0.0) {
          score *= flux_deriv
            + (simulation::micro_xs[deriv.diff_nuclide-1].total
            - simulation::micro_xs[deriv.diff_nuclide-1].absorption)
            / (simulation::material_xs.total
            - simulation::material_xs.absorption);
        } else if (i_nuclide == deriv.diff_nuclide-1) {
          score *= flux_deriv + 1. / atom_density;
        } else {
          score *= flux_deriv;
        }
        break;

      case SCORE_ABSORPTION:
        if (i_nuclide == -1 && simulation::material_xs.absorption > 0.0) {
          score *= flux_deriv
            + simulation::micro_xs[deriv.diff_nuclide-1].absorption
            / simulation::material_xs.absorption;
        } else if (i_nuclide == deriv.diff_nuclide-1
                   && simulation::micro_xs[i_nuclide].absorption) {
          score *= flux_deriv + 1. / atom_density;
        } else {
          score *= flux_deriv;
        }
        break;

      case SCORE_FISSION:
        if (i_nuclide == -1 && simulation::material_xs.fission > 0.0) {
          score *= flux_deriv
            + simulation::micro_xs[deriv.diff_nuclide-1].fission
            / simulation::material_xs.fission;
        } else if (i_nuclide == deriv.diff_nuclide-1
                   && simulation::micro_xs[i_nuclide].fission) {
          score *= flux_deriv + 1. / atom_density;
        } else {
          score *= flux_deriv;
        }
        break;

      case SCORE_NU_FISSION:
        if (i_nuclide == -1 && simulation::material_xs.nu_fission > 0.0) {
          score *= flux_deriv
            + simulation::micro_xs[deriv.diff_nuclide-1].nu_fission
            / simulation::material_xs.nu_fission;
        } else if (i_nuclide == deriv.diff_nuclide-1
                   && simulation::micro_xs[i_nuclide].nu_fission) {
          score *= flux_deriv + 1. / atom_density;
        } else {
          score *= flux_deriv;
        }
        break;

      default:
        fatal_error("Tally derivative not defined for a score on tally "
          + std::to_string(tally.id_));
      }
      break;

    default:
      fatal_error("Differential tallies are only implemented for analog and "
        "collision estimators.");
    }
    break;

  //============================================================================
  // Temperature derivative:
  // If we are scoring a reaction rate for a single nuclide then
  // c = Sigma_MT_i
  // c = sigma_MT_i * N_i
  // d_c / d_T = (d_sigma_Mt_i / d_T) * N_i
  // (1 / c) * (d_c / d_T) = (d_sigma_MT_i / d_T) / sigma_MT_i
  // If the score is for the total material (i_nuclide = -1)
  // (1 / c) * (d_c / d_T) = Sum_i((d_sigma_MT_i / d_T) * N_i) / Sigma_MT_i
  // where i is the perturbed nuclide.  The d_sigma_MT_i / d_T term is
  // computed by multipole_deriv_eval.  It only works for the resolved
  // resonance range and requires multipole data.

  case DIFF_TEMPERATURE:
    switch (tally.estimator_) {

    case ESTIMATOR_ANALOG:
      {
        // Find the index of the event nuclide.
        int i;
        for (i = 0; i < material.nuclide_.size(); ++i)
          if (material.nuclide_[i] == p->event_nuclide-1) break;

        const auto& nuc {*data::nuclides[p->event_nuclide-1]};
        if (!multipole_in_range(&nuc, p->last_E)) {
          score *= flux_deriv;
          break;
        }

        switch (score_bin) {

        case SCORE_TOTAL:
          if (simulation::micro_xs[p->event_nuclide-1].total) {
            double dsig_s, dsig_a, dsig_f;
            std::tie(dsig_s, dsig_a, dsig_f)
              = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
            score *= flux_deriv + (dsig_s + dsig_a) * material.atom_density_(i)
              / simulation::material_xs.total;
          } else {
            score *= flux_deriv;
          }
          break;

        case SCORE_SCATTER:
          if (simulation::micro_xs[p->event_nuclide-1].total
              - simulation::micro_xs[p->event_nuclide-1].absorption) {
            double dsig_s, dsig_a, dsig_f;
            std::tie(dsig_s, dsig_a, dsig_f)
              = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
            score *= flux_deriv + dsig_s * material.atom_density_(i)
              / (simulation::material_xs.total
              - simulation::material_xs.absorption);
          } else {
            score *= flux_deriv;
          }
          break;

        case SCORE_ABSORPTION:
          if (simulation::micro_xs[p->event_nuclide-1].absorption) {
            double dsig_s, dsig_a, dsig_f;
            std::tie(dsig_s, dsig_a, dsig_f)
              = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
            score *= flux_deriv + dsig_a * material.atom_density_(i)
              / simulation::material_xs.absorption;
          } else {
            score *= flux_deriv;
          }
          break;

        case SCORE_FISSION:
          if (simulation::micro_xs[p->event_nuclide-1].fission) {
            double dsig_s, dsig_a, dsig_f;
            std::tie(dsig_s, dsig_a, dsig_f)
              = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
            score *= flux_deriv + dsig_f * material.atom_density_(i)
              / simulation::material_xs.fission;
          } else {
            score *= flux_deriv;
          }
          break;

        case SCORE_NU_FISSION:
          if (simulation::micro_xs[p->event_nuclide-1].fission) {
            double nu = simulation::micro_xs[p->event_nuclide-1].nu_fission
              / simulation::micro_xs[p->event_nuclide-1].fission;
            double dsig_s, dsig_a, dsig_f;
            std::tie(dsig_s, dsig_a, dsig_f)
              = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
            score *= flux_deriv + nu * dsig_f * material.atom_density_(i)
              / simulation::material_xs.nu_fission;
          } else {
            score *= flux_deriv;
          }
          break;

        default:
          fatal_error("Tally derivative not defined for a score on tally "
            + std::to_string(tally.id_));
        }
      }
      break;

    case ESTIMATOR_COLLISION:
      if (i_nuclide != -1) {
        const auto& nuc {data::nuclides[i_nuclide]};
        if (!multipole_in_range(nuc.get(), p->last_E)) {
          score *= flux_deriv;
          return;
        }
      }

      switch (score_bin) {

      case SCORE_TOTAL:
        if (i_nuclide == -1 && simulation::material_xs.total > 0.0) {
          double cum_dsig = 0;
          for (auto i = 0; i < material.nuclide_.size(); ++i) {
            auto i_nuc = material.nuclide_[i];
            const auto& nuc {*data::nuclides[i_nuc]};
            if (multipole_in_range(&nuc, p->last_E)
                && simulation::micro_xs[i_nuc].total) {
              double dsig_s, dsig_a, dsig_f;
              std::tie(dsig_s, dsig_a, dsig_f)
                = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
              cum_dsig += (dsig_s + dsig_a) * material.atom_density_(i);
            }
          }
          score *= flux_deriv + cum_dsig / simulation::material_xs.total;
        } else if (simulation::micro_xs[i_nuclide].total) {
          const auto& nuc {*data::nuclides[i_nuclide]};
          double dsig_s, dsig_a, dsig_f;
          std::tie(dsig_s, dsig_a, dsig_f)
            = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
          score *= flux_deriv
            + (dsig_s + dsig_a) / simulation::micro_xs[i_nuclide].total;
        } else {
          score *= flux_deriv;
        }
        break;

      case SCORE_SCATTER:
        if (i_nuclide == -1 && (simulation::material_xs.total
            - simulation::material_xs.absorption)) {
          double cum_dsig = 0;
          for (auto i = 0; i < material.nuclide_.size(); ++i) {
            auto i_nuc = material.nuclide_[i];
            const auto& nuc {*data::nuclides[i_nuc]};
            if (multipole_in_range(&nuc, p->last_E)
                && (simulation::micro_xs[i_nuc].total
                - simulation::micro_xs[i_nuc].absorption)) {
              double dsig_s, dsig_a, dsig_f;
              std::tie(dsig_s, dsig_a, dsig_f)
                = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
              cum_dsig += dsig_s * material.atom_density_(i);
            }
          }
          score *= flux_deriv + cum_dsig / (simulation::material_xs.total
            - simulation::material_xs.absorption);
        } else if (simulation::micro_xs[i_nuclide].total
                   - simulation::micro_xs[i_nuclide].absorption) {
          const auto& nuc {*data::nuclides[i_nuclide]};
          double dsig_s, dsig_a, dsig_f;
          std::tie(dsig_s, dsig_a, dsig_f)
            = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
          score *= flux_deriv + dsig_s / (simulation::micro_xs[i_nuclide].total
            - simulation::micro_xs[i_nuclide].absorption);
        } else {
          score *= flux_deriv;
        }
        break;

      case SCORE_ABSORPTION:
        if (i_nuclide == -1 && simulation::material_xs.absorption > 0.0) {
          double cum_dsig = 0;
          for (auto i = 0; i < material.nuclide_.size(); ++i) {
            auto i_nuc = material.nuclide_[i];
            const auto& nuc {*data::nuclides[i_nuc]};
            if (multipole_in_range(&nuc, p->last_E)
                && simulation::micro_xs[i_nuc].absorption) {
              double dsig_s, dsig_a, dsig_f;
              std::tie(dsig_s, dsig_a, dsig_f)
                = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
              cum_dsig += dsig_a * material.atom_density_(i);
            }
          }
          score *= flux_deriv + cum_dsig / simulation::material_xs.absorption;
        } else if (simulation::micro_xs[i_nuclide].absorption) {
          const auto& nuc {*data::nuclides[i_nuclide]};
          double dsig_s, dsig_a, dsig_f;
          std::tie(dsig_s, dsig_a, dsig_f)
            = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
          score *= flux_deriv
            + dsig_a / simulation::micro_xs[i_nuclide].absorption;
        } else {
          score *= flux_deriv;
        }
        break;

      case SCORE_FISSION:
        if (i_nuclide == -1 && simulation::material_xs.fission > 0.0) {
          double cum_dsig = 0;
          for (auto i = 0; i < material.nuclide_.size(); ++i) {
            auto i_nuc = material.nuclide_[i];
            const auto& nuc {*data::nuclides[i_nuc]};
            if (multipole_in_range(&nuc, p->last_E)
                && simulation::micro_xs[i_nuc].fission) {
              double dsig_s, dsig_a, dsig_f;
              std::tie(dsig_s, dsig_a, dsig_f)
                = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
              cum_dsig += dsig_f * material.atom_density_(i);
            }
          }
          score *= flux_deriv + cum_dsig / simulation::material_xs.fission;
        } else if (simulation::micro_xs[i_nuclide].fission) {
          const auto& nuc {*data::nuclides[i_nuclide]};
          double dsig_s, dsig_a, dsig_f;
          std::tie(dsig_s, dsig_a, dsig_f)
            = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
          score *= flux_deriv
            + dsig_f / simulation::micro_xs[i_nuclide].fission;
        } else {
          score *= flux_deriv;
        }
        break;

      case SCORE_NU_FISSION:
        if (i_nuclide == -1 && simulation::material_xs.nu_fission > 0.0) {
          double cum_dsig = 0;
          for (auto i = 0; i < material.nuclide_.size(); ++i) {
            auto i_nuc = material.nuclide_[i];
            const auto& nuc {*data::nuclides[i_nuc]};
            if (multipole_in_range(&nuc, p->last_E)
                && simulation::micro_xs[i_nuc].fission) {
              double nu = simulation::micro_xs[i_nuc].nu_fission
                / simulation::micro_xs[i_nuc].fission;
              double dsig_s, dsig_a, dsig_f;
              std::tie(dsig_s, dsig_a, dsig_f)
                = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
              cum_dsig += nu * dsig_f * material.atom_density_(i);
            }
          }
          score *= flux_deriv + cum_dsig / simulation::material_xs.nu_fission;
        } else if (simulation::micro_xs[i_nuclide].fission) {
          const auto& nuc {*data::nuclides[i_nuclide]};
          double dsig_s, dsig_a, dsig_f;
          std::tie(dsig_s, dsig_a, dsig_f)
            = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
          score *= flux_deriv
            + dsig_f / simulation::micro_xs[i_nuclide].fission;
        } else {
          score *= flux_deriv;
        }
        break;

      default:
        break;
      }
      break;

    default:
      fatal_error("Differential tallies are only implemented for analog and "
        "collision estimators.");
    }
    break;
  }
}

void
score_track_derivative(const Particle* p, double distance)
{
  // A void material cannot be perturbed so it will not affect flux derivatives.
  if (p->material == MATERIAL_VOID) return;
  //TODO: off-by-one
  const Material& material {*model::materials[p->material-1]};

  for (auto& deriv : model::tally_derivs) {
    if (deriv.diff_material != material.id_) continue;

    switch (deriv.variable) {

    case DIFF_DENSITY:
      // phi is proportional to e^(-Sigma_tot * dist)
      // (1 / phi) * (d_phi / d_rho) = - (d_Sigma_tot / d_rho) * dist
      // (1 / phi) * (d_phi / d_rho) = - Sigma_tot / rho * dist
      deriv.flux_deriv -= distance * simulation::material_xs.total
        / material.density_gpcc_;
      break;

    case DIFF_NUCLIDE_DENSITY:
      // phi is proportional to e^(-Sigma_tot * dist)
      // (1 / phi) * (d_phi / d_N) = - (d_Sigma_tot / d_N) * dist
      // (1 / phi) * (d_phi / d_N) = - sigma_tot * dist
      //TODO: off-by-one
      deriv.flux_deriv -= distance
        * simulation::micro_xs[deriv.diff_nuclide-1].total;
      break;

    case DIFF_TEMPERATURE:
      for (auto i = 0; i < material.nuclide_.size(); ++i) {
        const auto& nuc {*data::nuclides[material.nuclide_[i]]};
        if (multipole_in_range(&nuc, p->last_E)) {
          // phi is proportional to e^(-Sigma_tot * dist)
          // (1 / phi) * (d_phi / d_T) = - (d_Sigma_tot / d_T) * dist
          // (1 / phi) * (d_phi / d_T) = - N (d_sigma_tot / d_T) * dist
          double dsig_s, dsig_a, dsig_f;
          std::tie(dsig_s, dsig_a, dsig_f)
            = nuc.multipole_->evaluate_deriv(p->E, p->sqrtkT);
          deriv.flux_deriv -= distance * (dsig_s + dsig_a)
            * material.atom_density_(i);
        }
      }
      break;
    }
  }
}

void score_collision_derivative(const Particle* p)
{
  // A void material cannot be perturbed so it will not affect flux derivatives.
  if (p->material == MATERIAL_VOID) return;
  //TODO: off-by-one
  const Material& material {*model::materials[p->material-1]};

  for (auto& deriv : model::tally_derivs) {
    if (deriv.diff_material != material.id_) continue;

    switch (deriv.variable) {

    case DIFF_DENSITY:
      // phi is proportional to Sigma_s
      // (1 / phi) * (d_phi / d_rho) = (d_Sigma_s / d_rho) / Sigma_s
      // (1 / phi) * (d_phi / d_rho) = 1 / rho
      deriv.flux_deriv += 1. / material.density_gpcc_;
      break;

    case DIFF_NUCLIDE_DENSITY:
      //TODO: off-by-one throughout on diff_nuclide
      if (p->event_nuclide != deriv.diff_nuclide) continue;
      // Find the index in this material for the diff_nuclide.
      int i;
      for (i = 0; i < material.nuclide_.size(); ++i)
        if (material.nuclide_[i] == deriv.diff_nuclide - 1) break;
      // Make sure we found the nuclide.
      if (material.nuclide_[i] != deriv.diff_nuclide - 1) {
        std::stringstream err_msg;
        err_msg << "Could not find nuclide "
          << data::nuclides[deriv.diff_nuclide-1]->name_ << " in material "
          << material.id_ << " for tally derivative " << deriv.id;
        fatal_error(err_msg);
      }
      // phi is proportional to Sigma_s
      // (1 / phi) * (d_phi / d_N) = (d_Sigma_s / d_N) / Sigma_s
      // (1 / phi) * (d_phi / d_N) = sigma_s / Sigma_s
      // (1 / phi) * (d_phi / d_N) = 1 / N
      deriv.flux_deriv += 1. / material.atom_density_(i);
      break;

    case DIFF_TEMPERATURE:
      // Loop over the material's nuclides until we find the event nuclide.
      for (auto i_nuc : material.nuclide_) {
        const auto& nuc {*data::nuclides[i_nuc]};
        //TODO: off-by-one
        if (i_nuc == p->event_nuclide - 1
            && multipole_in_range(&nuc, p->last_E)) {
          // phi is proportional to Sigma_s
          // (1 / phi) * (d_phi / d_T) = (d_Sigma_s / d_T) / Sigma_s
          // (1 / phi) * (d_phi / d_T) = (d_sigma_s / d_T) / sigma_s
          const auto& micro_xs {simulation::micro_xs[i_nuc]};
          double dsig_s, dsig_a, dsig_f;
          std::tie(dsig_s, dsig_a, dsig_f)
            = nuc.multipole_->evaluate_deriv(p->last_E, p->sqrtkT);
          deriv.flux_deriv += dsig_s / (micro_xs.total - micro_xs.absorption);
          // Note that this is an approximation!  The real scattering cross
          // section is
          // Sigma_s(E'->E, uvw'->uvw) = Sigma_s(E') * P(E'->E, uvw'->uvw).
          // We are assuming that d_P(E'->E, uvw'->uvw) / d_T = 0 and only
          // computing d_S(E') / d_T.  Using this approximation in the vicinity
          // of low-energy resonances causes errors (~2-5% for PWR pincell
          // eigenvalue derivatives).
        }
      }
      break;
    }
  }
}

void zero_flux_derivs()
{
  for (auto& deriv : model::tally_derivs) deriv.flux_deriv = 0.;
}

//==============================================================================
// Fortran interop
//==============================================================================

extern "C" int n_tally_derivs() {return model::tally_derivs.size();}

extern "C" TallyDerivative*
tally_deriv_c(int i)
{
  return &model::tally_derivs[i];
}

}// namespace openmc