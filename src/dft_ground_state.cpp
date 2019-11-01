// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file dft_ground_state.cpp
 *
 *  \brief Contains implementation of sirius::DFT_ground_state class.
 */

#include "dft_ground_state.hpp"

namespace sirius {

double DFT_ground_state::ewald_energy() const
{
    PROFILE("sirius::DFT_ground_state::ewald_energy");

    double alpha = ctx_.ewald_lambda();

    double ewald_g{0};

    int ig0 = ctx_.gvec().skip_g0();

    #pragma omp parallel for schedule(static) reduction(+:ewald_g)
    for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
        int ig = ctx_.gvec().offset() + igloc;

        double g2 = std::pow(ctx_.gvec().gvec_len(ig), 2);

        double_complex rho(0, 0);

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            rho += ctx_.gvec_phase_factor(ig, ia) * static_cast<double>(unit_cell_.atom(ia).zn());
        }

        ewald_g += std::pow(std::abs(rho), 2) * std::exp(-g2 / 4 / alpha) / g2;
    }

    ctx_.comm().allreduce(&ewald_g, 1);
    if (ctx_.gvec().reduced()) {
        ewald_g *= 2;
    }
    /* remaining G=0 contribution */
    ewald_g -= std::pow(unit_cell_.num_electrons(), 2) / alpha / 4;
    ewald_g *= (twopi / unit_cell_.omega());

    /* remove self-interaction */
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        ewald_g -= std::sqrt(alpha / pi) * std::pow(unit_cell_.atom(ia).zn(), 2);
    }

    double ewald_r{0};
    #pragma omp parallel for reduction(+:ewald_r)
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        for (int i = 1; i < unit_cell_.num_nearest_neighbours(ia); i++) {
            int ja   = unit_cell_.nearest_neighbour(i, ia).atom_id;
            double d = unit_cell_.nearest_neighbour(i, ia).distance;
            ewald_r += 0.5 * unit_cell_.atom(ia).zn() * unit_cell_.atom(ja).zn() *
                       std::erfc(std::sqrt(alpha) * d) / d;
        }
    }

    return (ewald_g + ewald_r);
}

void DFT_ground_state::initial_state()
{
    density_.initial_density();
    potential_.generate(density_);
    if (!ctx_.full_potential()) {
        Hamiltonian0 H0(potential_);
        Band(ctx_).initialize_subspace(kset_, H0);
    }
}

void DFT_ground_state::update()
{
    PROFILE("sirius::DFT_ground_state::update");

    ctx_.update();
    kset_.update();
    potential_.update();
    density_.update();

    if (!ctx_.full_potential()) {
        ewald_energy_ = ewald_energy();
    }
}

/// Return nucleus energy in the electrostatic field.
/** Compute energy of nucleus in the electrostatic potential generated by the total (electrons + nuclei)
 *  charge density. Diverging self-interaction term z*z/|r=0| is excluded. */
double DFT_ground_state::energy_enuc() const
{
    double enuc{0};
    if (ctx_.full_potential()) {
        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            int zn = unit_cell_.atom(ia).zn();
            enuc -= 0.5 * zn * potential_.vh_el(ia);
        }
        ctx_.comm().allreduce(&enuc, 1);
    }
    return enuc;
}

/// Return eigen-value sum of core states.
double DFT_ground_state::core_eval_sum() const
{
    double sum{0};
    for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
        sum += unit_cell_.atom_symmetry_class(ic).core_eval_sum() * unit_cell_.atom_symmetry_class(ic).num_atoms();
    }
    return sum;
}

double DFT_ground_state::energy_kin_sum_pw() const
{
    double ekin{0};

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        int ik = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_[ik];

        #pragma omp parallel for schedule(static) reduction(+:ekin)
        for (int igloc = 0; igloc < kp->num_gkvec_loc(); igloc++) {
            auto Gk = kp->gkvec().gkvec_cart<index_domain_t::local>(igloc);

            double d{0};
            for (int ispin = 0; ispin < ctx_.num_spins(); ispin++) {
                for (int i = 0; i < kp->num_occupied_bands(ispin); i++) {
                    double f = kp->band_occupancy(i, ispin);
                    auto z = kp->spinor_wave_functions().pw_coeffs(ispin).prime(igloc, i);
                    d += f * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                }
            }
            if (kp->gkvec().reduced()) {
                d *= 2;
            }
            ekin += 0.5 * d * kp->weight() * Gk.length2();
        } // igloc
    } // ikloc
    ctx_.comm().allreduce(&ekin, 1);
    return ekin;
}

double DFT_ground_state::total_energy() const
{
    double tot_en{0};

    switch (ctx_.electronic_structure_method()) {
        case electronic_structure_method_t::full_potential_lapwlo: {
            tot_en = (energy_kin() + energy_exc() + 0.5 * energy_vha() + energy_enuc());
            break;
        }

        case electronic_structure_method_t::pseudopotential: {
            tot_en = (kset_.valence_eval_sum() - energy_vxc() - energy_bxc() - potential_.PAW_one_elec_energy()) -
                     0.5 * energy_vha() + energy_exc() + potential_.PAW_total_energy() + ewald_energy_;
            break;
        }
    }

    if (ctx_.hubbard_correction()) {
        tot_en += potential_.U().hubbard_energy();
    }

    return tot_en;
}

json DFT_ground_state::serialize()
{
    json dict;

    dict["mpi_grid"] = ctx_.mpi_grid_dims();

    std::vector<int> fftgrid = {ctx_.spfft().dim_x(),ctx_.spfft().dim_y(), ctx_.spfft().dim_z()};
    dict["fft_grid"] = fftgrid;
    fftgrid = {ctx_.spfft_coarse().dim_x(),ctx_.spfft_coarse().dim_y(), ctx_.spfft_coarse().dim_z()};
    dict["fft_coarse_grid"]         = fftgrid;
    dict["num_fv_states"]           = ctx_.num_fv_states();
    dict["num_bands"]               = ctx_.num_bands();
    dict["aw_cutoff"]               = ctx_.aw_cutoff();
    dict["pw_cutoff"]               = ctx_.pw_cutoff();
    dict["omega"]                   = ctx_.unit_cell().omega();
    dict["chemical_formula"]        = ctx_.unit_cell().chemical_formula();
    dict["num_atoms"]               = ctx_.unit_cell().num_atoms();
    dict["energy"]                  = json::object();
    dict["energy"]["total"]         = total_energy();
    dict["energy"]["enuc"]          = energy_enuc();
    dict["energy"]["core_eval_sum"] = core_eval_sum();
    dict["energy"]["vha"]           = energy_vha();
    dict["energy"]["vxc"]           = energy_vxc();
    dict["energy"]["exc"]           = energy_exc();
    dict["energy"]["bxc"]           = energy_bxc();
    dict["energy"]["veff"]          = energy_veff();
    dict["energy"]["eval_sum"]      = eval_sum();
    dict["energy"]["kin"]           = energy_kin();
    dict["energy"]["ewald"]         = energy_ewald();
    if (!ctx_.full_potential()) {
        dict["energy"]["vloc"]      = energy_vloc();
    }
    dict["efermi"]                  = kset_.energy_fermi();
    dict["band_gap"]                = kset_.band_gap();
    dict["core_leakage"]            = density_.core_leakage();

    return dict;
}

/// A quick check of self-constent density in case of pseudopotential.
json DFT_ground_state::check_scf_density()
{
    if (ctx_.full_potential()) {
        return json();
    }
    std::vector<double_complex> rho_pw(ctx_.gvec().count());
    for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
        rho_pw[ig] = density_.rho().f_pw_local(ig);
    }

    double etot = total_energy();

    /* create new potential */
    Potential pot(ctx_);
    /* generate potential from existing density */
    pot.generate(density_);
    /* create new Hamiltonian */
    Hamiltonian0 H0(pot);
    /* set the high tolerance */
    ctx_.iterative_solver_tolerance(ctx_.settings().itsol_tol_min_);
    /* initialize the subspace */
    Band(ctx_).initialize_subspace(kset_, H0);
    /* find new wave-functions */
    Band(ctx_).solve(kset_, H0, true);
    /* find band occupancies */
    kset_.find_band_occupancies();
    /* generate new density from the occupied wave-functions */
    density_.generate(kset_, true, false);
    /* symmetrize density and magnetization */
    if (ctx_.use_symmetry()) {
        density_.symmetrize();
        if (ctx_.electronic_structure_method() == electronic_structure_method_t::pseudopotential) {
            density_.symmetrize_density_matrix();
        }
    }
    density_.fft_transform(1);
    double rms{0};
    for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
        rms += std::pow(std::abs(density_.rho().f_pw_local(ig) - rho_pw[ig]), 2);
    }
    ctx_.comm().allreduce(&rms, 1);
    json dict;
    dict["rss"]   = rms;
    dict["rms"]   = std::sqrt(rms / ctx_.gvec().num_gvec());
    dict["detot"] = total_energy() - etot;

    if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 1) {
        printf("[sirius::DFT_ground_state::check_scf_density] RSS: %18.12E\n", dict["rss"].get<double>());
        printf("[sirius::DFT_ground_state::check_scf_density] RMS: %18.12E\n", dict["rms"].get<double>());
        printf("[sirius::DFT_ground_state::check_scf_density] dEtot: %18.12E\n", dict["detot"].get<double>());
        printf("[sirius::DFT_ground_state::check_scf_density] Eold: %18.12E  Enew: %18.12E\n", etot, total_energy());
    }

    return dict;
}

json DFT_ground_state::find(double rms_tol, double energy_tol, double initial_tolerance, int num_dft_iter, bool write_state)
{
    PROFILE("sirius::DFT_ground_state::scf_loop");

    auto tstart = std::chrono::high_resolution_clock::now();

    double eold{0}, rms{0};

    density_.mixer_init(ctx_.mixer_input());

    int num_iter{-1};
    std::vector<double> rms_hist;
    std::vector<double> etot_hist;

    if (ctx_.hubbard_correction()) { // TODO: move to inititialization functions
        potential_.U().hubbard_compute_occupation_numbers(kset_);
        potential_.U().calculate_hubbard_potential_and_energy();
    }

    ctx_.iterative_solver_tolerance(initial_tolerance);

    for (int iter = 0; iter < num_dft_iter; iter++) {
        utils::timer t1("sirius::DFT_ground_state::scf_loop|iteration");

        if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 1) {
            printf("\n");
            printf("+------------------------------+\n");
            printf("| SCF iteration %3i out of %3i |\n", iter, num_dft_iter);
            printf("+------------------------------+\n");
        }
        Hamiltonian0 H0(potential_);
        /* find new wave-functions */
        Band(ctx_).solve(kset_, H0, true);
        /* find band occupancies */
        kset_.find_band_occupancies();
        /* generate new density from the occupied wave-functions */
        density_.generate(kset_, ctx_.use_symmetry(), true, true);

        /* mix density */
        rms = density_.mix();

        /* transform mixed density to plane-wave domain */
        density_.fft_transform(-1);

        double old_tol = ctx_.iterative_solver_tolerance();
        /* estimate new tolerance of iterative solver */
        double tol = std::min(ctx_.settings().itsol_tol_scale_[0] * rms, ctx_.settings().itsol_tol_scale_[1] * old_tol);
        tol = std::max(ctx_.settings().itsol_tol_min_, tol);
        /* set new tolerance of iterative solver */
        ctx_.iterative_solver_tolerance(tol);

        /* check number of elctrons */
        density_.check_num_electrons();

        /* compute new potential */
        potential_.generate(density_);

        if (!ctx_.full_potential() && ctx_.control().verification_ >= 2) {
            ctx_.message(1, __func__, "checking functional derivative of Exc\n");
            double eps{0.1};
            for (int i = 0; i < 10; i++) {
                Potential p1(ctx_);
                p1.scale_rho_xc(1 + eps);
                p1.generate(density_);

                double evxc = potential_.energy_vxc(density_) + potential_.energy_vxc_core(density_) + energy_bxc();
                double deriv = (p1.energy_exc(density_) - potential_.energy_exc(density_)) / eps;

                printf("eps              : %18.12f\n", eps);
                printf("Energy Vxc       : %18.12f\n", evxc);
                printf("numerical deriv  : %18.12f\n", deriv);
                printf("difference       : %18.12f\n", std::abs(evxc - deriv));
                eps /= 10;
            }
        }

        /* symmetrize potential and effective magnetic field */
        if (ctx_.use_symmetry()) {
            potential_.symmetrize();
        }

        /* transform potential to real space after symmetrization */
        potential_.fft_transform(1);

        /* compute new total energy for a new density */
        double etot = total_energy();

        etot_hist.push_back(etot);

        rms_hist.push_back(rms);

        /* write some information */
        print_info();
        if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 1) {
            printf("iteration : %3i, RMS %18.12E, energy difference : %18.12E\n", iter, rms, etot - eold);
        }
        /* check if the calculation has converged */
        if (std::abs(eold - etot) < energy_tol && rms < rms_tol) {
            if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 1) {
                printf("\n");
                printf("converged after %i SCF iterations!\n", iter + 1);
            }
            num_iter = iter;
            break;
        }

        /* Compute the hubbard correction */
        if (ctx_.hubbard_correction()) {
            potential_.U().hubbard_compute_occupation_numbers(kset_);
            potential_.U().calculate_hubbard_potential_and_energy();
        }

        eold = etot;
    }

    if (write_state) {
        ctx_.create_storage_file();
        if (ctx_.full_potential()) { // TODO: why this is necessary?
            density_.rho().fft_transform(-1);
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                density_.magnetization(j).fft_transform(-1);
            }
        }
        potential_.save();
        density_.save();
        //kset_.save(storage_file_name);
    }

    auto tstop = std::chrono::high_resolution_clock::now();

    json dict = serialize();
    dict["scf_time"] = std::chrono::duration_cast<std::chrono::duration<double>>(tstop - tstart).count();
    dict["etot_history"] = etot_hist;
    if (num_iter >= 0) {
        dict["converged"]          = true;
        dict["num_scf_iterations"] = num_iter;
        dict["rms_history"]        = rms_hist;
    } else {
        dict["converged"] = false;
    }

    //if (ctx_.control().verification_ >= 1) {
    //    check_scf_density();
    //}

    // dict["volume"] = ctx.unit_cell().omega() * std::pow(bohr_radius, 3);
    // dict["volume_units"] = "angstrom^3";
    // dict["energy"] = dft.total_energy() * ha2ev;
    // dict["energy_units"] = "eV";

    return dict;
}

void DFT_ground_state::print_info()
{
    double evalsum1 = kset_.valence_eval_sum();
    double evalsum2 = core_eval_sum();
    double ekin     = energy_kin();
    double evxc     = energy_vxc();
    double eexc     = energy_exc();
    double ebxc     = energy_bxc();
    double evha     = energy_vha();
    double etot     = total_energy();
    double gap      = kset_.band_gap() * ha2ev;
    double ef       = kset_.energy_fermi();
    double enuc     = energy_enuc();

    double one_elec_en = evalsum1 - (evxc + evha);

    if (ctx_.electronic_structure_method() == electronic_structure_method_t::pseudopotential) {
        one_elec_en -= potential_.PAW_one_elec_energy();
    }

    auto result = density_.rho().integrate();

    auto total_charge = std::get<0>(result);
    auto it_charge    = std::get<1>(result);
    auto mt_charge    = std::get<2>(result);

    auto result_mag = density_.get_magnetisation();
    auto total_mag  = std::get<0>(result_mag);
    auto it_mag     = std::get<1>(result_mag);
    auto mt_mag     = std::get<2>(result_mag);

    //double total_mag[3];
    //std::vector<double> mt_mag[3];
    //double it_mag[3];
    //for (int j = 0; j < ctx_.num_mag_dims(); j++) {
    //    auto result = density_.magnetization(j).integrate();

    //    total_mag[j] = std::get<0>(result);
    //    it_mag[j]    = std::get<1>(result);
    //    mt_mag[j]    = std::get<2>(result);
    //}

    if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 1) {
        printf("\n");
        printf("Charges and magnetic moments\n");
        for (int i = 0; i < 80; i++) {
            printf("-");
        }
        printf("\n");
        if (ctx_.full_potential()) {
            double total_core_leakage{0.0};
            printf("atom      charge    core leakage");
            if (ctx_.num_mag_dims()) {
                printf("              moment                |moment|");
            }
            printf("\n");
            for (int i = 0; i < 80; i++) {
                printf("-");
            }
            printf("\n");

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                double core_leakage = unit_cell_.atom(ia).symmetry_class().core_leakage();
                total_core_leakage += core_leakage;
                printf("%4i  %10.6f  %10.8e", ia, mt_charge[ia], core_leakage);
                if (ctx_.num_mag_dims()) {
                    vector3d<double> v(mt_mag[ia]);
                    printf("  [%8.4f, %8.4f, %8.4f]  %10.6f", v[0], v[1], v[2], v.length());
                }
                printf("\n");
            }

            printf("\n");
            printf("total core leakage    : %10.8e\n", total_core_leakage);
            printf("interstitial charge   : %10.6f\n", it_charge);
            if (ctx_.num_mag_dims()) {
                vector3d<double> v(it_mag);
                printf("interstitial moment   : [%8.4f, %8.4f, %8.4f], magnitude : %10.6f\n", v[0], v[1], v[2],
                       v.length());
            }
        } else {
            if (ctx_.num_mag_dims()) {
                printf("atom              moment                |moment|");
                printf("\n");
                for (int i = 0; i < 80; i++) {
                    printf("-");
                }
                printf("\n");

                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    vector3d<double> v(mt_mag[ia]);
                    printf("%4i  [%8.4f, %8.4f, %8.4f]  %10.6f", ia, v[0], v[1], v[2], v.length());
                    printf("\n");
                }

                printf("\n");
            }
        }
        printf("total charge          : %10.6f\n", total_charge);

        if (ctx_.num_mag_dims()) {
            vector3d<double> v(total_mag);
            printf("total moment          : [%8.4f, %8.4f, %8.4f], magnitude : %10.6f\n", v[0], v[1], v[2], v.length());
        }

        printf("\n");
        printf("Energy\n");
        for (int i = 0; i < 80; i++) {
            printf("-");
        }
        printf("\n");

        printf("valence_eval_sum          : %18.8f\n", evalsum1);
        if (ctx_.full_potential()) {
            printf("core_eval_sum             : %18.8f\n", evalsum2);
            printf("kinetic energy            : %18.8f\n", ekin);
            printf("enuc                      : %18.8f\n", enuc);
        }
        printf("<rho|V^{XC}>              : %18.8f\n", evxc);
        printf("<rho|E^{XC}>              : %18.8f\n", eexc);
        printf("<mag|B^{XC}>              : %18.8f\n", ebxc);
        printf("<rho|V^{H}>               : %18.8f\n", evha);
        if (!ctx_.full_potential()) {
            printf("one-electron contribution : %18.8f (Ha), %18.8f (Ry)\n", one_elec_en,
                   one_elec_en * 2); // eband + deband in QE
            printf("hartree contribution      : %18.8f\n", 0.5 * evha);
            printf("xc contribution           : %18.8f\n", eexc);
            printf("ewald contribution        : %18.8f\n", ewald_energy_);
            printf("PAW contribution          : %18.8f\n", potential_.PAW_total_energy());
        }
        if (ctx_.hubbard_correction()) {
            printf("Hubbard energy            : %18.8f (Ha), %18.8f (Ry)\n", potential_.U().hubbard_energy(),
                   potential_.U().hubbard_energy() * 2.0);
        }

        printf("Total energy              : %18.8f (Ha), %18.8f (Ry)\n", etot, etot * 2);

        printf("\n");
        printf("band gap (eV) : %18.8f\n", gap);
        printf("Efermi        : %18.8f\n", ef);
        printf("\n");
        // if (ctx_.control().verbosity_ >= 3 && !ctx_.full_potential()) {
        //    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        //        printf("atom: %i\n", ia);
        //        int nbf = unit_cell_.atom(ia).type().mt_basis_size();
        //        for (int j = 0; j < ctx_.num_mag_comp(); j++) {
        //            //printf("component of density matrix: %i\n", j);
        //            //for (int xi1 = 0; xi1 < nbf; xi1++) {
        //            //    for (int xi2 = 0; xi2 < nbf; xi2++) {
        //            //        auto z = density_.density_matrix()(xi1, xi2, j, ia);
        //            //        printf("(%f, %f) ", z.real(), z.imag());
        //            //    }
        //            //    printf("\n");
        //            //}
        //            printf("diagonal components of density matrix: %i\n", j);
        //            for (int xi2 = 0; xi2 < nbf; xi2++) {
        //                auto z = density_.density_matrix()(xi2, xi2, j, ia);
        //                printf("(%10.6f, %10.6f) ", z.real(), z.imag());
        //            }
        //            printf("\n");
        //        }
        //    }
        //}
    }
}

} // namespace sirius

