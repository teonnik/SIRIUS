// Copyright (c) 2013-2018 Anton Kozhevnikov, Mathieu Taillefumier, Thomas Schulthess
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

#include "K_point/k_point.hpp"

namespace sirius {

    void K_point::generate_gkvec(double gk_cutoff__) {
        PROFILE("sirius::K_point::generate_gkvec");

        if (ctx_.full_potential() && (gk_cutoff__ * unit_cell_.max_mt_radius() > ctx_.lmax_apw()) &&
            comm_.rank() == 0 && ctx_.control().verbosity_ >= 0) {
            std::stringstream s;
            s << "G+k cutoff (" << gk_cutoff__ << ") is too large for a given lmax ("
              << ctx_.lmax_apw() << ") and a maximum MT radius (" << unit_cell_.max_mt_radius() << ")" << std::endl
              << "suggested minimum value for lmax : " << int(gk_cutoff__ * unit_cell_.max_mt_radius()) + 1;
            WARNING(s);
        }

        if (gk_cutoff__ * 2 > ctx_.pw_cutoff()) {
            std::stringstream s;
            s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
              << "  pw cutoff : " << ctx_.pw_cutoff() << std::endl
              << "  doubled G+k cutoff : " << gk_cutoff__ * 2;
            TERMINATE(s);
        }

        /* create G+k vectors; communicator of the coarse FFT grid is used because wave-functions will be transformed
         * only on the coarse grid; G+k-vectors will be distributed between MPI ranks assigned to the k-point */
        gkvec_ = std::unique_ptr<Gvec>(new Gvec(vk_, ctx_.unit_cell().reciprocal_lattice_vectors(), gk_cutoff__, comm(),
                                                ctx_.gamma_point()));

        gkvec_partition_ = std::unique_ptr<Gvec_partition>(new Gvec_partition(*gkvec_, ctx_.comm_fft_coarse(),
                                                                              ctx_.comm_band_ortho_fft_coarse()));

        gkvec_offset_ = gkvec().gvec_offset(comm().rank());
    }

    void K_point::update() {
        PROFILE("sirius::K_point::update");

        gkvec_->lattice_vectors(ctx_.unit_cell().reciprocal_lattice_vectors());

        if (ctx_.full_potential()) {
            if (ctx_.iterative_solver_input().type_ == "exact") {
                alm_coeffs_row_ = std::unique_ptr<Matching_coefficients>(
                        new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_row(), igk_row_, gkvec()));
                alm_coeffs_col_ = std::unique_ptr<Matching_coefficients>(
                        new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_col(), igk_col_, gkvec()));
            }
            alm_coeffs_loc_ = std::unique_ptr<Matching_coefficients>(
                    new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_loc(), igk_loc_, gkvec()));
        }

        if (!ctx_.full_potential()) {
            /* compute |beta> projectors for atom types */
            beta_projectors_ = std::unique_ptr<Beta_projectors>(new Beta_projectors(ctx_, gkvec(), igk_loc_));

            if (ctx_.iterative_solver_input().type_ == "exact") {
                beta_projectors_row_ = std::unique_ptr<Beta_projectors>(new Beta_projectors(ctx_, gkvec(), igk_row_));
                beta_projectors_col_ = std::unique_ptr<Beta_projectors>(new Beta_projectors(ctx_, gkvec(), igk_col_));

            }

            //if (false) {
            //    p_mtrx_ = mdarray<double_complex, 3>(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), unit_cell_.num_atom_types());
            //    p_mtrx_.zero();

            //    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            //        auto& atom_type = unit_cell_.atom_type(iat);

            //        if (!atom_type.pp_desc().augment) {
            //            continue;
            //        }
            //        int nbf = atom_type.mt_basis_size();
            //        int ofs = atom_type.offset_lo();

            //        matrix<double_complex> qinv(nbf, nbf);
            //        for (int xi1 = 0; xi1 < nbf; xi1++) {
            //            for (int xi2 = 0; xi2 < nbf; xi2++) {
            //                qinv(xi2, xi1) = ctx_.augmentation_op(iat).q_mtrx(xi2, xi1);
            //            }
            //        }
            //        linalg<device_t::CPU>::geinv(nbf, qinv);
            //
            //        /* compute P^{+}*P */
            //        linalg<device_t::CPU>::gemm(2, 0, nbf, nbf, num_gkvec_loc(),
            //                          beta_projectors_->beta_gk_t().at<CPU>(0, ofs), beta_projectors_->beta_gk_t().ld(),
            //                          beta_projectors_->beta_gk_t().at<CPU>(0, ofs), beta_projectors_->beta_gk_t().ld(),
            //                          &p_mtrx_(0, 0, iat), p_mtrx_.ld());
            //        comm().allreduce(&p_mtrx_(0, 0, iat), unit_cell_.max_mt_basis_size() * unit_cell_.max_mt_basis_size());

            //        for (int xi1 = 0; xi1 < nbf; xi1++) {
            //            for (int xi2 = 0; xi2 < nbf; xi2++) {
            //                qinv(xi2, xi1) += p_mtrx_(xi2, xi1, iat);
            //            }
            //        }
            //        /* compute (Q^{-1} + P^{+}*P)^{-1} */
            //        linalg<device_t::CPU>::geinv(nbf, qinv);
            //        for (int xi1 = 0; xi1 < nbf; xi1++) {
            //            for (int xi2 = 0; xi2 < nbf; xi2++) {
            //                p_mtrx_(xi2, xi1, iat) = qinv(xi2, xi1);
            //            }
            //        }
            //    }
            //}
        }

    }

    void K_point::get_fv_eigen_vectors(mdarray<double_complex, 2> &fv_evec__) const {
        assert((int) fv_evec__.size(0) >= gklo_basis_size());
        assert((int) fv_evec__.size(1) == ctx_.num_fv_states());
        assert(gklo_basis_size_row() == fv_eigen_vectors_.num_rows_local());

        mdarray<double_complex, 1> tmp(gklo_basis_size_row());

        fv_evec__.zero();

        for (int ist = 0; ist < ctx_.num_fv_states(); ist++) {
            auto loc = fv_eigen_vectors_.spl_col().location(ist);
            if (loc.rank == fv_eigen_vectors_.rank_col()) {
                std::copy(&fv_eigen_vectors_(0, loc.local_index),
                          &fv_eigen_vectors_(0, loc.local_index) + gklo_basis_size_row(),
                          &tmp(0));
            }
            fv_eigen_vectors_.blacs_grid().comm_col().bcast(&tmp(0), gklo_basis_size_row(), loc.rank);
            for (int jloc = 0; jloc < gklo_basis_size_row(); jloc++) {
                int j = fv_eigen_vectors_.irow(jloc);
                fv_evec__(j, ist) = tmp(jloc);
            }
            fv_eigen_vectors_.blacs_grid().comm_row().allreduce(&fv_evec__(0, ist), gklo_basis_size());
        }
    }

    //== void K_point::check_alm(int num_gkvec_loc, int ia, mdarray<double_complex, 2>& alm)
//== {
//==     static SHT* sht = NULL;
//==     if (!sht) sht = new SHT(ctx_.lmax_apw());
//==
//==     Atom* atom = unit_cell_.atom(ia);
//==     Atom_type* type = atom->type();
//==
//==     mdarray<double_complex, 2> z1(sht->num_points(), type->mt_aw_basis_size());
//==     for (int i = 0; i < type->mt_aw_basis_size(); i++)
//==     {
//==         int lm = type->indexb(i).lm;
//==         int idxrf = type->indexb(i).idxrf;
//==         double rf = atom->symmetry_class()->radial_function(atom->num_mt_points() - 1, idxrf);
//==         for (int itp = 0; itp < sht->num_points(); itp++)
//==         {
//==             z1(itp, i) = sht->ylm_backward(lm, itp) * rf;
//==         }
//==     }
//==
//==     mdarray<double_complex, 2> z2(sht->num_points(), num_gkvec_loc);
//==     blas<CPU>::gemm(0, 2, sht->num_points(), num_gkvec_loc, type->mt_aw_basis_size(), z1.ptr(), z1.ld(),
//==                     alm.ptr(), alm.ld(), z2.ptr(), z2.ld());
//==
//==     vector3d<double> vc = unit_cell_.get_cartesian_coordinates(unit_cell_.atom(ia)->position());
//==
//==     double tdiff = 0;
//==     for (int igloc = 0; igloc < num_gkvec_loc; igloc++)
//==     {
//==         vector3d<double> gkc = gkvec_cart(igkglob(igloc));
//==         for (int itp = 0; itp < sht->num_points(); itp++)
//==         {
//==             double_complex aw_value = z2(itp, igloc);
//==             vector3d<double> r;
//==             for (int x = 0; x < 3; x++) r[x] = vc[x] + sht->coord(x, itp) * type->mt_radius();
//==             double_complex pw_value = exp(double_complex(0, Utils::scalar_product(r, gkc))) / sqrt(unit_cell_.omega());
//==             tdiff += abs(pw_value - aw_value);
//==         }
//==     }
//==
//==     printf("atom : %i  absolute alm error : %e  average alm error : %e\n",
//==            ia, tdiff, tdiff / (num_gkvec_loc * sht->num_points()));
//== }


//Periodic_function<double_complex>* K_point::spinor_wave_function_component(Band* band, int lmax, int ispn, int jloc)
//{
//    Timer t("sirius::K_point::spinor_wave_function_component");
//
//    int lmmax = Utils::lmmax_by_lmax(lmax);
//
//    Periodic_function<double_complex, index_order>* func =
//        new Periodic_function<double_complex, index_order>(ctx_, lmax);
//    func->allocate(ylm_component | it_component);
//    func->zero();
//
//    if (basis_type == pwlo)
//    {
//        if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//
//        double fourpi_omega = fourpi / sqrt(ctx_.omega());
//
//        for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//        {
//            int igk = igkglob(igkloc);
//            double_complex z1 = spinor_wave_functions_(ctx_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//
//            // TODO: possilbe optimization with zgemm
//            for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//            {
//                int iat = ctx_.atom_type_index_by_id(ctx_.atom(ia)->type_id());
//                double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//
//                #pragma omp parallel for default(shared)
//                for (int lm = 0; lm < lmmax; lm++)
//                {
//                    int l = l_by_lm_(lm);
//                    double_complex z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc));
//                    for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//                        func->f_ylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//                }
//            }
//        }
//
//        for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//        {
//            Platform::allreduce(&func->f_ylm(0, 0, ia), lmmax * ctx_.max_num_mt_points(),
//                                ctx_.mpi_grid().communicator(1 << band->dim_row()));
//        }
//    }
//
//    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//    {
//        for (int i = 0; i < ctx_.atom(ia)->type()->mt_basis_size(); i++)
//        {
//            int lm = ctx_.atom(ia)->type()->indexb(i).lm;
//            int idxrf = ctx_.atom(ia)->type()->indexb(i).idxrf;
//            switch (index_order)
//            {
//                case angular_radial:
//                {
//                    for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(lm, ir, ia) +=
//                            spinor_wave_functions_(ctx_.atom(ia)->offset_wf() + i, ispn, jloc) *
//                            ctx_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//                case radial_angular:
//                {
//                    for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(ir, lm, ia) +=
//                            spinor_wave_functions_(ctx_.atom(ia)->offset_wf() + i, ispn, jloc) *
//                            ctx_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//            }
//        }
//    }
//
//    // in principle, wave function must have an overall e^{ikr} phase factor
//    ctx_.fft().input(num_gkvec(), &fft_index_[0],
//                            &spinor_wave_functions_(ctx_.mt_basis_size(), ispn, jloc));
//    ctx_.fft().transform(1);
//    ctx_.fft().output(func->f_it());
//
//    for (int i = 0; i < ctx_.fft().size(); i++) func->f_it(i) /= sqrt(ctx_.omega());
//
//    return func;
//}

//== void K_point::spinor_wave_function_component_mt(int lmax, int ispn, int jloc, mt_functions<double_complex>& psilm)
//== {
//==     Timer t("sirius::K_point::spinor_wave_function_component_mt");
//==
//==     //int lmmax = Utils::lmmax_by_lmax(lmax);
//==
//==     psilm.zero();
//==
//==     //if (basis_type == pwlo)
//==     //{
//==     //    if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//==
//==     //    double fourpi_omega = fourpi / sqrt(ctx_.omega());
//==
//==     //    mdarray<double_complex, 2> zm(ctx_.max_num_mt_points(),  num_gkvec_row());
//==
//==     //    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     //    {
//==     //        int iat = ctx_.atom_type_index_by_id(ctx_.atom(ia)->type_id());
//==     //        for (int l = 0; l <= lmax; l++)
//==     //        {
//==     //            #pragma omp parallel for default(shared)
//==     //            for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//==     //            {
//==     //                int igk = igkglob(igkloc);
//==     //                double_complex z1 = spinor_wave_functions_(ctx_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//==     //                double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia) * zil_[l];
//==     //                for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//==     //                    zm(ir, igkloc) = z2 * (*sbessel_[igkloc])(ir, l, iat);
//==     //            }
//==     //            blas<CPU>::gemm(0, 2, ctx_.atom(ia)->num_mt_points(), (2 * l + 1), num_gkvec_row(),
//==     //                            &zm(0, 0), zm.ld(), &gkvec_ylm_(Utils::lm_by_l_m(l, -l), 0), gkvec_ylm_.ld(),
//==     //                            &fylm(0, Utils::lm_by_l_m(l, -l), ia), fylm.ld());
//==     //        }
//==     //    }
//==     //    //for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//==     //    //{
//==     //    //    int igk = igkglob(igkloc);
//==     //    //    double_complex z1 = spinor_wave_functions_(ctx_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//==
//==     //    //    // TODO: possilbe optimization with zgemm
//==     //    //    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     //    //    {
//==     //    //        int iat = ctx_.atom_type_index_by_id(ctx_.atom(ia)->type_id());
//==     //    //        double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//==     //    //
//==     //    //        #pragma omp parallel for default(shared)
//==     //    //        for (int lm = 0; lm < lmmax; lm++)
//==     //    //        {
//==     //    //            int l = l_by_lm_(lm);
//==     //    //            double_complex z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc));
//==     //    //            for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//==     //    //                fylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//==     //    //        }
//==     //    //    }
//==     //    //}
//==
//==     //    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     //    {
//==     //        Platform::allreduce(&fylm(0, 0, ia), lmmax * ctx_.max_num_mt_points(),
//==     //                            ctx_.mpi_grid().communicator(1 << band->dim_row()));
//==     //    }
//==     //}
//==
//==     for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     {
//==         for (int i = 0; i < ctx_.atom(ia)->type()->mt_basis_size(); i++)
//==         {
//==             int lm = ctx_.atom(ia)->type()->indexb(i).lm;
//==             int idxrf = ctx_.atom(ia)->type()->indexb(i).idxrf;
//==             for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//==             {
//==                 psilm(lm, ir, ia) +=
//==                     spinor_wave_functions_(ctx_.atom(ia)->offset_wf() + i, ispn, jloc) *
//==                     ctx_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//==             }
//==         }
//==     }
//== }

    void K_point::test_spinor_wave_functions(int use_fft)
    {
        STOP();

//==     if (num_ranks() > 1) error_local(__FILE__, __LINE__, "test of spinor wave functions on multiple ranks is not implemented");
//==
//==     std::vector<double_complex> v1[2];
//==     std::vector<double_complex> v2;
//==
//==     if (use_fft == 0 || use_fft == 1) v2.resize(fft_->size());
//==
//==     if (use_fft == 0)
//==     {
//==         for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) v1[ispn].resize(num_gkvec());
//==     }
//==
//==     if (use_fft == 1)
//==     {
//==         for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) v1[ispn].resize(fft_->size());
//==     }
//==
//==     double maxerr = 0;
//==
//==     for (int j1 = 0; j1 < ctx_.num_bands(); j1++)
//==     {
//==         if (use_fft == 0)
//==         {
//==             for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==             {
//==                 fft_->input(num_gkvec(), gkvec_.index_map(),
//==                                        &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j1));
//==                 fft_->transform(1);
//==                 fft_->output(&v2[0]);
//==
//==                 for (int ir = 0; ir < fft_->size(); ir++) v2[ir] *= ctx_.step_function()->theta_r(ir);
//==
//==                 fft_->input(&v2[0]);
//==                 fft_->transform(-1);
//==                 fft_->output(num_gkvec(), gkvec_.index_map(), &v1[ispn][0]);
//==             }
//==         }
//==
//==         if (use_fft == 1)
//==         {
//==             for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==             {
//==                 fft_->input(num_gkvec(), gkvec_.index_map(),
//==                                        &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j1));
//==                 fft_->transform(1);
//==                 fft_->output(&v1[ispn][0]);
//==             }
//==         }
//==
//==         for (int j2 = 0; j2 < ctx_.num_bands(); j2++)
//==         {
//==             double_complex zsum(0, 0);
//==             for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==             {
//==                 for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==                 {
//==                     int offset_wf = unit_cell_.atom(ia)->offset_wf();
//==                     Atom_type* type = unit_cell_.atom(ia)->type();
//==                     Atom_symmetry_class* symmetry_class = unit_cell_.atom(ia)->symmetry_class();
//==
//==                     for (int l = 0; l <= ctx_.lmax_apw(); l++)
//==                     {
//==                         int ordmax = type->indexr().num_rf(l);
//==                         for (int io1 = 0; io1 < ordmax; io1++)
//==                         {
//==                             for (int io2 = 0; io2 < ordmax; io2++)
//==                             {
//==                                 for (int m = -l; m <= l; m++)
//==                                 {
//==                                     zsum += conj(spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m, io1), ispn, j1)) *
//==                                             spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m, io2), ispn, j2) *
//==                                             symmetry_class->o_radial_integral(l, io1, io2);
//==                                 }
//==                             }
//==                         }
//==                     }
//==                 }
//==             }
//==
//==             if (use_fft == 0)
//==             {
//==                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==                {
//==                    for (int ig = 0; ig < num_gkvec(); ig++)
//==                        zsum += conj(v1[ispn][ig]) * spinor_wave_functions_(unit_cell_.mt_basis_size() + ig, ispn, j2);
//==                }
//==             }
//==
//==             if (use_fft == 1)
//==             {
//==                 for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==                 {
//==                     fft_->input(num_gkvec(), gkvec_.index_map(), &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j2));
//==                     fft_->transform(1);
//==                     fft_->output(&v2[0]);
//==
//==                     for (int ir = 0; ir < fft_->size(); ir++)
//==                         zsum += std::conj(v1[ispn][ir]) * v2[ir] * ctx_.step_function()->theta_r(ir) / double(fft_->size());
//==                 }
//==             }
//==
//==             if (use_fft == 2)
//==             {
//==                 STOP();
//==                 //for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
//==                 //{
//==                 //    for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
//==                 //    {
//==                 //        int ig3 = ctx_.gvec().index_g12(ig1, ig2);
//==                 //        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==                 //        {
//==                 //            zsum += std::conj(spinor_wave_functions_(unit_cell_.mt_basis_size() + ig1, ispn, j1)) *
//==                 //                    spinor_wave_functions_(unit_cell_.mt_basis_size() + ig2, ispn, j2) *
//==                 //                    ctx_.step_function()->theta_pw(ig3);
//==                 //        }
//==                 //    }
//==                 //}
//==             }
//==
//==             zsum = (j1 == j2) ? zsum - double_complex(1.0, 0.0) : zsum;
//==             maxerr = std::max(maxerr, std::abs(zsum));
//==         }
//==     }
//==     std :: cout << "maximum error = " << maxerr << std::endl;
    }

/** The following HDF5 data structure is created:
  \verbatim
  /K_point_set/ik/vk
  /K_point_set/ik/band_energies
  /K_point_set/ik/band_occupancies
  /K_point_set/ik/gkvec
  /K_point_set/ik/gvec
  /K_point_set/ik/bands/ibnd/spinor_wave_function/ispn/pw
  /K_point_set/ik/bands/ibnd/spinor_wave_function/ispn/mt
  \endverbatim
*/
    void K_point::save(std::string const& name__, int id__) const
    {
        /* rank 0 creates placeholders in the HDF5 file */
        if (comm().rank() == 0) {
            /* open file with write access */
            HDF5_tree fout(name__, hdf5_access_t::read_write);
            /* create /K_point_set/ik */
            fout["K_point_set"].create_node(id__);
            fout["K_point_set"][id__].write("vk", &vk_[0], 3);
            fout["K_point_set"][id__].write("band_energies", band_energies_);
            fout["K_point_set"][id__].write("band_occupancies", band_occupancies_);

            /* save the entire G+k object */
            //TODO: only the list of z-columns is probably needed to recreate the G+k vectors
            serializer s;
            gkvec().pack(s);
            fout["K_point_set"][id__].write("gkvec", s.stream());

            /* save the order of G-vectors */
            mdarray<int, 2> gv(3, num_gkvec());
            for (int i = 0; i < num_gkvec(); i++) {
                auto v = gkvec().gvec(i);
                for (int x: {0, 1, 2}) {
                    gv(x, i) = v[x];
                }
            }
            fout["K_point_set"][id__].write("gvec", gv);
            fout["K_point_set"][id__].create_node("bands");
            for (int i = 0; i < ctx_.num_bands(); i++) {
                fout["K_point_set"][id__]["bands"].create_node(i);
                fout["K_point_set"][id__]["bands"][i].create_node("spinor_wave_function");
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    fout["K_point_set"][id__]["bands"][i]["spinor_wave_function"].create_node(ispn);
                }
            }
        }
        /* wait for rank 0 */
        comm().barrier();
        int gkvec_count = gkvec().count();
        int gkvec_offset = gkvec().offset();
        std::vector<double_complex> wf_tmp(num_gkvec());

        std::unique_ptr<HDF5_tree> fout;
        /* rank 0 opens a file */
        if (comm().rank() == 0) {
            fout = std::unique_ptr<HDF5_tree>(new HDF5_tree(name__, hdf5_access_t::read_write));
        }

        /* store wave-functions */
        for (int i = 0; i < ctx_.num_bands(); i++) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                /* gather full column of PW coefficients on rank 0 */
                comm().gather(&spinor_wave_functions_->pw_coeffs(ispn).prime(0, i), wf_tmp.data(), gkvec_offset, gkvec_count, 0);
                if (comm().rank() == 0) {
                    (*fout)["K_point_set"][id__]["bands"][i]["spinor_wave_function"][ispn].write("pw", wf_tmp);
                }
            }
            comm().barrier();
        }
    }

    void K_point::load(HDF5_tree h5in, int id)
    {
        STOP();
        //== band_energies_.resize(ctx_.num_bands());
        //== h5in[id].read("band_energies", band_energies_);

        //== band_occupancies_.resize(ctx_.num_bands());
        //== h5in[id].read("band_occupancies", band_occupancies_);
        //==
        //== h5in[id].read_mdarray("fv_eigen_vectors", fv_eigen_vectors_panel_);
        //== h5in[id].read_mdarray("sv_eigen_vectors", sv_eigen_vectors_);
    }

//== void K_point::save_wave_functions(int id)
//== {
//==     if (ctx_.mpi_grid().root(1 << _dim_col_))
//==     {
//==         HDF5_tree fout(storage_file_name, false);
//==
//==         fout["K_points"].create_node(id);
//==         fout["K_points"][id].write("coordinates", &vk_[0], 3);
//==         fout["K_points"][id].write("mtgk_size", mtgk_size());
//==         fout["K_points"][id].create_node("spinor_wave_functions");
//==         fout["K_points"][id].write("band_energies", &band_energies_[0], ctx_.num_bands());
//==         fout["K_points"][id].write("band_occupancies", &band_occupancies_[0], ctx_.num_bands());
//==     }
//==
//==     Platform::barrier(ctx_.mpi_grid().communicator(1 << _dim_col_));
//==
//==     mdarray<double_complex, 2> wfj(NULL, mtgk_size(), ctx_.num_spins());
//==     for (int j = 0; j < ctx_.num_bands(); j++)
//==     {
//==         int rank = ctx_.spl_spinor_wf_col().location(_splindex_rank_, j);
//==         int offs = ctx_.spl_spinor_wf_col().location(_splindex_offs_, j);
//==         if (ctx_.mpi_grid().coordinate(_dim_col_) == rank)
//==         {
//==             HDF5_tree fout(storage_file_name, false);
//==             wfj.set_ptr(&spinor_wave_functions_(0, 0, offs));
//==             fout["K_points"][id]["spinor_wave_functions"].write_mdarray(j, wfj);
//==         }
//==         Platform::barrier(ctx_.mpi_grid().communicator(_dim_col_));
//==     }
//== }
//==
//== void K_point::load_wave_functions(int id)
//== {
//==     HDF5_tree fin(storage_file_name, false);
//==
//==     int mtgk_size_in;
//==     fin["K_points"][id].read("mtgk_size", &mtgk_size_in);
//==     if (mtgk_size_in != mtgk_size()) error_local(__FILE__, __LINE__, "wrong wave-function size");
//==
//==     band_energies_.resize(ctx_.num_bands());
//==     fin["K_points"][id].read("band_energies", &band_energies_[0], ctx_.num_bands());
//==
//==     band_occupancies_.resize(ctx_.num_bands());
//==     fin["K_points"][id].read("band_occupancies", &band_occupancies_[0], ctx_.num_bands());
//==
//==     spinor_wave_functions_.set_dimensions(mtgk_size(), ctx_.num_spins(),
//==                                           ctx_.spl_spinor_wf_col().local_size());
//==     spinor_wave_functions_.allocate();
//==
//==     mdarray<double_complex, 2> wfj(NULL, mtgk_size(), ctx_.num_spins());
//==     for (int jloc = 0; jloc < ctx_.spl_spinor_wf_col().local_size(); jloc++)
//==     {
//==         int j = ctx_.spl_spinor_wf_col(jloc);
//==         wfj.set_ptr(&spinor_wave_functions_(0, 0, jloc));
//==         fin["K_points"][id]["spinor_wave_functions"].read_mdarray(j, wfj);
//==     }
//== }

}