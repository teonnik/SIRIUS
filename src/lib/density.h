namespace sirius
{

class Density
{
    private:
        
        mdarray<double,5> mt_density_matrix_;
        
        std::vector<kpoint_data_set*> kpoints_;

        std::map<int,int>kpoint_index_by_id_;

        std::vector< std::pair<int,int> > dmat_spins_;

        mdarray<complex16,3> complex_gaunt_;

        template <int num_dmat> void reduce_zdens(int ia, mdarray<complex16,3>& zdens)
        {
            AtomType* type = global.atom(ia)->type();
            int mt_basis_size = type->mt_basis_size();

            for (int lm3 = 0; lm3 < global.lmmax_rho(); lm3++)
            {
                int l3 = l_by_lm(lm3);
                
                for (int j2 = 0; j2 < mt_basis_size; j2++)
                {
                    int l2 = type->indexb(j2).l;
                    int lm2 = type->indexb(j2).lm;
                    int idxrf2 = type->indexb(j2).idxrf;
        
                    int j1 = 0;

                    // compute only upper triangular block and later use the symmetry properties of the density matrix
                    for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
                    {
                        int l1 = type->indexr(idxrf1).l;
                        
                        if ((l1 + l2 + l3) % 2 == 0)
                        {
                            for (int lm1 = lm_by_l_m(l1, -l1); lm1 <= lm_by_l_m(l1, l1); lm1++, j1++) 
                            {
                                complex16 gc = complex_gaunt_(lm1, lm2, lm3);

                                switch(num_dmat)
                                {
                                    case 4:
                                        mt_density_matrix_(idxrf1, idxrf2, lm3, ia, 2) += 2.0 * real(zdens(j1, j2, 2) * gc); 
                                        mt_density_matrix_(idxrf1, idxrf2, lm3, ia, 3) -= 2.0 * imag(zdens(j1, j2, 2) * gc);
                                    case 2:
                                        mt_density_matrix_(idxrf1, idxrf2, lm3, ia, 1) += real(zdens(j1, j2, 1) * gc);
                                    case 1:
                                        mt_density_matrix_(idxrf1, idxrf2, lm3, ia, 0) += real(zdens(j1, j2, 0) * gc);
                                }
                            
                                /*mt_density_(idxrf1, idxrf2, lm3, ia, 0) += real(zdens(j1, j2, 0) * complex_gaunt_(lm1, lm2, lm3));

                                if (num_dmat == 2 || num_dmat == 4)
                                    mt_density_(idxrf1, idxrf2, lm3, ia, 1) += real(zdens(j1, j2, 1) * complex_gaunt_(lm1, lm2, lm3));

                                if (num_dmat == 4)
                                {
                                    mt_density_(idxrf1, idxrf2, lm3, ia, 2) += 2.0 * real(zdens(j1, j2, 2) * complex_gaunt_(lm1, lm2, lm3));
                                    mt_density_(idxrf1, idxrf2, lm3, ia, 3) -= 2.0 * imag(zdens(j1, j2, 2) * complex_gaunt_(lm1, lm2, lm3));
                                }*/
                            }
                        } 
                        else
                            j1 += (2 * l1 + 1);
                    }
                } // j2
            } // lm3
        }

        void add_k_contribution(kpoint_data_set& kp)
        {
            Timer t("sirius::Density::add_k_contribution");
            
            std::vector< std::pair<int,double> > bands;
            for (int j = 0; j < global.num_states(); j++)
            {
                double wo = kp.occupancy(j) * kp.weight();
                if (wo > 1e-14)
                    bands.push_back(std::pair<int,double>(j, wo));
            }
           
            // if we have ud and du spin blocks, don't compute one of them (du in this implementation)
            // because density matrix is symmetric
            int num_zdmat = (global.num_dmat() == 4) ? 3 : global.num_dmat();

            mdarray<complex16,3> zdens(global.max_mt_basis_size(), global.max_mt_basis_size(), num_zdmat);
            mdarray<complex16,3> wf1(global.max_mt_basis_size(), bands.size(), global.num_spins());
            mdarray<complex16,3> wf2(global.max_mt_basis_size(), bands.size(), global.num_spins());
       
            Timer t1("sirius::Density::add_k_contribution:zdens", false);
            Timer t2("sirius::Density::add_k_contribution:reduce_zdens", false);
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                t1.start();
                
                int offset_wf = global.atom(ia)->offset_aw();
                int mt_basis_size = global.atom(ia)->type()->mt_basis_size();
                
                for (int i = 0; i < (int)bands.size(); i++)
                    for (int ispn = 0; ispn < global.num_spins(); ispn++)
                    {
                        memcpy(&wf1(0, i, ispn), &kp.spinor_wave_function(offset_wf, ispn, bands[i].first), 
                               mt_basis_size * sizeof(complex16));
                        for (int j = 0; j < mt_basis_size; j++) 
                            wf2(j, i, ispn) = wf1(j, i, ispn) * bands[i].second;
                    }

                for (int j = 0; j < num_zdmat; j++)
                    gemm<cpu>(0, 2, mt_basis_size, mt_basis_size, bands.size(), complex16(1.0, 0.0), 
                        &wf1(0, 0, dmat_spins_[j].first), global.max_mt_basis_size(), 
                        &wf2(0, 0, dmat_spins_[j].second), global.max_mt_basis_size(),complex16(0.0, 0.0), 
                        &zdens(0, 0, j), global.max_mt_basis_size());
                
                t1.stop();

                t2.start();
                
                switch(global.num_dmat())
                {
                    case 1:
                        reduce_zdens<1>(ia, zdens);
                        break;

                    case 2:
                        reduce_zdens<2>(ia, zdens);
                        break;

                    case 4:
                        reduce_zdens<4>(ia, zdens);
                        break;

                    default:
                        error(__FILE__, __LINE__, "wrong number of density matrix components");
                }
                
                t2.stop();

            } // ia
        
            
            Timer t3("sirius::Density::add_k_contribution:it");
            mdarray<double,2> it_density(global.fft().size(), global.num_dmat());
            it_density.zero();
            
            mdarray<complex16,2> wfit(global.fft().size(), global.num_spins());

            for (int i = 0; i < (int)bands.size(); i++)
            {
                for (int ispn = 0; ispn < global.num_spins(); ispn++)
                {
                    global.fft().input(kp.num_gkvec(), kp.fft_index(), 
                                       &kp.spinor_wave_function(global.mt_basis_size(), ispn, bands[i].first));
                    global.fft().transform(1);
                    global.fft().output(&wfit(0, ispn));
                }
                
                double w = bands[i].second / global.omega();
                
                switch(global.num_dmat())
                {
                    case 4:
                        for (int ir = 0; ir < global.fft().size(); ir++)
                        {
                            complex16 z = wfit(ir, 0) * conj(wfit(ir, 1)) * w;
                            it_density(ir, 2) += 2.0 * real(z);
                            it_density(ir, 3) -= 2.0 * imag(z);
                        }
                    case 2:
                        for (int ir = 0; ir < global.fft().size(); ir++)
                            it_density(ir, 1) += real(wfit(ir, 1) * conj(wfit(ir, 1))) * w;
                    case 1:
                        for (int ir = 0; ir < global.fft().size(); ir++)
                            it_density(ir, 0) += real(wfit(ir, 0) * conj(wfit(ir, 0))) * w;
                }
            }
             
            t3.stop();
            
            for (int ir = 0; ir < global.fft().size(); ir++)
                global.charge_density().f_it(ir) += it_density(ir, 0);
       }

    public:
        
        void set_charge_density_ptr(double* rhomt, double* rhoir)
        {
            global.charge_density().set_rlm_ptr(rhomt);
            global.charge_density().set_it_ptr(rhoir);
            global.charge_density().zero();
            //error(__FILE__, __LINE__, "stop execution");
        }
    
        void initialize()
        {
            clear();

            global.charge_density().set_dimensions(global.lmax_rho(), global.max_num_mt_points(), global.num_atoms(), 
                                                   global.fft().size(), global.num_gvec());

            global.charge_density().allocate(pw_component);

            dmat_spins_.clear();
            dmat_spins_.push_back(std::pair<int,int>(0, 0));
            dmat_spins_.push_back(std::pair<int,int>(1, 1));
            dmat_spins_.push_back(std::pair<int,int>(0, 1));
            dmat_spins_.push_back(std::pair<int,int>(1, 0));
            
            complex_gaunt_.set_dimensions(global.lmmax_apw(), global.lmmax_apw(), global.lmmax_rho());
            complex_gaunt_.allocate();

            for (int l1 = 0; l1 <= global.lmax_apw(); l1++) 
            for (int m1 = -l1; m1 <= l1; m1++)
            {
                int lm1 = lm_by_l_m(l1, m1);
                for (int l2 = 0; l2 <= global.lmax_apw(); l2++)
                for (int m2 = -l2; m2 <= l2; m2++)
                {
                    int lm2 = lm_by_l_m(l2, m2);
                    for (int l3 = 0; l3 <= global.lmax_pot(); l3++)
                    for (int m3 = -l3; m3 <= l3; m3++)
                    {
                        int lm3 = lm_by_l_m(l3, m3);
                        complex_gaunt_(lm1, lm2, lm3) = SHT::complex_gaunt(l1, l3, l2, m1, m3, m2);
                    }
                }
            }

            mt_density_matrix_.set_dimensions(global.max_mt_radial_basis_size(), global.max_mt_radial_basis_size(), 
                                              global.lmmax_rho(), global.num_atoms(), global.num_dmat());
            mt_density_matrix_.allocate();
        }

        void clear()
        {
            for (int ik = 0; ik < (int)kpoints_.size(); ik++)
                delete kpoints_[ik];
            
            kpoints_.clear();
            kpoint_index_by_id_.clear();
        }
        
        void initial_density()
        {
            std::vector<double> enu;
            for (int i = 0; i < global.num_atom_types(); i++)
                global.atom_type(i)->solve_free_atom(1e-8, 1e-5, 1e-4, enu);

            global.charge_density().zero();

            double mt_charge = 0.0;
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                int nmtp = global.atom(ia)->type()->num_mt_points();
                Spline<double> rho(nmtp, global.atom(ia)->type()->radial_grid());
                for (int ir = 0; ir < nmtp; ir++)
                {
                    rho[ir] = global.atom(ia)->type()->free_atom_density(ir);
                    global.charge_density().f_rlm(0, ir, ia) = rho[ir] / y00;
                }
                rho.interpolate();

                // add charge of the MT sphere
                mt_charge += fourpi * rho.integrate(nmtp - 1, 2);
            }
            
            // distribute remaining charge
            for (int i = 0; i < global.fft().size(); i++)
                global.charge_density().f_it(i) = (global.num_electrons() - mt_charge) / global.volume_it();
        }

        void total_charge(void)
        {
            double charge = 0.0;

            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                int nmtp = global.atom(ia)->type()->num_mt_points();
                Spline<double> rho(nmtp, global.atom(ia)->type()->radial_grid());
                for (int ir = 0; ir < nmtp; ir++)
                    rho[ir] = global.charge_density().f_rlm(0, ir, ia);
                rho.interpolate();
                charge += rho.integrate(2) * fourpi * y00;
            }

            double dv = global.omega() / global.fft().size();
            for (int ir = 0; ir < global.fft().size(); ir++)
                charge += global.charge_density().f_it(ir) * global.step_function(ir) * dv;

            if (fabs(charge - global.num_electrons()) > 1e-10)
            {
                std::stringstream s;
                s << "Wrong nuber of electrons " << charge;
                error(__FILE__, __LINE__, s);
            }
        }

        void generate()
        {
            Timer t("sirius::Density::generate");
            
            double wt = 0.0;
            double ot = 0.0;
            for (int ik = 0; ik < (int)kpoints_.size(); ik++)
            {
                wt += kpoints_[ik]->weight();
                for (int j = 0; j < global.num_states(); j++)
                    ot += kpoints_[ik]->weight() * kpoints_[ik]->occupancy(j);
            }

            if (fabs(wt - 1.0) > 1e-12)
                error(__FILE__, __LINE__, "kpoint weights don't sum to one");

            if (fabs(ot - global.num_valence_electrons()) > 1e-12)
                error(__FILE__, __LINE__, "wrong occupancies");

            // generate radial functions and integrals
            band.radial();

            // generate plane-wave coefficients of the potential in the interstitial region
            for (int ir = 0; ir < global.fft().size(); ir++)
                 global.effective_potential().f_it(ir) *= global.step_function(ir);

            global.fft().input(global.effective_potential().f_it());
            global.fft().forward();
            global.fft().output(global.num_gvec(), global.fft_index(), global.effective_potential().f_pw());

            // zero auxiliary density matrix
            mt_density_matrix_.zero();

            // zero density
            global.charge_density().zero();

            for (int ik = 0; ik < num_kpoints(); ik++)
            {
                // solve secular equatiion and generate wave functions
                band.find_eigen_states(*kpoint(ik));
                // add to charge density and magnetization
                add_k_contribution(*kpoint(ik));
            }

            Timer t1("sirius::Density::generate:convert_mt");
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                for (int lm = 0; lm < global.lmmax_rho(); lm++)
                {
                    for (int i = 0; i < global.num_dmat(); i++)
                        for (int idxrf = 0; idxrf < global.atom(ia)->type()->mt_radial_basis_size(); idxrf++)
                            mt_density_matrix_(idxrf, idxrf, lm, ia, i) *= 0.5; 
                    
                    for (int idxrf2 = 0; idxrf2 < global.atom(ia)->type()->mt_radial_basis_size(); idxrf2++)
                    {
                        for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
                        {
                            for (int ir = 0; ir < global.atom(ia)->type()->num_mt_points(); ir++)
                                global.charge_density().f_rlm(lm, ir, ia) += 
                                    2 * mt_density_matrix_(idxrf1, idxrf2, lm, ia, 0) * 
                                    global.atom(ia)->symmetry_class()->radial_function(ir, idxrf1) * 
                                    global.atom(ia)->symmetry_class()->radial_function(ir, idxrf2);
                        }
                    }
                }
            }
            t1.stop();
        }

        void add_kpoint(int kpoint_id, double* vk, double weight)
        {
            if (kpoint_index_by_id_.count(kpoint_id))
                error(__FILE__, __LINE__, "kpoint is already in list");

            kpoints_.push_back(new kpoint_data_set(vk, weight));
            kpoint_index_by_id_[kpoint_id] = kpoints_.size();

            std::vector<double> initial_occupancies(global.num_states(), 0.0);

            // in case of non-magnetic, or magnetic non-collinear case occupy first N bands
            if (global.num_dmat() == 1 || global.num_dmat() == 4)
            {
                int m = global.num_valence_electrons() / global.max_occupancy();
                for (int i = 0; i < m; i++)
                    initial_occupancies[i] = double(global.max_occupancy());
                initial_occupancies[m] = double(global.num_valence_electrons() % global.max_occupancy());
            }
            else // otherwise occupy up and down bands
            {
                int m = global.num_valence_electrons() / 2;
                for (int i = 0; i < m; i++)
                    initial_occupancies[i] = initial_occupancies[i + global.num_fv_states()] = 1.0;
                initial_occupancies[m] = initial_occupancies[m + global.num_fv_states()] = 
                    0.5 * global.num_valence_electrons() - double(m);
            }

            kpoints_.back()->set_occupancies(&initial_occupancies[0]);
        }

        void set_occupancies(int kpoint_id, double* occupancies)
        {
            kpoints_[kpoint_index_by_id_[kpoint_id]]->set_occupancies(occupancies);
        }

        inline kpoint_data_set* kpoint(int ik)
        {
            return kpoints_[ik];
        }

        inline int num_kpoints()
        {
            return kpoints_.size();
        }

        void print_info()
        {
            printf("\n");
            printf("Density\n");
            for (int i = 0; i < 80; i++) printf("-");
            printf("\n");
            printf("number of k-points : %i\n", (int)kpoints_.size());
            for (int ik = 0; ik < (int)kpoints_.size(); ik++)
                printf("ik=%4i    vk=%12.6f %12.6f %12.6f    weight=%12.6f\n", ik, kpoints_[ik]->vk()[0], 
                                                                                   kpoints_[ik]->vk()[1], 
                                                                                   kpoints_[ik]->vk()[2], 
                                                                                   kpoints_[ik]->weight());
        }
};

Density density;

};