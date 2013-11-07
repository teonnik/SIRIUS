#include <sirius.h>

using namespace sirius;

//== // N : subspace dimension
//== // n : required number of eigen pairs
//== // bs : block size (number of new basis functions)
//== // phi : basis functions
//== // hphi : H|phi>
//== void subspace_diag(Global& parameters, int N, int n, int bs, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& hphi, 
//==                    mdarray<complex16, 2>& res, complex16 v0, mdarray<complex16, 2>& evec, std::vector<double>& eval)
//== {
//==     //== for (int i = 0; i < N; i++)
//==     //== {
//==     //==     for (int j = 0; j < N; j++)
//==     //==     {
//==     //==         complex16 z(0, 0);
//==     //==         for (int ig = 0; ig < parameters.num_gvec(); ig++)
//==     //==         {
//==     //==             z += conj(phi(ig, i)) * phi(ig, j);
//==     //==         }
//==     //==         if (i == j) z -= 1.0;
//==     //==         if (abs(z) > 1e-12) error_local(__FILE__, __LINE__, "basis is not orthogonal");
//==     //==     }
//==     //== }
//== 
//==     standard_evp* solver = new standard_evp_lapack();
//== 
//==     eval.resize(N);
//== 
//==     mdarray<complex16, 2> hmlt(N, N);
//==     blas<cpu>::gemm(2, 0, N, N, parameters.num_gvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());
//==     //== for (int i = 0; i < N; i++)
//==     //== {
//==     //==     for (int j = 0; j < N; j++)
//==     //==     {
//==     //==         for (int ig = 0; ig < parameters.num_gvec(); ig++) hmlt(i, j) += conj(phi(ig, i)) * hphi(ig, j);
//==     //==     }
//==     //== }
//== 
//==     solver->solve(N, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
//== 
//==     delete solver;
//== 
//==     printf("\n");
//==     printf("Lowest eigen-values : \n");
//==     for (int i = 0; i < std::min(N, 10); i++)
//==     {
//==         printf("i : %i,  eval : %16.10f\n", i, eval[i]);
//==     }
//== 
//==     // compute residuals
//==     res.zero();
//==     for (int j = 0; j < bs; j++)
//==     {
//==         int i = j; //n - bs + j;
//==         for (int mu = 0; mu < N; mu++)
//==         {
//==             for (int ig = 0; ig < parameters.num_gvec(); ig++)
//==             {
//==                 res(ig, j) += (evec(mu, i) * hphi(ig, mu) - eval[i] * evec(mu, i) * phi(ig, mu));
//==             }
//==         }
//==         double norm = 0.0;
//==         for (int ig = 0; ig < parameters.num_gvec(); ig++) norm += real(conj(res(ig, j)) * res(ig, j));
//==         for (int ig = 0; ig < parameters.num_gvec(); ig++) res(ig, j) /= sqrt(norm);
//==     }
//==     
//==     // additional basis vectors
//==     for (int j = 0; j < bs; j++)
//==     {
//==         int i = j; //n - bs + j;
//==         for (int ig = 0; ig < parameters.num_gvec(); ig++)
//==         {
//==             complex16 t = pow(parameters.gvec_len(ig), 2) / 2.0 + v0 - eval[i];
//==             if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
//==             res(ig, j) /= t;
//==         }
//==     }
//== }
//== 

void apply_h(Global& parameters, K_point& kp, int n, std::vector<complex16>& v_r, complex16* phi__, complex16* hphi__)
{
    Timer t("apply_h");

    mdarray<complex16, 2> phi(phi__, kp.num_gkvec(), n);
    mdarray<complex16, 2> hphi(hphi__, kp.num_gkvec(), n);
    std::vector<complex16> phi_r(parameters.fft().size());

    for (int i = 0; i < n; i++)
    {
        parameters.fft().input(kp.num_gkvec(), kp.fft_index(), &phi(0, i));
        parameters.fft().transform(1);
        parameters.fft().output(&phi_r[0]);

        for (int ir = 0; ir < parameters.fft().size(); ir++) phi_r[ir] *= v_r[ir];

        parameters.fft().input(&phi_r[0]);
        parameters.fft().transform(-1);
        parameters.fft().output(kp.num_gkvec(), kp.fft_index(), &hphi(0, i));

        for (int ig = 0; ig < kp.num_gkvec(); ig++) hphi(ig, i) += phi(ig, i) * pow(kp.gkvec_cart(ig).length(), 2) / 2.0;
    }
}

//==int diag_davidson(Global& parameters, int niter, int bs, int n, std::vector<complex16>& v_pw, mdarray<complex16, 2>& phi, 
//==                  mdarray<complex16, 2>& evec)
//=={
//==    std::vector<complex16> v_r(parameters.fft().size());
//==    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
//==    parameters.fft().transform(1);
//==    parameters.fft().output(&v_r[0]);
//==
//==    for (int ir = 0; ir < parameters.fft().size(); ir++)
//==    {
//==        if (fabs(imag(v_r[ir])) > 1e-14) error_local(__FILE__, __LINE__, "potential is complex");
//==    }
//==
//==    mdarray<complex16, 2> hphi(parameters.num_gvec(), phi.size(1));
//==    mdarray<complex16, 2> res(parameters.num_gvec(), bs);
//==    
//==    int N = n;
//==
//==    apply_h(parameters, N, v_r, &phi(0, 0), &hphi(0, 0));
//==
//==    std::vector<double> eval1;
//==    std::vector<double> eval2;
//==
//==    for (int iter = 0; iter < niter; iter++)
//==    {
//==        eval2 = eval1;
//==        subspace_diag(parameters, N, n, bs, phi, hphi, res, v_pw[0], evec, eval1);
//==        expand_subspace(parameters, N, bs, phi, res);
//==        apply_h(parameters, bs, v_r, &res(0, 0), &hphi(0, N));
//==
//==        if (iter)
//==        {
//==            double diff = 0;
//==            for (int i = 0; i < n; i++) diff += fabs(eval1[i] - eval2[i]);
//==            std::cout << "Eigen-value error : " << diff << std::endl;
//==        }
//==        
//==        N += bs;
//==    }
//==    return N - bs;
//==}

void orthonormalize(mdarray<complex16, 2>& f)
{
    Timer t("orthonormalize");

    std::vector<complex16> v(f.size(0));
    for (int j = 0; j < f.size(1); j++)
    {
        memcpy(&v[0], &f(0, j), f.size(0) * sizeof(complex16));
        for (int j1 = 0; j1 < j; j1++)
        {
            complex16 z(0, 0);
            for (int ig = 0; ig < f.size(0); ig++) z += conj(f(ig, j1)) * v[ig];
            for (int ig = 0; ig < f.size(0); ig++) v[ig] -= z * f(ig, j1);
        }
        double norm = 0;
        for (int ig = 0; ig < f.size(0); ig++) norm += real(conj(v[ig]) * v[ig]);
        for (int ig = 0; ig < f.size(0); ig++) f(ig, j) = v[ig] / sqrt(norm);
    }
}

void check_orth(mdarray<complex16, 2>& f, int num_f)
{
    for (int i = 0; i < num_f; i++)
    {
        for (int j = 0; j < num_f; j++)
        {
            complex16 z(0, 0);
            for (int ig = 0; ig < f.size(0); ig++)
            {
                z += conj(f(ig, i)) * f(ig, j);
            }
            if (i == j) z -= 1.0;
            if (abs(z) > 1e-10)
            {
                std::stringstream s;
                s << "basis is not orthonormal, error : " << abs(z);
                error_local(__FILE__, __LINE__, s);
            }
        }
    }
}

void apply_p(K_point& kp, mdarray<complex16, 2>& r)
{
    for (int i = 0; i < r.size(1); i++)
    {
        // compute kinetic energy of the vector
        double ekin = 0;
        for (int ig = 0; ig < kp.num_gkvec(); ig++) ekin += real(conj(r(ig, i)) * r(ig, i)) * pow(kp.gkvec_cart(ig).length(), 2) / 2.0;

        // apply the preconditioner
        for (int ig = 0; ig < kp.num_gkvec(); ig++)
        {
            double x = pow(kp.gkvec_cart(ig).length(), 2) / 2 / 1.5 / ekin;
            r(ig, i) = r(ig, i) * (27 + 18 * x + 12 * x * x + 8 * x * x * x) / (27 + 18 * x + 12 * x * x + 8 * x * x * x + 16 * x * x * x * x);
        }
    }
}

void diag_lobpcg(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands)
{
    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
    }

    // initial basis functions
    mdarray<complex16, 2> phi(kp.num_gkvec(), num_bands);
    phi.zero();
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0;

    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_bands);

    apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));

    mdarray<complex16, 2> ovlp(3 * num_bands, 3 * num_bands);

    mdarray<complex16, 2> hmlt(3 * num_bands, 3 * num_bands);
    blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());

    std::vector<double> eval(3 * num_bands);
    mdarray<complex16, 2> evec(3 * num_bands, 3 * num_bands);
    
    standard_evp* solver = new standard_evp_lapack();
    solver->solve(num_bands, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
    delete solver;

    mdarray<complex16, 2> zm(kp.num_gkvec(), num_bands);
    blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, num_bands, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), &zm(0, 0), zm.ld());
    zm >> phi;

    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);
    mdarray<complex16, 2> res_active(kp.num_gkvec(), num_bands);
    mdarray<complex16, 2> hres(kp.num_gkvec(), num_bands);

    mdarray<complex16, 2> grad(kp.num_gkvec(), num_bands);
    mdarray<complex16, 2> grad_active(kp.num_gkvec(), num_bands);
    grad.zero();
    mdarray<complex16, 2> hgrad(kp.num_gkvec(), num_bands);
    
    generalized_evp* gevp = new generalized_evp_lapack(-1.0);

    std::vector<bool> converged(num_bands);
    std::vector<int> active_idx(num_bands); 

    for (int k = 1; k < 300; k++)
    {
        apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));
        // res = H|phi> - E|phi>
        for (int i = 0; i < num_bands; i++)
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) 
            {
                //complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - eval[i];
                res(ig, i) = hphi(ig, i) - eval[i] * phi(ig, i);
                
                //if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                //res(ig, i) /= t;
            }
        }

        std::cout << "Iteration : " << k << std::endl;
        int num_res = 0;
        for (int i = 0; i < num_bands; i++)
        {
            double r = 0;
            for (int ig = 0; ig < kp.num_gkvec(); ig++) r += real(conj(res(ig, i)) * res(ig, i));
            converged[i] = (r < 1e-6) ? true : false;
            printf("band : %2i, r : %12.8f, eval : %12.8f, converged : %i\n", i, r, eval[i], (int)converged[i]);

            if (!converged[i])
            {
                memcpy(&res_active(0, num_res), &res(0, i), kp.num_gkvec() * sizeof(complex16));
                memcpy(&grad_active(0, num_res), &grad(0, i), kp.num_gkvec() * sizeof(complex16));
                active_idx[num_res] = i;
                num_res++;
            }
        }
        std::cout << "number of non-converged : " << num_res << std::endl;

        if (num_res == 0) break;

        //apply_p(kp, res);

        //orthonormalize(res);




        apply_h(parameters, kp, num_res, v_r, &res_active(0, 0), &hres(0, 0));

        hmlt.zero();
        ovlp.zero();
        for (int i = 0; i < 3 * num_bands; i++) ovlp(i, i) = complex16(1, 0);

        // <phi|H|phi>
        blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), 
                        &hmlt(0, 0), hmlt.ld());
        // <phi|H|res>
        blas<cpu>::gemm(2, 0, num_bands, num_res, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hres(0, 0), hres.ld(), 
                        &hmlt(0, num_bands), hmlt.ld());
        // <res|H|res>
        blas<cpu>::gemm(2, 0, num_res, num_res, kp.num_gkvec(), &res_active(0, 0), res_active.ld(), &hres(0, 0), hres.ld(), 
                        &hmlt(num_bands, num_bands), hmlt.ld());

        // <phi|res> 
        blas<cpu>::gemm(2, 0, num_bands, num_res, kp.num_gkvec(), &phi(0, 0), phi.ld(), &res_active(0, 0), res_active.ld(), 
                        &ovlp(0, num_bands), ovlp.ld());
        
        // <res|res> 
        blas<cpu>::gemm(2, 0, num_res, num_res, kp.num_gkvec(), &res_active(0, 0), res_active.ld(), &res_active(0, 0), res_active.ld(), 
                        &ovlp(num_bands, num_bands), ovlp.ld());

        if (k > 1)
        {
            //orthonormalize(grad);
            apply_h(parameters, kp, num_res, v_r, &grad_active(0, 0), &hgrad(0, 0));

            // <phi|H|grad>
            blas<cpu>::gemm(2, 0, num_bands, num_res, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hgrad(0, 0), hgrad.ld(), 
                            &hmlt(0, num_bands + num_res), hmlt.ld());
            // <res|H|grad>
            blas<cpu>::gemm(2, 0, num_res, num_res, kp.num_gkvec(), &res_active(0, 0), res_active.ld(), &hgrad(0, 0), hgrad.ld(), 
                            &hmlt(num_bands, num_bands + num_res), hmlt.ld());
            // <grad|H|grad>
            blas<cpu>::gemm(2, 0, num_res, num_res, kp.num_gkvec(), &grad_active(0, 0), grad_active.ld(), &hgrad(0, 0), hgrad.ld(), 
                            &hmlt(num_bands + num_res, num_bands + num_res), hmlt.ld());
            
            // <phi|grad> 
            blas<cpu>::gemm(2, 0, num_bands, num_res, kp.num_gkvec(), &phi(0, 0), phi.ld(), &grad_active(0, 0), grad_active.ld(), 
                            &ovlp(0, num_bands + num_res), ovlp.ld());
            // <res|grad> 
            blas<cpu>::gemm(2, 0, num_res, num_res, kp.num_gkvec(), &res_active(0, 0), res_active.ld(), &grad_active(0, 0), grad_active.ld(), 
                            &ovlp(num_bands, num_bands + num_res), ovlp.ld());
            // <grad|grad> 
            blas<cpu>::gemm(2, 0, num_res, num_res, kp.num_gkvec(), &grad_active(0, 0), grad_active.ld(), &grad_active(0, 0), grad_active.ld(), 
                            &ovlp(num_bands + num_res, num_bands + num_res), ovlp.ld());
        }

        //for (int i = 0; i < num_bands; i++)
        //{
        //    if (converged[i])
        //    {
        //        for (int k = 0; k < 3 * num_bands; k++)
        //        {
        //            if (k != i)
        //            {
        //                hmlt(k, i) = hmlt(i, k) = 0;
        //                //ovlp(k, i) = ovlp(i, k) = 0;
        //            }
        //            else
        //            {
        //                hmlt(i, i) = eval[i];
        //                //ovlp(i, i) = 1.0;
        //            }
        //        }
        //    }
        //}




        if (k == 1)
        {
            gevp->solve(num_bands + num_res, num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), 
                        &eval[0], evec.get_ptr(), evec.ld());
        }
        else
        {
            gevp->solve(num_bands + 2 * num_res, num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), 
                        &eval[0], evec.get_ptr(), evec.ld());
        }

        
        memcpy(&zm(0, 0), &grad_active(0, 0), num_res * kp.num_gkvec() * sizeof(complex16));
        // P^{k+1} = P^{k} * Z_{grad} + res^{k} * Z_{res}
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_res, num_res, &res_active(0, 0), res_active.ld(), 
                        &evec(num_bands, 0), evec.ld(), &grad_active(0, 0), grad_active.ld());
        if (k > 1) 
        {
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_res, num_res, complex16(1, 0), &zm(0, 0), zm.ld(), 
                            &evec(num_bands + num_res, 0), evec.ld(), complex16(1, 0), &grad_active(0, 0), grad_active.ld());
        }




        // phi^{k+1} = phi^{k} * Z_{phi} + P^{k+1}
        phi >> zm;

        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, num_bands, &zm(0, 0), zm.ld(), &evec(0, 0), evec.ld(), 
                        &phi(0, 0), phi.ld());

        //for (int i = 0; i < num_bands; i++)
        //{
        //    if (converged[i])
        //    {
        //        memcpy(&phi(0, i), &zm(0, i), kp.num_gkvec() * sizeof(complex16));
        //    }
        //}

        for (int j = 0; j < num_res; j++)
        {
            int i = active_idx[j];
            memcpy(&grad(0, i), &grad_active(0, j), kp.num_gkvec() * sizeof(complex16));
            for (int ig = 0; ig < kp.num_gkvec(); ig++) phi(ig, i) += grad_active(ig, j);
        }

            

        check_orth(phi, num_bands);    
    }
    

    delete gevp;

}

void expand_subspace(K_point& kp, int N, int num_bands, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& res)
{
    Timer t("expand_subspace");

    // overlap between new addisional basis vectors and old basis vectors
    mdarray<complex16, 2> ovlp(N, num_bands);
    ovlp.zero();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < num_bands; j++) 
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) ovlp(i, j) += conj(phi(ig, i)) * res(ig, j);
        }
    }

    // project out the the old subspace
    for (int j = 0; j < num_bands; j++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, j) -= ovlp(i, j) * phi(ig, i);
        }
    }
    orthonormalize(res);

    for (int j = 0; j < num_bands; j++)
    {
        for (int ig = 0; ig < kp.num_gkvec(); ig++) phi(ig, N + j) = res(ig, j);
    }
}

void diag_davidson(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands)
{
    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
    }

    int num_iter = 5;

    int num_big_iter = 5;

    // initial basis functions
    mdarray<complex16, 2> phi(kp.num_gkvec(), num_bands * num_iter);
    phi.zero();
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0;

    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_bands * num_iter);


    mdarray<complex16, 2> hmlt(num_bands * num_iter, num_bands * num_iter);
    mdarray<complex16, 2> evec(num_bands * num_iter, num_bands * num_iter);
    std::vector<double> eval(num_bands * num_iter);
    
    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);

    standard_evp* solver = new standard_evp_lapack();

    for (int l = 0; l < num_big_iter; l++)
    {
        for (int k = 1; k <= num_iter; k++)
        {
            int N = k * num_bands;

            apply_h(parameters, kp, num_bands, v_r, &phi(0, (k - 1) * num_bands), &hphi(0, (k - 1) * num_bands));

            blas<cpu>::gemm(2, 0, N, N, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());

            solver->solve(N, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
            
            // compute residuals
            res.zero();
            for (int j = 0; j < num_bands; j++)
            {
                for (int mu = 0; mu < N; mu++)
                {
                    for (int ig = 0; ig < kp.num_gkvec(); ig++)
                    {
                        res(ig, j) += (evec(mu, j) * hphi(ig, mu) - eval[j] * evec(mu, j) * phi(ig, mu));
                    }
                }
            }

            std::cout << "Iteration : " << k << std::endl;
            for (int i = 0; i < num_bands; i++)
            {
                double r = 0;
                for (int ig = 0; ig < kp.num_gkvec(); ig++) r += real(conj(res(ig, i)) * res(ig, i));
                //for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, i) /= sqrt(r);
                std::cout << "band : " << i << " residiual : " << r << " eigen-value : " << eval[i] << std::endl;
            }

            //apply_p(kp, res);
            
            for (int j = 0; j < num_bands; j++)
            {
                for (int ig = 0; ig < kp.num_gkvec(); ig++)
                {
                    complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - eval[j];
                    if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                    res(ig, j) /= t;
                }
            }

            if (k < num_iter) expand_subspace(kp, N, num_bands, phi, res);
        }

        mdarray<complex16, 2> zm(kp.num_gkvec(), num_bands);
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, num_bands * num_iter, &phi(0, 0), phi.ld(), 
                        &evec(0, 0), evec.ld(), &zm(0, 0), zm.ld());
        //check_orth(zm);
        memcpy(phi.get_ptr(), zm.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));
    }

    delete solver;
}

void expand_subspace_v2(K_point& kp, int N, int num_bands, mdarray<complex16, 2>& phi, complex16* res__)
{
    mdarray<complex16, 2> res(res__, kp.num_gkvec(), num_bands);

    // overlap between new addisional basis vectors and old basis vectors
    mdarray<complex16, 2> ovlp(N, num_bands);
    ovlp.zero();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < num_bands; j++) 
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) ovlp(i, j) += conj(phi(ig, i)) * res(ig, j);
        }
    }

    // project out the the old subspace
    for (int j = 0; j < num_bands; j++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, j) -= ovlp(i, j) * phi(ig, i);
        }
    }
    orthonormalize(res);

    for (int j = 0; j < num_bands; j++)
    {
        for (int ig = 0; ig < kp.num_gkvec(); ig++) phi(ig, N + j) = res(ig, j);
    }
}

void diag_davidson_v2(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands)
{
    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
    }

    int num_big_iter = 5;

    // initial basis functions
    mdarray<complex16, 2> phi(kp.num_gkvec(), num_bands * 4);
    phi.zero();
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0;

    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_bands * 4);


    mdarray<complex16, 2> hmlt(num_bands * 4, num_bands * 4);
    mdarray<complex16, 2> evec(num_bands * 4, num_bands * 4);
    std::vector<double> eval(num_bands * 4);
    
    mdarray<complex16, 2> res(kp.num_gkvec(), 2 * num_bands);

    standard_evp* solver = new standard_evp_lapack();

    for (int l = 0; l < num_big_iter; l++)
    {
        
        // 1x1 evp
        apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));

        blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());

        solver->solve(num_bands, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
            
        // compute residuals
        res.zero();
        for (int j = 0; j < num_bands; j++)
        {
            for (int mu = 0; mu < num_bands; mu++)
            {
                for (int ig = 0; ig < kp.num_gkvec(); ig++)
                {
                    res(ig, j) += (evec(mu, j) * hphi(ig, mu) - eval[j] * evec(mu, j) * phi(ig, mu));
                }
            }
        }
        
        std::cout << "Iteration : " << l << std::endl;
        for (int i = 0; i < num_bands; i++)
        {
            double r = 0;
            for (int ig = 0; ig < kp.num_gkvec(); ig++) r += real(conj(res(ig, i)) * res(ig, i));
            //for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, i) /= sqrt(r);
            std::cout << "band : " << i << " residiual : " << r << " eigen-value : " << eval[i] << std::endl;
        }

        expand_subspace_v2(kp, num_bands, num_bands, phi, &res(0, 0));
        check_orth(phi, 2 * num_bands);

        // 2x2 evp
        apply_h(parameters, kp, num_bands, v_r, &phi(0, num_bands), &hphi(0, num_bands));

        blas<cpu>::gemm(2, 0, 2 * num_bands, 2 * num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());

        solver->solve(2 * num_bands, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
            
        // compute residuals
        res.zero();
        for (int j = 0; j < 2 * num_bands; j++)
        {
            for (int mu = 0; mu < 2 * num_bands; mu++)
            {
                for (int ig = 0; ig < kp.num_gkvec(); ig++)
                {
                    res(ig, j) += (evec(mu, j) * hphi(ig, mu) - eval[j] * evec(mu, j) * phi(ig, mu));
                }
            }
        }
        for (int i = 0; i < 2 * num_bands; i++)
        {
            double r = 0;
            for (int ig = 0; ig < kp.num_gkvec(); ig++) r += real(conj(res(ig, i)) * res(ig, i));
            //for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, i) /= sqrt(r);
        }
        expand_subspace_v2(kp, 2 * num_bands, 2 * num_bands, phi, &res(0, 0));
        check_orth(phi, 4 * num_bands);


        // 4x4 evp
        apply_h(parameters, kp, 2 * num_bands, v_r, &phi(0, 2 * num_bands), &hphi(0, 2 * num_bands));

        blas<cpu>::gemm(2, 0, 4 * num_bands, 4 * num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());

        solver->solve(4 * num_bands, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
            
        
        mdarray<complex16, 2> zm(kp.num_gkvec(), num_bands);
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, num_bands * 4, &phi(0, 0), phi.ld(), 
                        &evec(0, 0), evec.ld(), &zm(0, 0), zm.ld());

        check_orth(zm, num_bands);
        memcpy(phi.get_ptr(), zm.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));

            //apply_p(kp, res);
            
            //for (int j = 0; j < num_bands; j++)
            //{
            //    for (int ig = 0; ig < kp.num_gkvec(); ig++)
            //    {
            //        complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - eval[j];
            //        if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
            //        res(ig, j) /= t;
            //    }
            //}

    }

    delete solver;
}

void diag_davidson_v3(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands)
{
    Timer t("diag_davidson");

    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
    }

    int num_phi = num_bands * 5;

    int num_iter = 100;

    mdarray<complex16, 2> phi(kp.num_gkvec(), num_phi);
    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_phi);
    
    // initial basis functions
    phi.zero();
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0;
    apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));


    mdarray<complex16, 2> hmlt(num_phi, num_phi);
    mdarray<complex16, 2> evec(num_phi, num_phi);
    std::vector<double> eval(num_phi);
    
    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);

    mdarray<complex16, 2> res_active(kp.num_gkvec(), num_bands);

    standard_evp* solver = new standard_evp_lapack();

    int N = num_bands; // intial eigen-value problem

    for (int k = 0; k < num_iter; k++)
    {
        blas<cpu>::gemm(2, 0, N, N, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());

        solver->solve(N, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
        
        // compute residuals
        res.zero();
        for (int j = 0; j < num_bands; j++)
        {
            for (int mu = 0; mu < N; mu++)
            {
                for (int ig = 0; ig < kp.num_gkvec(); ig++)
                {
                    res(ig, j) += (evec(mu, j) * hphi(ig, mu) - eval[j] * evec(mu, j) * phi(ig, mu));
                }
            }
        }

        int n = 0;
        std::cout << "Iteration : " << k << ", subspace size : " << N << std::endl;
        for (int i = 0; i < num_bands; i++)
        {
            double r = 0;
            for (int ig = 0; ig < kp.num_gkvec(); ig++) r += real(conj(res(ig, i)) * res(ig, i));
            if (r > 1e-5) 
            {
                memcpy(&res_active(0, n), &res(0, i), kp.num_gkvec() * sizeof(complex16));

                for (int ig = 0; ig < kp.num_gkvec(); ig++)
                {
                    complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - eval[i];
                    if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                    res_active(ig, n) /= t;
                }

                n++;
            }
            std::cout << "band : " << i << " residiual : " << r << " eigen-value : " << eval[i] << std::endl;
        }
        std::cout << "number of non-converged eigen-vectors : " << n << std::endl;
        if (n == 0) break;

        if (N + n > num_phi)
        {
            mdarray<complex16, 2> zm(kp.num_gkvec(), num_bands);
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), 
                            &evec(0, 0), evec.ld(), &zm(0, 0), zm.ld());
            //check_orth(zm);
            memcpy(phi.get_ptr(), zm.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));
            apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));
            N = num_bands;
        }
        
        //apply_p(kp, res_active);


        expand_subspace(kp, N, n, phi, res_active);
        apply_h(parameters, kp, n, v_r, &phi(0, N), &hphi(0, N));

        N += n;


        
        //for (int j = 0; j < num_bands; j++)
        //{
        //    for (int ig = 0; ig < kp.num_gkvec(); ig++)
        //    {
        //        complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - eval[j];
        //        if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
        //        res(ig, j) /= t;
        //    }
        //}

    }

    delete solver;
}



void test_lobpcg()
{
    Global parameters;

    double a0[] = {12.975 * 1.889726125, 0, 0};
    double a1[] = {0, 12.975 * 1.889726125, 0};
    double a2[] = {0, 0, 12.975 * 1.889726125};

    double Ekin = 3.0; // 40 Ry = 20 Ha

    parameters.set_lattice_vectors(a0, a1, a2);
    parameters.set_pw_cutoff(2 * sqrt(2 * Ekin) + 0.5);
    parameters.initialize();
    parameters.print_info();
    
    double vk[] = {0, 0, 0};
    K_point kp(parameters, vk, 1.0);
    kp.generate_gkvec(sqrt(2 * Ekin));

    std::cout << "num_gkvec = " << kp.num_gkvec() << std::endl;

    // generate some potential in plane-wave domain
    std::vector<complex16> v_pw(parameters.num_gvec());
    for (int ig = 0; ig < parameters.num_gvec(); ig++) v_pw[ig] = complex16(1.0 / pow(parameters.gvec_len(ig) + 1.0, 1), 0.0);

    //== // cook the Hamiltonian
    //== mdarray<complex16, 2> hmlt(kp.num_gkvec(), kp.num_gkvec());
    //== hmlt.zero();
    //== for (int ig1 = 0; ig1 < kp.num_gkvec(); ig1++)
    //== {
    //==     for (int ig2 = 0; ig2 < kp.num_gkvec(); ig2++)
    //==     {
    //==         int ig = parameters.index_g12(kp.gvec_index(ig2), kp.gvec_index(ig1));
    //==         hmlt(ig2, ig1) = v_pw[ig];
    //==         if (ig1 == ig2) hmlt(ig2, ig1) += pow(kp.gkvec_cart(ig1).length(), 2) / 2.0;
    //==     }
    //== }

    //== standard_evp* solver = new standard_evp_lapack();

    //== std::vector<double> eval(kp.num_gkvec());
    //== mdarray<complex16, 2> evec(kp.num_gkvec(), kp.num_gkvec());

    //== solver->solve(kp.num_gkvec(), hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());

    //== delete solver;
    
    int num_bands = 20;

    //== printf("\n");
    //== printf("Lowest eigen-values (exact): \n");
    //== for (int i = 0; i < num_bands; i++)
    //== {
    //==     printf("i : %i,  eval : %16.10f\n", i, eval[i]);
    //== }


    //diag_lobpcg(parameters, kp, v_pw, num_bands);

    diag_davidson_v3(parameters, kp, v_pw, num_bands);

    
    parameters.clear();
}

int main(int argn, char** argv)
{
    Platform::initialize(true);

    test_lobpcg();

    Timer::print();

    Platform::finalize();
}
