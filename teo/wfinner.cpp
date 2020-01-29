#include <sirius.h>

#include <cmath>

void test_wf_inner(std::vector<int> pgrid, double cutoff, int m_bands, int n_bands)
{

    using namespace sirius;

    constexpr int bs  = 32; // block size
    auto la           = get_linalg_t("blas");
    auto mem_bra      = get_memory_t("host");
    auto mem_ket      = get_memory_t("host");
    auto mem_o        = get_memory_t("host");
    constexpr int nsp = 1; // num spins

    int num_procs      = pgrid[0] * pgrid[1];
    int rank           = Communicator::world().rank();
    auto blacs_grid    = std::make_unique<BLACS_grid>((num_procs == 1) ? Communicator::self() : Communicator::world(),
                                                   pgrid[0], pgrid[1]);
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Gvec gvec(M, cutoff, Communicator::world(), false);
    Gvec_partition gvp(gvec, Communicator::world(), Communicator::self());

    if (rank == 0) {
        printf("mnk = %d %d %d\n", m_bands, n_bands, gvec.num_gvec());
        printf("pgrid = %d %d\n", pgrid[0], pgrid[1]);
    }
    // printf("local k: %i\n", gvec.count());

    // init wave functions: bra -> A / ket -> B
    Wave_functions bra(gvp, m_bands, mem_bra, nsp);
    Wave_functions ket(gvp, n_bands, mem_ket, nsp);

    for (int is = 0; is < nsp; is++) {
        for (int i = 0; i < m_bands; i++) {
            for (int igloc = 0; igloc < gvec.count(); igloc++) {
                bra.pw_coeffs(is).prime(igloc, i) = utils::random<double_complex>();
            }
        }
        for (int i = 0; i < n_bands; i++) {
            for (int igloc = 0; igloc < gvec.count(); igloc++) {
                ket.pw_coeffs(is).prime(igloc, i) = utils::random<double_complex>();
            }
        }
    }

    // result matrix: ovlp -> C
    dmatrix<double_complex> ovlp(m_bands, n_bands, *blacs_grid, bs, bs);

    using clock_t           = std::chrono::high_resolution_clock;
    using seconds_t         = std::chrono::duration<double>;
    constexpr int num_iters = 5;
    for (int i = 0; i < num_iters; ++i) {
        ovlp.zero(); // reset

        auto t0 = clock_t::now();
        sddk::inner(mem_o, la, 0, bra, 0, m_bands, ket, 0, n_bands, ovlp, 0, 0);
        auto t1 = clock_t::now();

        if (rank == 0) {
            printf("%d: t [s] = %.5f\n", i, seconds_t(t1 - t0).count());
        }
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mnk=", "{int int int} dimensions of MPI grid");
    args.register_key("--pgrid=", "{int int} dimensions of MPI grid");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    constexpr double sphere_factor = 4.18879;
    auto mnk                       = args.value<std::vector<int>>("mnk", {200, 200, 50000});
    auto pgrid                     = args.value<std::vector<int>>("pgrid", {1, 1});
    int m_bands                    = mnk[0];
    int n_bands                    = mnk[1];
    double cutoff                  = std::pow(mnk[2] / sphere_factor, 1.0 / 3.0);

    sirius::initialize(1);

    test_wf_inner(pgrid, cutoff, m_bands, n_bands);

    Communicator::world().barrier();
    if (Communicator::world().rank() == 0) {
        utils::timer::print();
    }
    sirius::finalize();
}
