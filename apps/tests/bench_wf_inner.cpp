#include <sirius.h>
#include <chrono>
#include <ctime>    

using namespace sirius;

void test_wf_inner(std::string title, std::string motivation, int num_iters, std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__, int bs__,
                   linalg_t la__, memory_t mem_bra__, memory_t mem_ket__, memory_t mem_o__)
{
    std::unique_ptr<BLACS_grid> blacs_grid;
    if (mpi_grid_dims__[0] * mpi_grid_dims__[1] == 1) {
        blacs_grid =
            std::unique_ptr<BLACS_grid>(new BLACS_grid(Communicator::self(), mpi_grid_dims__[0], mpi_grid_dims__[1]));
    } else {
        blacs_grid =
            std::unique_ptr<BLACS_grid>(new BLACS_grid(Communicator::world(), mpi_grid_dims__[0], mpi_grid_dims__[1]));
    }

    // Defines the unit block of the grid
    //
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    // The cutoff selects a sphere of unit blocks on the grid
    //

    /* create G-vectors */
    Gvec gvec(M, cutoff__, Communicator::world(), false);

    Gvec_partition gvp(gvec, Communicator::world(), Communicator::self());

    int nsp{1};
    Wave_functions bra(gvp, 2 * num_bands__, mem_bra__, nsp);

    for (int is = 0; is < nsp; is++) {
        for (int i = 0; i < 2 * num_bands__; i++) {
            for (int igloc = 0; igloc < gvec.count(); igloc++) {
                bra.pw_coeffs(is).prime(igloc, i) = utils::random<double_complex>();
            }
        }
    }

    if (is_device_memory(mem_bra__)) {
        for (int ispn = 0; ispn < nsp; ispn++) {
            bra.allocate(spin_range(ispn), mem_bra__);
            bra.copy_to(spin_range(ispn), mem_bra__, 0, 2 * num_bands__);
        }
    }

    Wave_functions ket(gvp, 2 * num_bands__, mem_ket__, nsp);
    if (is_device_memory(mem_ket__)) {
        for (int ispn = 0; ispn < nsp; ispn++) {
            ket.allocate(spin_range(ispn), mem_ket__);
        }
    }
    for (int ispn = 0; ispn < nsp; ispn++) {
        ket.copy_from(bra, 2 * num_bands__, ispn, 0, ispn, 0);
    }

    dmatrix<double_complex> ovlp(2 * num_bands__, 2 * num_bands__, *blacs_grid, bs__, bs__);

    if (is_device_memory(mem_o__)) {
        ovlp.allocate(mem_o__);
    }
    ovlp.zero();

    using clock_t   = std::chrono::high_resolution_clock;
    using seconds_t = std::chrono::duration<double>;

    std::cout << "timestamp,title,motivation,m,n,k,P,cutoff,bands,time [s]\n";
    for (int i = 0; i <= num_iters; ++i) {
        Communicator::world().barrier();
        auto t_start = clock_t::now();

        sddk::inner(mem_o__, la__, 0, bra, 0, num_bands__, ket, 0, num_bands__, ovlp, 0, 0);
        sddk::inner(mem_o__, la__, 0, bra, 0, num_bands__, ket, num_bands__, num_bands__, ovlp, 0, num_bands__);
        sddk::inner(mem_o__, la__, 0, bra, num_bands__, num_bands__, ket, 0, num_bands__, ovlp, num_bands__, 0);
        sddk::inner(mem_o__, la__, 0, bra, num_bands__, num_bands__, ket, num_bands__, num_bands__, ovlp, num_bands__,
                    num_bands__);

        Communicator::world().barrier();
        auto t_end = clock_t::now();

        // Skip warm-up run
        //
        if (Communicator::world().rank() == 0 && i != 0) {
            auto t_run = seconds_t(t_end - t_start).count();

            // Timestamp
            //
            auto now_time = std::chrono::system_clock::to_time_t(t_end);
            auto gmt_time = gmtime(&now_time);
            auto timestamp = std::put_time(gmt_time, "%Y-%m-%d %H:%M:%S");

            int m = 2 * num_bands__;
            int n = m;
            int k = gvec.num_gvec();
            int P = Communicator::world().size();

            std::cout << timestamp << ","
                      << title << "," 
                      << motivation << ","
                      << m << ","
                      << n << ","
                      << k << ","
                      << P << ","
                      << cutoff__ << ","
                      << t_run << ","
                      << num_bands__ << "\n";
        }
    }
}

// The benchmark is adapted from test_wf_inner_v4.
//
int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--title=", "{string} the benchmark title");
    args.register_key("--motivation=", "{string} background information about the benchmark to record motivation");

    args.register_key("--num_iters=", "{int} number of profiling iterations");

    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--num_bands=", "{int} number of bands");

    args.register_key("--bs=", "{int} block size");
    args.register_key("--linalg_t=", "{string} type of the linear algebra driver");
    args.register_key("--mem_bra=", "{string} memory type of the <bra| states");
    args.register_key("--mem_ket=", "{string} memory type of the |ket> states");
    args.register_key("--mem_o=", "{string} memory type of the resulting overlap matrix");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto title         = args.value<std::string>("title", "N/A");
    auto motivation    = args.value<std::string>("motivation", "N/A");

    auto num_iters     = args.value<int>("num_iters", 4);
    auto mpi_grid_dims = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});
    auto cutoff        = args.value<double>("cutoff", 8.0);
    auto bs            = args.value<int>("bs", 32);
    auto num_bands     = args.value<int>("num_bands", 100);
    auto la            = get_linalg_t(args.value<std::string>("linalg_t", "blas"));
    auto mem_bra       = get_memory_t(args.value<std::string>("mem_bra", "host"));
    auto mem_ket       = get_memory_t(args.value<std::string>("mem_ket", "host"));
    auto mem_o         = get_memory_t(args.value<std::string>("mem_o", "host"));

    sirius::initialize(1);

    test_wf_inner(title, motivation, num_iters, mpi_grid_dims, cutoff, num_bands, bs, la, mem_bra, mem_ket, mem_o);

    Communicator::world().barrier();
    if (Communicator::world().rank() == 0) {
        utils::timer::print();
    }
    // reset_device__ = false;
    sirius::finalize();
}
