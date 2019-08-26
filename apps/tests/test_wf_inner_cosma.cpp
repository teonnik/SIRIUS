#include <sirius.h>

#include <cosma/multiply.hpp>
#include <grid2grid/transform.hpp>

using zdouble_t = std::complex<double>;

extern template struct grid2grid::block<zdouble_t>;
extern template void cosma::multiply_using_layout<zdouble_t>(grid2grid::grid_layout<zdouble_t>& A_layout,
                                                             grid2grid::grid_layout<zdouble_t>& B_layout,
                                                             grid2grid::grid_layout<zdouble_t>& C_layout, int m, int n,
                                                             int k, zdouble_t alpha, zdouble_t beta, char trans_A,
                                                             char trans_B, MPI_Comm comm);

namespace teo {

// The number of wave functions (num_wfs / `m` or `n`) is much smaller than the dimensionality of the space (`k`).
//
// Row / column indexing starts from zero. The end is not included.
//
grid2grid::grid_layout<zdouble_t> get_wf_grid_layout(Wave_functions& phi, int i_spin, int index_of_start_wf,
                                                     int num_wfs)
{
    using namespace grid2grid;
    assert(i_spin == 0 || i_spin == 1);

    int this_rank         = phi.comm().rank();
    int num_procs         = phi.comm().size();
    int num_basis_vectors = phi.gkvec().num_gvec();

    // The rows are split in slabs starting from process 0 in ascending order. The slabs may be of differnt size.
    //
    std::vector<int> rows_split;
    rows_split.reserve(num_procs + 1);
    for (int rank = 0; rank < num_procs; ++rank) {
        rows_split.push_back(phi.gkvec().gvec_offset(rank));
    }
    rows_split.push_back(num_basis_vectors);

    // Columns are not split.
    //
    std::vector<int> cols_split{0, num_wfs};

    // Create a map between ranks and grid.
    //
    std::vector<std::vector<int>> owners(num_procs, std::vector<int>(1));
    for (int rank = 0; rank < num_procs; ++rank) {
        owners[rank][0] = rank;
    }

    // Initialize a local 2D view of data.
    //
    // Note: In SIRIUS' case there is only one local block.
    //
    zdouble_t* data_ptr = phi.pw_coeffs(i_spin).prime().at(phi.preferred_memory_t(), 0, index_of_start_wf);
    block_coordinates coords{this_rank, 0};
    interval r_interval{rows_split[this_rank], rows_split[this_rank + 1]};
    interval c_interval{0, num_wfs};
    block<zdouble_t> loc_blk{r_interval, c_interval, coords, data_ptr};

    // Create a grid of blocks where each block belongs to a process.
    //
    grid2D grid{std::move(rows_split), std::move(cols_split)};
    assigned_grid2D a_grid{std::move(grid), std::move(owners), num_procs};
    std::vector<block<zdouble_t>> loc_blocks{loc_blk};
    local_blocks<zdouble_t> local_memory(std::move(loc_blocks));
    return grid2grid::grid_layout<zdouble_t>(std::move(a_grid), std::move(local_memory));
}

// `dmatrix` inherits from a column-major 2D mdarray. It also holds an object of the BLACS_grid clss which is built on
// the MPI_grid class. MPI_grid uses the default MPI row-major rank ordering.
//
grid2grid::grid_layout<zdouble_t> get_dmatrix_grid_layout(dmatrix<zdouble_t>& result, int irow0__, int jcol0__,
                                                          int submatrix_row_size, int submatrix_col_size)
{
    using namespace grid2grid::scalapack;

    int this_rank   = result.comm().rank();
    char transpose  = 'N';
    zdouble_t* data = result.at(sirius::memory_t::host);

    matrix_dim m_dim{result.num_rows(), result.num_cols()}; // global matrix size
    block_dim b_dim{result.bs_row(), result.bs_col()};      // block dimension

    elem_grid_coord ij{irow0__ + 1, jcol0__ + 1};                // start of submatrix
    matrix_dim subm_dim{submatrix_row_size, submatrix_col_size}; // dim of submatrix

    // Note: Using BLACS routines directly might not be the best way of retrieving relevant information. SIRIUS can work
    // without the ScaLAPACK back-end.
    //
    int const* descr = result.descriptor();
    int blacs_ctx    = descr[1];
    int lld          = descr[8];                  // local leading dimension
    rank_grid_coord rank_src{descr[6], descr[7]}; // rank src

    int rank_grid_rows;
    int rank_grid_cols;
    int this_rank_row;
    int this_rank_col;
    Cblacs_gridinfo(blacs_ctx, &rank_grid_rows, &rank_grid_cols, &this_rank_row, &this_rank_col);

    rank_decomposition r_grid{rank_grid_rows, rank_grid_cols};

    // From `blacs_grid.hpp` it appears that the ordering is always column-major.
    // 
    // SIRIUS enforces that BLACS and MPI grids have the same ordering. 
    //
    ordering rank_grid_ordering{ordering::row_major};

    return grid2grid::get_scalapack_grid(lld, m_dim, ij, subm_dim, b_dim, r_grid, rank_grid_ordering, transpose,
                                         rank_src, data, this_rank);
}

// Succeeds if `bra`, `ket` and `result` are at least MPI_CONGRUENT. This guarantees that they have identical
// MPI_Group's even if their MPI contexts differ.
//
// Note that MPI_SIMILAR requires remapping process ranks between communicators and is currently not supported.
//
void assert_communicators_compatibility(Wave_functions const& bra, Wave_functions const& ket,
                                        dmatrix<zdouble_t> const& result)
{
    int bra_ket_comp_bit;
    MPI_Comm_compare(bra.comm().mpi_comm(), ket.comm().mpi_comm(), &bra_ket_comp_bit);
    if (!(bra_ket_comp_bit == MPI_IDENT || bra_ket_comp_bit == MPI_CONGRUENT)) {
        std::cout << "[ERROR] inner(): `bra` and `ket` have incompatible communicators!\n";
        std::terminate();
    }

    int bra_result_comp_bit;
    MPI_Comm_compare(bra.comm().mpi_comm(), result.comm().mpi_comm(), &bra_result_comp_bit);
    if (!(bra_result_comp_bit == MPI_IDENT || bra_result_comp_bit == MPI_CONGRUENT)) {
        std::cout << "[ERROR] inner(): `bra` and `result` have incompatible communicators!\n";
        std::terminate();
    }
}

// There are `m` wavefunctions starting from index `i0` and `n` wavefunctions starting from index `j0`. Only portion
// of the resulting `dmatrix` is updated at every call.
//
// A wave function has 2 components (plane wave and mt??) and 2 spins. A matrix multiplication has to be performed
// for all of them. The result of all 4 multiplications is accumulated into `dmatrix`.
//
// The inner product only calculates a fraction of the complete distributed matrix. The resulting matrix is assumed
// to have a 2D block-cyclic layout. The matrix is Hermitian (???) and only pieces of it are updated at a time. It
// has to be in a block-cyclic layout as they are other ScaLAPACK operations performed on it (diagonalization).
//
// Note: ispn: 0 - gemm `0`-th cpt, 1 - gemm `1`-th cpt, 2 - gemm both cpt.
//
// Note: the resulting matrix may be double.
//
// Note: The `k` parameter is not dependent on the spin. (from discussion with Anton on slack)
//
void inner(sirius::memory_t mem, sirius::linalg_t la, int spin_param, sirius::Wave_functions& bra, int bra_index, int m,
           sirius::Wave_functions& ket, int ket_index, int n, sirius::dmatrix<zdouble_t>& result, int irow0, int jcol0)
{
    // assert_communicators_compatibility(bra, ket, result);
    MPI_Comm const comm   = bra.comm().mpi_comm();
    char trans_A          = 'C';
    char trans_B          = 'N';
    zdouble_t const alpha = 1;
    zdouble_t beta        = 0;
    int k                 = bra.gkvec().num_gvec();

    for (int i_spin : get_spins(spin_param)) {
        // The layouts for both spin components are equivalent, but the data pointers are different, hence we can't
        // refactor the following out of the loop (atm).
        //
        grid2grid::grid_layout<zdouble_t> A_layout = teo::get_wf_grid_layout(bra, i_spin, bra_index, m);
        grid2grid::grid_layout<zdouble_t> B_layout = teo::get_wf_grid_layout(ket, i_spin, ket_index, n);
        grid2grid::grid_layout<zdouble_t> C_layout = teo::get_dmatrix_grid_layout(result, irow0, jcol0, m, n);
        // A_layout.transpose_or_conjugate(trans_A);

        cosma::multiply_using_layout(A_layout, B_layout, C_layout, m, n, k, alpha, beta, trans_A, trans_B, comm);
        beta = 1;

        // TODO: mt coeffs
    }
}

} // namespace teo

// The following is identical to test_wf_inner_v4.cpp except for accepting command-line arguments.
//
int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--bs=", "{int} block size");
    args.register_key("--num_bands=", "{int} number of bands");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    auto mpi_grid_dims = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});
    auto cutoff        = args.value<double>("cutoff", 8.0);
    auto bs            = args.value<int>("bs", 32);
    auto num_bands     = args.value<int>("num_bands", 100);

    sddk::linalg_t la      = get_linalg_t("blas");
    sddk::memory_t mem_bra = get_memory_t("host");
    sddk::memory_t mem_ket = get_memory_t("host");
    sddk::memory_t mem_o   = get_memory_t("host");

    sirius::initialize();

    std::unique_ptr<BLACS_grid> blacs_grid;
    if (mpi_grid_dims[0] * mpi_grid_dims[1] == 1) {
        blacs_grid =
            std::unique_ptr<BLACS_grid>(new BLACS_grid(Communicator::self(), mpi_grid_dims[0], mpi_grid_dims[1]));
    } else {
        blacs_grid =
            std::unique_ptr<BLACS_grid>(new BLACS_grid(Communicator::world(), mpi_grid_dims[0], mpi_grid_dims[1]));
    }

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    Gvec gvec(M, cutoff, Communicator::world(), false);
    Gvec_partition gvp(gvec, Communicator::world(), Communicator::self());

    int num_spins{1};
    Wave_functions bra(gvp, 2 * num_bands, mem_bra, num_spins); // A

    for (int i_spin = 0; i_spin < num_spins; i_spin++) { // spins
        auto& pw = bra.pw_coeffs(i_spin);
        for (int i_band = 0; i_band < 2 * num_bands; i_band++) {
            for (int i_gloc = 0; i_gloc < gvec.count(); i_gloc++) {
                pw.prime(i_gloc, i_band) = utils::random<double_complex>();
            }
        }
    }

    Wave_functions ket(gvp, 2 * num_bands, mem_ket, num_spins); // B

    for (int i_spin = 0; i_spin < num_spins; i_spin++) {
        ket.copy_from(bra, 2 * num_bands, num_spins, 0, num_spins, 0);
    }

    dmatrix<double_complex> result(2 * num_bands, 2 * num_bands, *blacs_grid, bs, bs);

    teo::inner(mem_o, la, 0, bra, 0, num_bands, ket, 0, num_bands, result, 0, 0);
    teo::inner(mem_o, la, 0, bra, 0, num_bands, ket, num_bands, num_bands, result, 0, num_bands);
    teo::inner(mem_o, la, 0, bra, num_bands, num_bands, ket, 0, num_bands, result, num_bands, 0);
    teo::inner(mem_o, la, 0, bra, num_bands, num_bands, ket, num_bands, num_bands, result, num_bands, num_bands);

    //    sddk::inner(mem_o, la, 0, bra, 0, num_bands, ket, 0, num_bands, result, 0, 0);
    //    sddk::inner(mem_o, la, 0, bra, 0, num_bands, ket, num_bands, num_bands, result, 0, num_bands);
    //    sddk::inner(mem_o, la, 0, bra, num_bands, num_bands, ket, 0, num_bands, result, num_bands, 0);
    //    sddk::inner(mem_o, la, 0, bra, num_bands, num_bands, ket, num_bands, num_bands, result, num_bands, num_bands);

    // Check if results make sense
    //
    auto max_diff = check_hermitian(result, 2 * num_bands);
    if (Communicator::world().rank() == 0) {
        printf("maximum difference: %18.12f\n", max_diff);
        if (max_diff > 1e-12) {
            printf("!!!! FAIL\n");
        } else {
            printf("OK\n");
        }
    }
    sirius::finalize();
}
