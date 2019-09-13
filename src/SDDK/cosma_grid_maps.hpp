#pragma once

#include "wave_functions.hpp"

#include <grid2grid/transform.hpp>

#include <type_traits>

namespace sddk {

// 1. Initializes / Retrieves component and slab offsets within the stacked matrix.
// 2. Retrieves pointers to slab data
// 3. Cast to T as the wave function data is always complex but T can also be `double` in which case the real and
//    imaginary components are interpreted as separate elements.
//
// Note: spin_param = 0 means spin UP.
//
template <typename T>
void init_wf_cpt_arrs(Wave_functions& phi, int spin_param, int index_of_start_col,
                      std::vector<int const*>& slab_sizes_arr, std::vector<T*>& loc_blk_data_arr)
{
    // All 4 components.
    //
    if (spin_param == 2 && phi.has_mt()) {
        slab_sizes_arr = {
            phi.pw_counts(),
            phi.mt_counts(),
            phi.pw_counts(),
            phi.mt_counts(),
        };
        loc_blk_data_arr = {
            reinterpret_cast<T*>(phi.pw_coeffs(0).prime().at(phi.preferred_memory_t(), 0, index_of_start_col)),
            reinterpret_cast<T*>(phi.mt_coeffs(0).prime().at(phi.preferred_memory_t(), 0, index_of_start_col)),
            reinterpret_cast<T*>(phi.pw_coeffs(1).prime().at(phi.preferred_memory_t(), 0, index_of_start_col)),
            reinterpret_cast<T*>(phi.mt_coeffs(1).prime().at(phi.preferred_memory_t(), 0, index_of_start_col)),
        };

        // Both spins, no Muffin-tin
        //
    } else if (spin_param == 2) {
        slab_sizes_arr = {
            phi.pw_counts(),
            phi.pw_counts(),
        };
        loc_blk_data_arr = {
            reinterpret_cast<T*>(phi.pw_coeffs(0).prime().at(phi.preferred_memory_t(), 0, index_of_start_col)),
            reinterpret_cast<T*>(phi.pw_coeffs(1).prime().at(phi.preferred_memory_t(), 0, index_of_start_col)),
        };

        // One spin with Muffin-tin
        //
    } else if (phi.has_mt()) {
        slab_sizes_arr = {
            phi.pw_counts(),
            phi.mt_counts(),
        };
        loc_blk_data_arr = {
            reinterpret_cast<T*>(phi.pw_coeffs(spin_param).prime().at(phi.preferred_memory_t(), 0, index_of_start_col)),
            reinterpret_cast<T*>(phi.mt_coeffs(spin_param).prime().at(phi.preferred_memory_t(), 0, index_of_start_col)),
        };

        // One spin no Muffin-tin
        //
    } else {
        slab_sizes_arr = {
            phi.pw_counts(),
        };
        loc_blk_data_arr = {
            reinterpret_cast<T*>(phi.pw_coeffs(spin_param).prime().at(phi.preferred_memory_t(), 0, index_of_start_col)),
        };
    }
}

// Wafe function columns are atomic bands, rows are coefficients of basis expansion. The number of columns is much
// smaller than the number of rows.
//
// Wave functions can be stacked into a single matrix.
//
// Row / column indexing starts from zero. The end is not included.
//
// The wave function is composed from up to four components. The componenets are stacked on top of each other. Each
// component is distributed among all processes in a block non-cyclic manner. Each process holds one block (slab) of
// each component. Slabs are not guaranteed to be of the same size.
//
// The GLOBAL (across all ranks) components of the wavefunction are stacked as follows if they exist:
//   1) spin   up, plane wave
//   2) spin   up, muffin-tin
//   3) spin down, plane wave
//   4) spin down, muffin-tin
//
// Each rank stores a single block from each component. Local blocks are stored in column-major.
//
template <typename T>
grid2grid::grid_layout<T> get_wf_grid_layout(std::vector<Wave_functions*> phi_arr, int spin_param,
                                             int index_of_start_wf, int num_cols)
{
    using namespace grid2grid;

    assert(spin_param == 0 || spin_param == 1 || spin_param == 2);
    assert(!phi_arr.empty());

    // Note: All wavefunctions have the same distribution and componenets
    //
    int this_rank = phi_arr[0]->comm().rank();
    int num_procs = phi_arr[0]->comm().size();
    bool have_mt  = phi_arr[0]->has_mt();
    int num_wfs   = static_cast<int>(phi_arr.size());

    // When `T` is double, the rows have to be multipled by 2. The real and imaginary cpts are interpreted as separate
    // elements stacked on top of each other.
    //
    constexpr int elem_type_factor = (std::is_same<T, double>::value) ? 2 : 1;

    // Columns are not split.
    //
    std::vector<int> cols_split{0, num_cols};

    // spin_param == 2 means both spins are needed
    //
    int num_spin_cpt = (spin_param != 2) ? 1 : 2;
    int num_pw_cpt   = num_spin_cpt;
    int num_mt_cpt   = (have_mt) ? num_spin_cpt : 0;
    int num_wf_cpt   = num_mt_cpt + num_pw_cpt;

    // Assumption : Both the Muffin-tin and Plane-wave components are distributed over all process.
    //
    int num_delims = num_procs * num_wf_cpt * num_wfs;
    std::vector<int> rows_split(num_delims + 1);
    std::vector<std::vector<int>> owners(num_delims, std::vector<int>(1));
    std::vector<block<T>> loc_blocks;
    loc_blocks.reserve(num_wf_cpt * num_wfs);

    // Iterate over each wave function bundle.
    //
    for (int i_wf = 0; i_wf < num_wfs; ++i_wf) {
        Wave_functions* phi = phi_arr[i_wf];

        // The start delimiter of the wave function
        //
        int wf_split = i_wf * num_wf_cpt * num_procs;

        std::vector<int const*> slab_sizes_arr;
        std::vector<T*> loc_blk_data_arr;
        init_wf_cpt_arrs(*phi, spin_param, index_of_start_wf, slab_sizes_arr, loc_blk_data_arr);

        // For each componenet within the wavefunction
        //
        rows_split[0] = 0;
        for (int i_cpt = 0; i_cpt < num_wf_cpt; ++i_cpt) {
            int const* slabs_sizes = slab_sizes_arr[i_cpt];
            T* loc_blk_data        = loc_blk_data_arr[i_cpt];

            // The start delimiter of the componenet
            //
            int cpt_split = wf_split + i_cpt * num_procs;

            // For each slab within the component
            //
            for (int rank = 0; rank < num_procs; ++rank) {
                int slab_split = cpt_split + rank;

                // Save the start delimiter of the next slab
                //
                rows_split[slab_split + 1] = rows_split[slab_split] + slabs_sizes[rank] * elem_type_factor;

                // There is only one column
                //
                owners[slab_split][0] = rank;
            }

            // clang-format off
            loc_blocks.push_back(
                {
                    {rows_split[cpt_split + this_rank], rows_split[cpt_split + this_rank + 1]},
                    {0, num_cols},
                    {cpt_split + this_rank, 0},
                    loc_blk_data
                }
            );
            // clang-format on
        }
    }

    // clang-format off
    return grid2grid::grid_layout<T>
    {
        {
            {
                std::move(rows_split),
                std::move(cols_split)
            },
            std::move(owners),
                    num_procs
        },
        {
            std::move(loc_blocks)
        }
    };
    // clang-format on
}

// `dmatrix` inherits from a column-major 2D mdarray. It also holds an object of the BLACS_grid clss which is built on
// the MPI_grid class. MPI_grid uses the default MPI row-major rank ordering.
//
template <typename T>
grid2grid::grid_layout<T> get_dmatrix_grid_layout(dmatrix<T>& result, int irow0__, int jcol0__, int submatrix_row_size,
                                                  int submatrix_col_size)
{
    using namespace grid2grid::scalapack;

    int this_rank  = result.comm().rank();
    char transpose = 'N';
    T* data        = result.at(sddk::memory_t::host);

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

    // From `blacs_grid.hpp` the ordering is always row-major.
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
template <typename T>
void assert_communicators_compatibility(Wave_functions const& bra, Wave_functions const& ket, dmatrix<T> const& result)
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

} // end namespace sddk
