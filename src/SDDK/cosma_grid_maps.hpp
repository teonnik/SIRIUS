#pragma once

#include "wave_functions.hpp"

#include <grid2grid/transform.hpp>

namespace sddk {

// 1. Initializes / Retrieves component and slab offsets within the stacked matrix.
// 2. Retrieves pointers to slab data
//
// Note: spin_param = 0 means spin UP.
//
template <typename T>
void init_wf_cpt_arrs(Wave_functions& phi, int spin_param, int index_of_start_wf, std::vector<int>& cpt_offsets_arr,
                      std::vector<int const*>& slab_offsets_arr, std::vector<T*>& loc_blk_data_arr)
{
    int num_pw_rows = phi.gkvec().num_gvec();
    int num_mt_rows = phi.mt_size();

    // All 4 components.
    //
    if (spin_param == 2 && phi.has_mt()) {
        cpt_offsets_arr  = {0, num_pw_rows, num_pw_rows + num_mt_rows, 2 * num_pw_rows + num_mt_rows};
        slab_offsets_arr = {
            phi.pw_offsets(),
            phi.mt_offsets(),
            phi.pw_offsets(),
            phi.mt_offsets(),
        };
        loc_blk_data_arr = {
            phi.pw_coeffs(0).prime().at(phi.preferred_memory_t(), 0, index_of_start_wf),
            phi.mt_coeffs(0).prime().at(phi.preferred_memory_t(), 0, index_of_start_wf),
            phi.pw_coeffs(1).prime().at(phi.preferred_memory_t(), 0, index_of_start_wf),
            phi.mt_coeffs(1).prime().at(phi.preferred_memory_t(), 0, index_of_start_wf),
        };

        // Both spins, no Muffin-tin
        //
    } else if (spin_param == 2) {
        cpt_offsets_arr = {
            0,
            num_pw_rows,
        };
        slab_offsets_arr = {
            phi.pw_offsets(),
            phi.pw_offsets(),
        };
        loc_blk_data_arr = {
            phi.pw_coeffs(0).prime().at(phi.preferred_memory_t(), 0, index_of_start_wf),
            phi.pw_coeffs(1).prime().at(phi.preferred_memory_t(), 0, index_of_start_wf),
        };

        // One spin with Muffin-tin
        //
    } else if (phi.has_mt()) {
        cpt_offsets_arr = {
            0,
            num_pw_rows,
        };
        slab_offsets_arr = {
            phi.pw_offsets(),
            phi.mt_offsets(),
        };
        loc_blk_data_arr = {
            phi.pw_coeffs(spin_param).prime().at(phi.preferred_memory_t(), 0, index_of_start_wf),
            phi.pw_coeffs(spin_param).prime().at(phi.preferred_memory_t(), 0, index_of_start_wf),
        };

        // One spin no Muffin-tin
        //
    } else {
        cpt_offsets_arr = {
            0,
        };
        slab_offsets_arr = {
            phi.pw_offsets(),
        };
        loc_blk_data_arr = {
            phi.pw_coeffs(spin_param).prime().at(phi.preferred_memory_t(), 0, index_of_start_wf),
        };
    }
}

// The number of wave functions (num_wfs / `m` or `n`) is much smaller than the dimensionality of the space (`k`).
//
// Row / column indexing starts from zero. The end is not included.
//
// The wave function is composed from up to 4 components. The componenets are stacked on top of each other. Each
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
grid2grid::grid_layout<T> get_wf_grid_layout(Wave_functions& phi, int spin_param, int index_of_start_wf, int num_cols)
{
    using namespace grid2grid;
    assert(spin_param == 0 || spin_param == 1 || spin_param == 2);

    int this_rank = phi.comm().rank();
    int num_procs = phi.comm().size();

    // Columns are not split.
    //
    std::vector<int> cols_split{0, num_cols};

    // spin_param == 2 means both spins are needed
    //
    int num_spin_cpt = (spin_param != 2) ? 1 : 2;
    int num_pw_cpt   = num_spin_cpt;
    int num_mt_cpt   = (phi.has_mt()) ? num_spin_cpt : 0;
    int num_wf_cpt   = num_mt_cpt + num_pw_cpt;

    int num_pw_rows = phi.gkvec().num_gvec();
    int num_mt_rows = phi.mt_size();

    std::vector<int> rows_split(num_procs * num_wf_cpt + 1);
    rows_split[num_procs * num_wf_cpt] = num_pw_cpt * num_pw_rows + num_mt_cpt * num_mt_rows; // last split delimiter
    std::vector<std::vector<int>> owners(rows_split.size() - 1, std::vector<int>(1));
    std::vector<block<T>> loc_blocks;
    loc_blocks.reserve(num_wf_cpt);

    std::vector<int> cpt_rows_offsets_arr;
    std::vector<int const*> slab_offsets_arr;
    std::vector<T*> loc_blk_data_arr;
    init_wf_cpt_arrs(phi, spin_param, index_of_start_wf, cpt_rows_offsets_arr, slab_offsets_arr, loc_blk_data_arr);

    // Iterate over each componenet.
    //
    for (int i_cpt = 0; i_cpt < num_wf_cpt; ++i_cpt) {
        int const* slab_offsets = slab_offsets_arr[i_cpt];
        int cpt_offset          = cpt_rows_offsets_arr[i_cpt];
        T* loc_blk_data         = loc_blk_data_arr[i_cpt];

        // Split among all processes and note that components are stacked.
        //
        int split_offset = i_cpt * num_procs;

        // For each slab
        //
        for (int rank = 0; rank < num_procs; ++rank) {
            int i_split = split_offset + rank;

            // Add the offset of the current component to the offset of the slab within it
            //
            rows_split[i_split] = cpt_offset + slab_offsets[rank];

            // There is only one column
            //
            owners[i_split][0] = rank;
        }

        // clang-format off
        loc_blocks.push_back(
            {
                {rows_split[split_offset + this_rank], rows_split[split_offset + this_rank + 1]},
                {0, num_cols},
                {split_offset + this_rank, 0},
                loc_blk_data
            }
        );
        // clang-format on
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
