#+private
package mat_test

import mat "../../mat"

import "core:testing"
import "core:log"

iota_mat :: proc($ROWS, $COLS: uint) -> (out: [ROWS*COLS]f32) {
    for i in 0..<len(out) { out[i] = f32(i) }

    return
}

// Test that this kernel works correctly for dimensions:
// ```odin
// A: 6xN
// B: Nx16
// ```
// See `kernel6x16_single_test` for more details.
// 
// It is exact because the kernel and output matrix `c` have the same dimensions. The test is run for
// multiple $N$ as it should work for all $N$(with some caveats)
@(test)
kernel6x16_single_exact :: proc(t: ^testing.T) {
    kernel6x16_single_test(t, 6, 2, 16)
    kernel6x16_single_test(t, 6, 4, 16)
    kernel6x16_single_test(t, 6, 5, 16)
    kernel6x16_single_test(t, 6, 10, 16)
    kernel6x16_single_test(t, 6, 25, 16)
    kernel6x16_single_test(t, 6, 30, 16)
    kernel6x16_single_test(t, 6, 40, 16)
}

// Like `kernel6x16_single_exact` tests with matrices whose dimensions
// are bigger than the kernel.
//
// This test targets the indexing logic in the kernel.
@(test)
kernel6x16_single_nonexact :: proc(t: ^testing.T) {
    // This did in fact help find a bug

    // Larger than kernel matrices
    kernel6x16_single_test(t, 12, 2, 32)
    kernel6x16_single_test(t, 14, 4, 17)
    kernel6x16_single_test(t, 21, 5, 21)
    kernel6x16_single_test(t, 14, 11, 16)
    kernel6x16_single_test(t, 14, 23, 43)
    kernel6x16_single_test(t, 14, 34, 20)

    // Larger than kernel but offset by (row, col)
    kernel6x16_single_test(t, 12, 2, 32, 4, 12)
    kernel6x16_single_test(t, 14, 4, 17, 6, 1)
    kernel6x16_single_test(t, 21, 5, 21, 13, 5)
    kernel6x16_single_test(t, 14, 11, 18, 4, 2)
    kernel6x16_single_test(t, 14, 23, 43, 5, 23)

    // Larger than kernel but offset by (row, col)
    // and also restricted to slice_start:slice_end
  
}

// Test that this kernel correctly performs the matrix multiplication:
// ```odin
// C[row:row+6][col:col+16] += A[row:row+6][slice_start:slice_end] * 
//                             B[slice_start:slice_end][col:col+16]
// ```
// 
// In other words, does it correctly compute the (partial) value of the 6x16 sub-matrix 
// of C which starts at `(row, col)`? Partial because sometimes the range of columns of A and range of rows of B
// included in the calculation is limited to `slice_start:slice_end`
//
// `kernel6x16_single_*` because it only computes a single 6x16 sub-matrix
//
// **Success criteria**: The result `c` should be equal to that of `smat_matmul_naive`
//
// This test is parameterized so it may be used for more specific tests
//
// ### Caveats
// This kernel utilizes SIMD and as a result, its arithmetic operations can ordered differently
// to that of `smat_matmul_naive`. Different rounding errors may occur due to the non-associativity of FP operations so 
// this kernel could produce different results for higher `A_COLS`. Limiting `A_COLS` and the magnitude of 
// matrix elements, we optimistically test with an exact comparison. 
kernel6x16_single_test :: proc(t: ^testing.T, 
    $A_ROWS, $A_COLS, $B_COLS: uint, 
    row: uint = 0, col: uint = 0, 
    slice_start: uint = 0, slice_end: Maybe(uint) = nil, 
    nc_rem: u16 = 0, nr_rem: u16 = 0 
) {
    #assert(A_ROWS >= 6)
    #assert(B_COLS >= 16)
    _slice_end := slice_end.? or_else A_COLS
    assert(row + 6 <= A_ROWS)
    assert(col + 16 <= B_COLS)
    assert(slice_start < A_COLS && _slice_end <= A_COLS && slice_start <= _slice_end)
    assert(nc_rem < 16 && nr_rem < 6 && col + uint(nc_rem) <= B_COLS && row + uint(nr_rem) <= A_ROWS)

    a := iota_mat(A_ROWS, A_COLS)
    b := iota_mat(A_COLS, B_COLS)
    c := [A_ROWS*B_COLS]f32{}
    c_ref := [A_ROWS*B_COLS]f32{}
    count, kr, kc := 0, uint(6), uint(16)

    switch {
        case nc_rem == 0 && nr_rem == 0:
            mat.smat_matmul_kernel(a[:], b[:], c[:], row, col, slice_start, _slice_end, A_COLS, B_COLS)
        case nc_rem > 0 && nr_rem == 0:
            kc = uint(nc_rem)
            mask0, mask1 := mat.create_row_mask(nc_rem)
            mat.smat_matmul_kernel_masked(a[:], b[:], c[:], row, col, slice_start, _slice_end, A_COLS, B_COLS, mask0, mask1)
        case nc_rem == 0 && nr_rem > 0:
            kr = uint(nr_rem)
            mat.smat_matmul_kernel_Nx16(a[:], b[:], c[:], row, col, slice_start, _slice_end, A_COLS, B_COLS, nr_rem)
        case:
            kr, kc = uint(nr_rem), uint(nc_rem)
            mask0, mask1 := mat.create_row_mask(nc_rem)
            mat.smat_matmul_kernel_Nx16_masked(a[:], b[:], c[:], row, col, slice_start, _slice_end, A_COLS, B_COLS, nr_rem, mask0, mask1)
    }

    // mat.smat_matmul_kernel(a[:], b[:], c[:], row, col, slice_start, _slice_end, A_COLS, B_COLS)
    
    // TODO: Stare at this for longer
    mat.smat_matmul_naive_strides(
        {kr, _slice_end-slice_start, a[row*A_COLS+slice_start:]}, 
        {_slice_end-slice_start, kc, b[slice_start*B_COLS+col:]}, 
        {kr, kc, c_ref[row*B_COLS+col:]},
        A_COLS, B_COLS, B_COLS)

    for i in 0..<len(c) { count += int(c[i] == c_ref[i]) }

    // Use relative difference to test equality in this case
    if count < len(c) { log.warnf("Only %d/%d elements matched. The test result may be inconclusive", count, len(c)) }
    testing.expect(t, count == len(c), "kernel result differs from reference")
}
