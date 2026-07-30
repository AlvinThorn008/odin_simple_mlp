package mat

import "core:simd"

f32x8 :: simd.f32x8
u16x8 :: simd.u16x8

// A typical matrix type
SMat :: struct {
    rows, cols: uint,
    data: []f32,
}

// A view over a SMat limited to slices with
// consecutive rows
SMatSlice :: struct {
    rows, cols: uint,
    // Row stride
    // The offset from one element to another in the same column but next row
    row_stride: uint,
    data: []f32
}

// Create a new matrix of given dimensions.
//
// The backing store is aligned to 32 bytes. This is not a requirement for constructing `SMat`s however.
new_smat :: proc(rows, cols: uint) -> SMat {
    return SMat { rows, cols, make_aligned([]f32, rows*cols, 32) }
}

new_smat_2d_array :: proc(arr: ^[$ROWS][$COLS]f32) -> SMat {
    LEN :: ROWS * COLS
    return SMat { ROWS, COLS, ([^]f32)(arr)[:LEN] }
}

// Deletes the matrix's backing store
delete_smat :: proc(self: SMat) {
    delete(self.data)
}

// Copy a matrix and return the copy
copy_smat :: proc(self: SMat) -> SMat {
    out := SMat { self.rows, self.cols, make_aligned([]f32, len(self.data), 32) }
    copy(out.data, self.data)
    return out
}  

/* Matrix operations */

// Element-wise addition
smat_add :: proc(self, other: SMat) {
    assert(smat_same_size(self, other), "size mismatch: could not add matrices")
    for i := 0; i < len(self.data); i += 1 {
        self.data[i] += other.data[i]
    }
}

smat_sub :: proc(self, other: SMat) {
    assert(smat_same_size(self, other), "size mismatch: could not subtract matrices")
    for i := 0; i < len(self.data); i += 1 {
        self.data[i] -= other.data[i]
    }
}

smat_scale :: proc(self: SMat, scalar: f32) {
    for i := 0; i < len(self.data); i += 1 {
        self.data[i] *= scalar
    }
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_naive :: proc(self, other, out: SMat) {
    for r in uint(0)..<self.rows {
        for c in uint(0)..<other.cols { // c(r, c) = a(r) . b(c)
            for k in uint(0)..<self.cols {
                out.data[r * out.cols + c] += self.data[r * self.cols + k] * other.data[k * other.cols + c]
            }
        }
    }
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_naive_strides :: proc(self, other, out: SMat, as, bs, cs: uint) {
    for r in uint(0)..<self.rows {
        for c in uint(0)..<other.cols { // c(r, c) = a(r) . b(c)
            for k in uint(0)..<self.cols {
                out.data[r * cs + c] += self.data[r * as + k] * other.data[k * bs + c]
            }
        }
    }
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_loop_order :: proc(self, other, out: SMat) {
    for r in uint(0)..<self.rows {
        for k in uint(0)..<self.cols {
            for c in uint(0)..<other.cols { // c(r, c) = a(r) . b(c)
                out.data[r * out.cols + c] += self.data[r * self.cols + k] * other.data[k * other.cols + c]
            }
        }
    }
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_with_kernel :: proc(a, b, c: SMat) {
    // nr x nc forms the dimension of the inner 6a x 16b sub-matrix the fixed kernel
    // will act on 
    nc := b.cols & ~uint(15)      
    nr := a.rows / 6 * 6  
    nc_rem := u16(b.cols & 15)  
    nr_rem := u16(a.rows - nr)
 
    // Mask creation
    // <mask0 mask1> forms a 16 element array: <1,1,1,...,  0,0,...>
    //                                          1 x nc_rem
    idxs0 := simd.from_array([8]u16{0,1,2,3,4,5,6,7})
    idxs1 := simd.from_array([8]u16{8,9,10,11,12,13,14,15})
    rems  := u16x8(nc_rem)
    mask0 := simd.lanes_gt(rems, idxs0)
    mask1 := simd.lanes_gt(rems, idxs1)
    
    row, col: uint
    for row = 0; row < nr; row += 6 {     
        for col = 0; col < nc; col += 16 {
            // C[row:row+6][col:col+16] += A[row:row+6][:] x B[:][col:col+16]
            #force_inline smat_matmul_kernel(a.data, b.data, c.data, row, col, 0, a.cols, a.cols, b.cols)
        }
        if nc_rem > 0 { // Remaining columns to sweep
            // C[row:row+6][col:col+nc_rem] += A[row:row+6][:] x B[:][col:col+nc_rem]
            #force_inline smat_matmul_kernel_masked(a.data, b.data, c.data, row, col, 0, a.cols, a.cols, b.cols, mask0, mask1)
        }
    }
    if nr_rem > 0 { // Remaining rows to sweep
        for col = 0; col < nc; col += 16 {
            // C[row:row+nr_rem][col:col+16] += A[row:row+nr_rem][:] x B[:][col:col+16]
            #force_inline smat_matmul_kernel_Nx16(a.data, b.data, c.data, row, col, 0, a.cols, a.cols, b.cols, nr_rem)
        }
        if nc_rem > 0 {
            // C[row:row+nr_rem][col:col+nc_rem] += A[row:row+nr_rem][:] x B[:][col:col+nc_rem]
            #force_inline smat_matmul_kernel_Nx16_masked(a.data, b.data, c.data, row, col, 0, a.cols, a.cols, b.cols, nr_rem, mask0, mask1)
        }
    }
}

BLOCKING_COMMON_MAX :: 2500
BLOCKING_COMMON_MIN :: 430

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_blocking :: proc(a, b, c: SMat) {
    switch {
        case a.cols < BLOCKING_COMMON_MIN: 
            smat_matmul_with_kernel(a, b, c)
        case a.cols < BLOCKING_COMMON_MAX: 
            smat_matmul_blocking_untuned(a, b, c, 230, 180, 96)
        case: 
            smat_matmul_blocking_untuned(a, b, c, 350, 6, 16)
    }
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_blocking_untuned :: proc(a, b, c: SMat, $s1, $s2, $s3: uint) {
    #assert(s2 % 6 == 0, "s2 must be divisible by 6")
    #assert(s3 % 16 == 0, "s3 must be divisible by 16")

    // nr x nc forms the dimension of the inner 6a x 16b sub-matrix the fixed kernel
    // will act on 
    nc := b.cols & ~uint(15)      
    nr := a.rows / 6 * 6  
    nc_rem := u16(b.cols & 15)  
    nr_rem := u16(a.rows - nr)
 
    // Mask creation
    // <mask0 mask1> forms a 16 element array: <1,1,1,...,  0,0,...>
    //                                          1 x nc_rem
    idxs0 := simd.from_array([8]u16{0,1,2,3,4,5,6,7})
    idxs1 := simd.from_array([8]u16{8,9,10,11,12,13,14,15})
    rems  := u16x8(nc_rem)
    mask0 := simd.lanes_gt(rems, idxs0)
    mask1 := simd.lanes_gt(rems, idxs1)


    for i3 := uint(0); i3 < nc; i3 += s3 {
        b_col_end := min(i3 + s3, nc)

        for i2 := uint(0); i2 < nr; i2 += s2 {
            a_col_end := min(i2 + s2, nr)

            for i1 := uint(0); i1 < a.cols; i1 += s1 {
                l := i1
                r := min(i1 + s1, a.cols)

                #force_inline smat_matmul_kernel_super(
                    a.data, b.data, c.data, l, r, a.cols, b.cols,
                    nr_rem, nc_rem, mask0, mask1, a_col_end, b_col_end,
                    i2, i3, nc, nr
                )
            }
        }
    }
}

@(private)
@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_kernel_super :: proc(
    a, b, c: []f32, l, r, a_cols, b_cols: uint, 
    nr_rem, nc_rem: u16, mask0: u16x8, mask1: u16x8, a_col_end, b_col_end: uint,
    i2, i3, nc, nr: uint
) {
    row, col: uint
    for row = i2; row < a_col_end; row += 6 {     
        for col = i3; col < b_col_end; col += 16 {
            // C[row:row+6][col:col+16] += A[row:row+6][:] x B[:][col:col+16]
            #force_inline smat_matmul_kernel(a, b, c, row, col, l, r, a_cols, b_cols)
        }
        if nc_rem > 0 && b_col_end == nc { // Remaining columns to sweep
            // C[row:row+6][col:col+nc_rem] += A[row:row+6][:] x B[:][col:col+nc_rem]
            #force_inline smat_matmul_kernel_masked(a, b, c, row, col, l, r, a_cols, b_cols, mask0, mask1)
        }
    }
    if nr_rem > 0 && a_col_end == nr { // Remaining rows to sweep
        for col = i3; col < b_col_end; col += 16 {
            // C[row:row+nr_rem][col:col+16] += A[row:row+nr_rem][:] x B[:][col:col+16]
            #force_inline smat_matmul_kernel_Nx16(a, b, c, row, col, l, r, a_cols, b_cols, nr_rem)
        }
        if nc_rem > 0 && b_col_end == nc {
            // C[row:row+nr_rem][col:col+nc_rem] += A[row:row+nr_rem][:] x B[:][col:col+nc_rem]
            #force_inline smat_matmul_kernel_Nx16_masked(a, b, c, row, col, l, r, a_cols, b_cols, nr_rem, mask0, mask1)
        }
    }

}

smat_same_size :: proc(a, b: SMat) -> bool {
    return a.cols == b.cols && a.rows == b.rows
}

smat_matmul_agree :: proc(a, b: SMat) -> bool {
    return a.cols == b.rows
}



