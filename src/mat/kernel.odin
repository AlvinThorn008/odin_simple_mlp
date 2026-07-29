package mat
import "core:simd"
import "base:intrinsics"

/*
A vectorized matmul kernel procedure

Compute a 6x16 sub-matrix of C

### Operation
```odin
C[x:x+6][y:y+16] += A[x:x+6][l:r] * B[l:r][y:y+16]
```

If `l:r = 0:A.cols`, the sub-matrix is computed completely, otherwise the column slice and row slice l:r
of the relevant row slice of A and column slice of B respectively is used to partially compute the sub-matrix.
This [graphic](https://cnugteren.github.io/tutorial/pages/page4.html) might help.

### Implementation details
The sub-matrix of C is read into 12 vector registers. Matrix A is read 6 elements down for every column in
the column slice `l:r`. Each element is broadcasted to a 256-bit register* and FMA'd with 16 elements of Matrix B. The FMA
loop is unrolled to maximize utilization of the FMA ports although I never verified the actual throughput and latency of the
instruction on my CPU.

All memory accesses and writes use unaligned load and store intrinsics. If you want to avoid unaligned loads and stores, matrices B and C
must have every row aligned to the vector width(=32 bytes). 

### Variants
Three other versions of this procedure exist to handle edge cases(literally edge cases) when computing the entire matrix C where the dimensions
can be expressed as `6ax16b`:
- `smat_matmul_kernel_masked` for left and right edges or 6xN sub-matrices, `N <= 16`
- `smat_matmul_kernel_Nx16` - for bottom edge or Nx16 sub-matrices, `N <= 6`
- `smat_matmul_kernel_Nx16_masked` - for bottom right corner or NxM sub-matrices, `N <= 6, M <= 16`

\* : Assuming the compiler does its job
*/
@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_kernel :: proc(a, b, c: []f32, x, y, l, r, a_cols, b_cols: uint) {
    t: [6][2]f32x8
    // Read c into vector registers
    #unroll for i in uint(0)..<6 {
        #unroll for j in uint(0)..<2 {
            t[i][j] = simd_from_slice(f32x8, c[(x + i) * b_cols + y + 8 * j:])
        }
    }
    
    for k in l..<r {
        b0 := simd_from_slice(f32x8, b[k * b_cols + y:])
        b1 := simd_from_slice(f32x8, b[k * b_cols + y + 8:])

        #unroll for i in uint(0)..<6 { // Assume through
            a_broad := f32x8(a[(x + i) * a_cols + k]) // Broadcast
            t[i][0] = simd.fma(a_broad, b0, t[i][0])
            t[i][1] = simd.fma(a_broad, b1, t[i][1])
        }
    }

    #unroll for i in uint(0)..<6 {
        #unroll for j in uint(0)..<2 {
            intrinsics.unaligned_store(
                (^f32x8)(&c[(x + i) * b_cols + y + 8 * j]),
                t[i][j]
            )
        }
    }
}

// Make a mask to select the first `nc_rem` elements for a 16 element vector
// 
// Returns the mask as two 8-element vectors. This is more convenient for the masked kernel procedures.
create_row_mask :: proc(nc_rem: u16) -> (u16x8, u16x8) {
    idxs0 := simd.from_array([8]u16{0,1,2,3,4,5,6,7})
    idxs1 := simd.from_array([8]u16{8,9,10,11,12,13,14,15})
    rems  := u16x8(nc_rem)
    mask0 := simd.lanes_gt(rems, idxs0)
    mask1 := simd.lanes_gt(rems, idxs1)

    return mask0, mask1
}

// Masked variant of `smat_matmul_kernel`
@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_kernel_masked :: proc(a, b, c: []f32, x, y, l, r, a_cols, b_cols: uint, mask0: u16x8, mask1: u16x8) {
    zero := f32x8(0.0)
    b_ptr := raw_data(b)
    c_ptr := raw_data(c)
    // Laziness
    nc_rem := uint(card(simd.extract_lsbs(mask0)) + card(simd.extract_lsbs(mask1))) // Just pass in nc_rem blud 💔💔💔

    t: [6][2]f32x8
    // Read c into vector registers
    #unroll for i in uint(0)..<6 {
        assert((x + i) * b_cols + y + nc_rem <= len(c))
        t[i][0] = simd.masked_load(&c_ptr[(x + i) * b_cols + y], zero, mask0)
        t[i][1] = simd.masked_load(&c_ptr[(x + i) * b_cols + y + 8], zero, mask1)
    }

    for k in l..<r {  
        // Slices are bound checked by default so use a multi-pointer
        // to get out of bound references. This is fine since `simd.masked_load`
        // doesn't masked off values from `src` given our masks are calculated correctly
        // of course.
        assert(k * b_cols + y + nc_rem <= len(b))
        b0 := simd.masked_load(&b_ptr[k * b_cols + y], zero, mask0)
        b1 := simd.masked_load(&b_ptr[k * b_cols + y + 8], zero, mask1)

        #unroll for i in uint(0)..<6 {
            a_broad := f32x8(a[(x + i) * a_cols + k]) // Broadcast
            t[i][0] = simd.fma(a_broad, b0, t[i][0])
            t[i][1] = simd.fma(a_broad, b1, t[i][1])
        }
    }

    #unroll for i in uint(0)..<6 {
        assert((x + i) * b_cols + y + nc_rem <= len(c))
        simd.masked_store(&c_ptr[(x + i) * b_cols + y], t[i][0], mask0)
        simd.masked_store(&c_ptr[(x + i) * b_cols + y + 8], t[i][1], mask1)
    }
}

// Like `smat_matmul_kernel` but can compute Nx16 sub-matrices of C. 
// 
// `N = nr_rem <= 6`
@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_kernel_Nx16 :: proc(a, b, c: []f32, x, y, l, r, a_cols, b_cols: uint, nr_rem: u16) {
    t: [6][2]f32x8
    // Read c into vector registers
    for i in 0..<uint(nr_rem) {
        #unroll for j in uint(0)..<2 {
            t[i][j] = simd_from_slice(f32x8, c[(x + i) * b_cols + y + 8 * j:])
        }
    }

    for k in l..<r {
        b0 := simd_from_slice(f32x8, b[k * b_cols + y:])
        b1 := simd_from_slice(f32x8, b[k * b_cols + y + 8:])

        for i in 0..<uint(nr_rem) {
            a_broad := f32x8(a[(x + i) * a_cols + k]) // Broadcast
            t[i][0] = simd.fma(a_broad, b0, t[i][0])
            t[i][1] = simd.fma(a_broad, b1, t[i][1])
        }
    }

    for i in 0..<uint(nr_rem) {
        #unroll for j in 0..<uint(2) {
            intrinsics.unaligned_store(
                (^f32x8)(&c[(x + i) * b_cols + y + 8 * j]),
                t[i][j]
            )
        }
    }
}

// Masked variant of `smat_matmul_kernel_Nx16`
@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_kernel_Nx16_masked :: proc(a, b, c: []f32, x, y, l, r, a_cols, b_cols: uint, nr_rem: u16, mask0: u16x8, mask1: u16x8) {
    zero := f32x8(0.0)
    b_ptr := raw_data(b)
    c_ptr := raw_data(c)
    nc_rem := uint(card(simd.extract_lsbs(mask0)) + card(simd.extract_lsbs(mask1))) // Just pass in nc_rem blud 💔💔💔

    t: [6][2]f32x8
    // Read c into vector registers
    for i in 0..<uint(nr_rem) {
        assert((x + i) * b_cols + y + nc_rem <= len(c))
        t[i][0] = simd.masked_load(&c_ptr[(x + i) * b_cols + y], zero, mask0)
        t[i][1] = simd.masked_load(&c_ptr[(x + i) * b_cols + y + 8], zero, mask1)
    }

    for k in l..<r {
        assert(k * b_cols + y + nc_rem <= len(b))
        b0 := simd.masked_load(&b_ptr[k * b_cols + y], zero, mask0)
        b1 := simd.masked_load(&b_ptr[k * b_cols + y + 8], zero, mask1)

        for i in 0..<uint(nr_rem) {
            a_broad := f32x8(a[(x + i) * a_cols + k]) // Broadcast
            t[i][0] = simd.fma(a_broad, b0, t[i][0])
            t[i][1] = simd.fma(a_broad, b1, t[i][1])
        }
    }

    for i in 0..<uint(nr_rem) {
        assert((x + i) * b_cols + y + nc_rem <= len(c))
        simd.masked_store(&c_ptr[(x + i) * b_cols + y], t[i][0], mask0)
        simd.masked_store(&c_ptr[(x + i) * b_cols + y + 8], t[i][1], mask1)
    }
}