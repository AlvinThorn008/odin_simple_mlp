package mat
import "core:simd"
import "base:intrinsics"

/*
Vectorized matmul kernel

In the matmul C = AB, the 6x16 sub-matrix of C starting at row x 
and column y can be computed as such 
```python
C[x:x+6][y:y+16] = A[x:x+6][:] * B[:][y:y+16]
```
The



## Operation
```python
A[x:x+6][l:r]  B[l:r][y:y+16]
```
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

    // 
    for k in l..<r {
        b0 := simd_from_slice(f32x8, b[k * b_cols + y:])
        b1 := simd_from_slice(f32x8, b[k * b_cols + y + 8:])

        #unroll for i in uint(0)..<6 {
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

create_row_mask :: proc(nc_rem: u16) -> (u16x8, u16x8) {
    idxs0 := simd.from_array([8]u16{0,1,2,3,4,5,6,7})
    idxs1 := simd.from_array([8]u16{8,9,10,11,12,13,14,15})
    rems  := u16x8(nc_rem)
    mask0 := simd.lanes_gt(rems, idxs0)
    mask1 := simd.lanes_gt(rems, idxs1)

    return mask0, mask1
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_kernel_masked :: proc(a, b, c: []f32, x, y, l, r, a_cols, b_cols: uint, mask0: u16x8, mask1: u16x8) {
    zero := f32x8(0.0)
    b_ptr := raw_data(b)
    c_ptr := raw_data(c)
    nc_rem := uint(card(simd.extract_lsbs(mask0)) + card(simd.extract_lsbs(mask1))) // Just pass in nc_rem blud 💔💔💔

    t: [6][2]f32x8
    // Read c into vector registers
    #unroll for i in uint(0)..<6 {
        assert((x + i) * b_cols + y + nc_rem <= len(c))
        t[i][0] = simd.masked_load(&c_ptr[(x + i) * b_cols + y], zero, mask0)
        t[i][1] = simd.masked_load(&c_ptr[(x + i) * b_cols + y + 8], zero, mask1)
    }

    // 
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