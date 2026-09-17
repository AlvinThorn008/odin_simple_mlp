package network

import "base:intrinsics"
import "core:simd"
import "core:math"
import "core:simd/x86"
import "core:mem"
import smat "../mat"

i32x8 :: simd.i32x8
u32x8 :: simd.u32x8
u16x8 :: simd.u16x8

// Rectified Linear Unit (ReLU): `out = max(mat, 0.0)`
relu :: proc(mat, out: SMat) {
    assert(smat.smat_same_size(mat, out), "size mismatch: ReLU is an element-wise operation")
    zero := simd.f32x8(0.0)
    len_vecs := len(mat.data) & ~int(7)
    i := 0
    for ; i < len_vecs; i += 8 {
        // Assumes SMat's data is aligned to 32 bytes
        a := (^simd.f32x8)(&mat.data[i])
        b := (^simd.f32x8)(&out.data[i])
        b^ = simd.max(a^, zero)
    }
    for j in i..<len(mat.data) do out.data[j] = max(mat.data[j], 0.0)
}

diff_relu :: proc(mat, out: SMat) {
    assert(smat.smat_same_size(mat, out), "size mismatch: dReLU is an element-wise operation")
    zero := simd.f32x8(0.0)
    one := transmute(simd.u32x8)(simd.f32x8(1.0))
    len_vecs := len(mat.data) & ~int(7)
    i := 0
    for ; i < len_vecs; i += 8 {
        // Assumes SMat's data is aligned to 32 bytes
        a := (^simd.f32x8)(&mat.data[i])
        b := (^simd.f32x8)(&out.data[i])
        // NOTE: lanes_gt sets all bits in a lane if lane > 0, and 0 otherwise
        // since f32(+0.0) and u32(0) have the same repr, the result
        // will have 1.0 where the lane > 0 and +0.0 otherwise
        b^ = transmute(simd.f32x8)(simd.bit_and(simd.lanes_gt(a^, zero), one))
    }
    for j in i..<len(mat.data) do out.data[j] = mat.data[j] > 0.0 ? 1.0 : 0.0
}

@(enable_target_feature="avx")
exp_best_avx2 :: proc(x: f32x8) -> f32x8 {
    exp_hi :: f32x8(88.3762626647949)
    exp_lo :: f32x8(-88.3762626647949)
    cephes_LOG2EF :: f32x8(1.44269504088896341)
    inv_LOG2EF :: f32x8(0.693147180559945)
    

    cephes_exp_p0 :: f32x8(1.9875691500E-4)
    cephes_exp_p1 :: f32x8(1.3981999507E-3)
    cephes_exp_p2 :: f32x8(8.3334519073E-3)
    cephes_exp_p3 :: f32x8(4.1665795894E-2)
    cephes_exp_p4 :: f32x8(1.6666665459E-1)
    cephes_exp_p5 :: f32x8(5.0000001201E-1)

    one :: f32x8(1.0)

    fx, y, z, pow2n: f32x8
    imm0: i32x8

    x := x

    x = simd.min(x, exp_hi)
    x = simd.max(x, exp_lo)

    fx = simd.mul(x, cephes_LOG2EF)
    // fx = x86._mm_round_ps(fx, 0x00)
    fx = simd.nearest(fx)
    z  = simd.mul(fx, inv_LOG2EF)
    x  = simd.sub(x, z)
    z  = simd.mul(x, x)

    y  = simd.fma(cephes_exp_p0, x, cephes_exp_p1)
    y  = simd.fma(y, x, cephes_exp_p2)
    y  = simd.fma(y, x, cephes_exp_p3)
    y  = simd.fma(y, x, cephes_exp_p4)
    y  = simd.fma(y, x, cephes_exp_p5)
    y  = simd.fma(y, z, x)
    y  = simd.add(y, one)

    imm0 = transmute(i32x8)x86._mm256_cvttps_epi32(fx)
    imm0 = simd.add(imm0, i32x8(0x7f))
    imm0 = simd.shl(imm0, u32x8(23))

    y = simd.mul(y, transmute(f32x8)imm0)

    return y
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
softmax_batched2 :: proc(x, y: SMat) {
    rows, cols := x.rows, x.cols
    num_vecs, len8 := cols >> 3, cols & ~uint(7)
    rem := cols & uint(7)
    zero := f32x8(0.0)
    
    maxes := make_aligned([]f32, cols, 32, context.temp_allocator)
    sums  := make_aligned([]f32, cols, 32, context.temp_allocator)
    copy(maxes[:], x.data[:cols]) // y(0, :) = x(0, :)
    mem.zero_slice(sums)
    maxes8 := mem.slice_data_cast([]f32x8, maxes[:len8]) // []f32x8 view into y(0, :)
    sums8  := mem.slice_data_cast([]f32x8, sums[:len8])

    mask := simd.lanes_gt(u32x8(u32(rem)), simd.iota(u32x8))

    // Find maximum along each column by taking the max of the rows
    // acc = max(acc, row1, row2, row3, ...)
    for r in 1..<rows {
        row := ([^]f32x8)(raw_data(x.data[r*cols:]))[:num_vecs]
        // row := #force_inline mem.slice_data_cast([]f32x8, x.data[r*cols:][:len8])
        for b in 0..<num_vecs do maxes8[b] = simd.max(maxes8[b], intrinsics.unaligned_load(&row[b]))

        for i in len8..<cols {
            v := x.data[r*cols + i]
            maxes[i] = max(maxes[i], v)
        }
    }

    for r in 0..<rows {
        row := ([^]f32x8)(raw_data(x.data[r*cols:]))[:num_vecs]
        y_row := ([^]f32x8)(raw_data(y.data[r*cols:]))[:num_vecs]
        
        for b in 0..<num_vecs {
            exps := exp_best_avx2(simd.sub(intrinsics.unaligned_load(&row[b]), maxes8[b]))
            intrinsics.unaligned_store(&y_row[b], exps)
            sums8[b] = simd.add(exps, sums8[b])
        }

        if rem != 0 {
            exps := exp_best_avx2(simd.sub(
                simd.masked_load(&x.data[r*cols + len8], zero, mask),
                simd.masked_load(&maxes[len8], zero, mask)
            ))

            simd.masked_store(&y.data[r*cols + len8], exps, mask)
            simd.masked_store(
                &sums[len8],
                simd.add(exps, simd.masked_load(&sums[len8], zero, mask)),
                mask
            )
        }
    }
    
    for r in 0..<rows {
        y_row := ([^]f32x8)(raw_data(y.data[r*cols:]))[:num_vecs]     

        for b in 0..<num_vecs do intrinsics.unaligned_store(
            &y_row[b],
            simd.div(intrinsics.unaligned_load(&y_row[b]), sums8[b])
        )

        for i in len8..<cols {
            y.data[r*cols + i] = y.data[r*cols + i] / sums[i]
        }
    }
    
    free_all(context.temp_allocator)

}


@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
softmax_batched :: proc(mat, out: SMat) {
    assert(smat.smat_same_size(mat, out), "size mismatch: softmax is an element-wise operation")

    for c in 0..<mat.cols {
        max_val := -math.INF_F32
        sum := f32(0.0)
        for r in 0..<mat.rows do max_val = max(max_val, mat.data[r * mat.cols + c])

        for r in 0..<mat.rows {
            s0 := math.exp(mat.data[r * mat.cols + c] - max_val)
            out.data[r * out.cols + c] = s0
            sum += s0
        }

        for r in 0..<mat.rows do out.data[r * out.cols + c] /= sum
    }
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
softmax :: proc(mat, out: SMat) {
    assert(smat.smat_same_size(mat, out), "size mismatch: softmax is an element-wise operation")
    assert(mat.cols == 1, "inputs must be column vectors")
    max_val := -math.INF_F32
    len_vecs := len(mat.data) & ~int(7)

    i := 0
    sum := f32(0.0)
    for ; i < len_vecs; i += 8 {
        a := (^simd.f32x8)(&mat.data[i])
        max_val = max(simd.reduce_max(a^), max_val)
    }
    for val in mat.data[i:] {
        max_val = max(val, max_val)
    }

    for val in mat.data {
        sum += math.exp(val - max_val)
    }

    for k in 0..<len(mat.data) {
        out.data[k] = math.exp(mat.data[k] - max_val) / sum
    }
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
cross_entropy :: proc(a, b: SMat) -> f64 {
    assert(smat.smat_same_size(a, b), "size mismatch: matrices must be the same size")

    // Cross entropy is normally computed with vectors but it can be useful to compute
    // it over one axis of same size matrices.
    // Just saying, I'm not doing that here atm however this version is useful or also efficient for
    // calculating the average cost of a batch of examples

    sum := f64(0.0)
    for i in 0..<len(a.data) do sum -= f64(b.data[i]) * math.ln(f64(max(a.data[i], 1e-7)))

    return sum
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
diff_cross_entropy :: proc(output, target, res: SMat) {
    assert(smat.smat_same_size(output, target) && smat.smat_same_size(target, res), "size mismatch: matrices must be the same size")

    for i in 0..<len(output.data) {
        res.data[i] -= -target.data[i] / (output.data[i] + 1e-7)
    }
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
squared_error :: proc(a, b: SMat) -> f64 {
    assert(smat.smat_same_size(a, b), "size mismatch: matrices must be the same size")
    sum: f64 = 0.0
    for i := 0; i < len(a.data); i += 1 {
        diff := f64(a.data[i]) - f64(b.data[i])
        sum += diff * diff
    }
    return sum
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
diff_squared_error :: proc(a, b, out: SMat) {
    assert(smat.smat_same_size(a, b), "size mismatch: matrices must be the same size")
    for i := 0; i < len(a.data); i += 1 {
        out.data[i] = 2.0 * (a.data[i] - b.data[i])
    }
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
sigmoid :: proc(self, out: SMat) {
    assert(smat.smat_same_size(self, out), "size mismatch: matrices must be the same size")
    for i := 0; i < len(self.data); i += 1 {
        out.data[i] = 1.0 / (1.0 + math.exp_f32(-self.data[i]))
    }
}

@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
diff_sigmoid :: proc(self, out: SMat) {
    assert(smat.smat_same_size(self, out), "size mismatch: matrices must be the same size")
    sig: f32
    for i := 0; i < len(self.data); i += 1 {
        sig = 1.0 / (1.0 + math.exp_f32(-self.data[i]))
        out.data[i] = sig * (1.0 - sig)
    }
}

argmax :: proc(mat: SMat) -> uint {
    max_val := -math.INF_F32
    len_vecs := len(mat.data) & ~int(7)

    max_idx: uint = 0
    for val, i in mat.data {
        if val >= max_val { max_val = val; max_idx = uint(i) } 
    }
    return max_idx 
}

argmax_batched :: proc(mat: SMat, out: []u32) {
    assert(uint(len(out)) == mat.cols, "size mismatch: `out` must have one element per column")
    assert(mem.is_aligned(raw_data(out), 32), "`out` must be aligned")

    rows, cols := mat.rows, mat.cols
    num_vecs, len8 := cols >> 3, cols & ~uint(7)

    max_vals := make_aligned([]f32, cols, 32, context.temp_allocator)
    out_vec := ([^]u32x8)(raw_data(out))[:num_vecs]
    max_vals_vec := ([^]f32x8)(raw_data(max_vals))[:num_vecs]
    
    copy(max_vals, mat.data[:cols])
    mem.zero_slice(out)
    
    for r in 1..<rows {
        row := ([^]f32x8)(raw_data(mat.data[r*cols:]))[:num_vecs]

        for b in 0..<num_vecs {
            a := intrinsics.unaligned_load(&row[b])
            max_val := max_vals_vec[b]
            mask := simd.lanes_ge(a, max_val)
            max_vals_vec[b] = simd.select(mask, a, max_val)
            out_vec[b] = simd.select(mask, u32x8(r), out_vec[b])
        }

        for i in len8..<cols {
            a, max_val := mat.data[r*cols + i], max_vals[i]
            sel := a >= max_val
            max_vals[i] = sel ? a : max_val
            if sel do out[i] = u32(r)
        }
    }

    free_all(context.temp_allocator)
}

