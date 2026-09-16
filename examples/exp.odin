package exp_avx2

import "core:slice"
import "base:intrinsics"
import "core:fmt"
import "core:math/rand"
import "core:math"
import "core:simd"
import "core:simd/x86"
import "core:mem"
import mat "../src/mat"

f32x8 :: simd.f32x8
i32x8 :: simd.i32x8
u32x8 :: simd.u32x8
u16x8 :: simd.u16x8
SMat  :: mat.SMat

main :: proc() {
    // x: [500]f32
    // for i in 0..<50 do x[i] = rand.float32_range(-25.0, 40.0)

    // for i in 0..<50 {
    //     target := math.exp_f32(x[i])
    //     approx := simd.extract(exp_avx2(x[i]), 0)
    //     approx2 := simd.extract(exp_best_avx2(x[i]), 0)

    //     rel_tol := math.abs(target - approx) / min(math.abs(target), math.abs(approx))
    //     rel_tol2 := math.abs(target - approx2) / min(math.abs(target), math.abs(approx2))
    //     fmt.printfln("x = %13e   y = %13e   y1 = %13e   y2 = %13e   rel_err1 = %13e   rel_err2 = %13e", x[i], target, approx, approx2, rel_tol, rel_tol2)
    // }
    
    x, y := mat.new_smat(33, 33), mat.new_smat(33, 33)
    for &v, i in x.data[5*33:][:33] do v = f32(i)

    softmax(x, y)
}

// Has no input range clamp
@(enable_target_feature="avx")
exp_avx2 :: proc(x: f32x8) -> f32x8 {

    t, f, p, r: f32x8
    i, j: i32x8

    l2e := f32x8(1.442695041)
    l2h := f32x8(-6.93145752e-1)
    l2l := f32x8(-1.42860677e-6)
    c0  := f32x8(0.041944388)
    c1  := f32x8(0.168006673)
    c2  := f32x8(0.499999940)
    c3  := f32x8(0.999956906)
    c4  := f32x8(0.999999642)

    t = simd.mul(x, l2e)
    r = simd.nearest(t)

    f = simd.fma(r, l2h, x)
    f = simd.fma(r, l2l, f)

    i = transmute(i32x8)x86._mm256_cvtps_epi32(t)

    p = c0
    p = simd.fma(p, f, c1)
    p = simd.fma(p, f, c2)
    p = simd.fma(p, f, c3)
    p = simd.fma(p, f, c4)

    j = simd.shl(i, u32x8(23))
    r = transmute(f32x8) simd.add(j, transmute(i32x8)p)

    return r
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

softmax :: proc(x, y: SMat) {
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

    fmt.println(maxes)


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

    /*DEBUG*/
    fmt.println(sums)
    lowest, highest, mean: f32
    matches: int
    for r in 0..<rows {
        for c in 0..<cols {
            gre := maxes[c]
            a := y.data[r*cols + c] 
            b := math.exp(x.data[r*cols + c] - gre)

            diff := math.abs(a - b)
            mag := max(math.abs(a), math.abs(b))

            rel_err := diff / mag
            lowest, highest = min(lowest, rel_err), max(highest, rel_err)
            mean += rel_err

            if rel_err < 0.01 {
                matches += 1
            } else {
                fmt.printf("%v -> %v | ", a, b)
            }
        }
    }
    mean /= f32(rows * cols)
    fmt.printfln("Lowest: %f\nHighest: %f\nMean: %f\nMatches: %v/%v", lowest, highest, mean, matches, rows * cols)
    /*END DEBUG*/
    
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

    /*DEBUG*/
    sums2 := make_aligned([]f32, cols, 32, context.temp_allocator)
    for r in 0..<rows {
        for c in 0..<cols {
            sums2[c] += y.data[r*cols+c]
        }
    }

    fmt.println(sums2)
    /*END DEBUG*/

}

/*

softmax(x, y):
    find max x -> max
    y = exp(x - max)
    sum = sum(y)
    y = y / sum



[
1 2 3 4 5
2 3 4 5 6
3 4 5 6 7
]
*/