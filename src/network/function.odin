package network

import "core:simd"
import "core:math"
import smat "../mat"

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


softmax :: proc(mat, out: SMat) {
    assert(smat.smat_same_size(mat, out), "size mismatch: softmax is an element-wise operation")
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

cross_entropy :: proc(a, b: SMat) -> f64 {
    assert(smat.smat_same_size(a, b), "size mismatch: matrices must be the same size")

    // Cross entropy is normally computed with vectors but it can be useful to compute
    // it over one axis of same size matrices.
    // Just saying, I'm not doing that here atm

    sum := f64(0.0)
    for i in 0..<len(a.data) do sum -= f64(b.data[i]) * math.ln(f64(max(a.data[i], 1e-7)))

    return sum
}

squared_error :: proc(a, b: SMat) -> f64 {
    assert(smat.smat_same_size(a, b), "size mismatch: matrices must be the same size")
    sum: f64 = 0.0
    for i := 0; i < len(a.data); i += 1 {
        diff := f64(a.data[i]) - f64(b.data[i])
        sum += diff * diff
    }
    return sum
}

dsquared_error :: proc(a, b, out: SMat) {
    assert(smat.smat_same_size(a, b), "size mismatch: matrices must be the same size")
    for i := 0; i < len(a.data); i += 1 {
        out.data[i] = 2.0 * (a.data[i] - b.data[i])
    }
}

sigmoid :: proc(self, out: SMat) {
    assert(smat.smat_same_size(self, out), "size mismatch: matrices must be the same size")
    for i := 0; i < len(self.data); i += 1 {
        out.data[i] = 1.0 / (1.0 + math.exp_f32(-self.data[i]))
    }
}

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

