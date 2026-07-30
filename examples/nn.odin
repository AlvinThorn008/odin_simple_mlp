package nn

import "core:math"
import "core:mem"
import mat "../src/mat"
import "core:simd"
import "base:intrinsics"

main :: proc() {
    
}

Network :: struct {
    w1, w2, b1, b2, z1, z2, a1, a2: mat.SMat,
}

matmul :: mat.smat_matmul_blocking
SMat :: mat.SMat

forward_prop :: proc(net: ^Network, input: SMat) {
    // Z_ahead = W Z + B

    mat.copy_smat_to(net.b1, net.z1)
    mat.copy_smat_to(net.b2, net.z2)

    matmul(net.w1, input, net.z1)
    relu(net.z1, net.a1)
    matmul(net.w2, net.a1, net.z2)
    relu(net.z2, net.a2)
    // matmul()
}

relu :: proc(mat, out: SMat) {
    zero := simd.f32x8(0.0)
    len_vecs := len(mat.data) & ~int(7)
    i := 0
    for ; i < len_vecs; i += 8 {
        a := (^simd.f32x8)(&mat.data[i])
        b := (^simd.f32x8)(&out.data[i])
        b^ = simd.max(a^, zero)
    }
    for j in i..len(mat.data) do out.data[j] = max(mat.data[j], 0.0)
}

softmax :: proc(mat, out: SMat) {
    max_val := -math.F32_MAX
    len_vecs := len(mat.data) & ~int(7)

    i := 0
    for ; i < len_vecs; i += 8 {
        a := (^simd.f32x8)(&mat.data[i])
        max_val = max(simd.reduce_max(a), max_val)
    }
}



