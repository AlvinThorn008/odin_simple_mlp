package mat

import "core:simd"

@(private)
Operation :: enum {
    Add,
    Sub,
    Mul
}

smat_add :: proc(self, other: SMat) { smat_elementwise(self, other, .Add) }
smat_sub :: proc(self, other: SMat) { smat_elementwise(self, other, .Sub) }
smat_mul :: proc(self, other: SMat) { smat_elementwise(self, other, .Mul) }

smat_scale :: proc(self: SMat, scalar: f32) {
    smat_scale_inner :: #force_inline proc(#no_alias a: [^]f32, size: int, scalar: f32) {
        for i := 0; i < size; i += 1 {
            a[i] *= scalar
        }
    }
    smat_scale_inner(raw_data(self.data), len(self.data), scalar)
}

smat_elementwise :: #force_inline proc(self, other: SMat, $OP: Operation) {
    // Had to vectorize this manually as auto-vec wouldn't occur without
    // #no_alias. This version still requires that self and other's data don't
    // alias with the exception of self.data == other.data
    assert(smat_same_size(self, other), "size mismatch: could not add matrices")
    
    t0, t1, t2, t3: f32x8

    a := ([^]f32x8)(raw_data(self.data))
    b := ([^]f32x8)(raw_data(other.data))

    op :: #force_inline proc(a, b: #simd[$N]$T, $I: Operation) -> #simd[N]T {
        when I == .Add do return simd.add(a, b)
        else when I == .Sub do return simd.sub(a, b)
        else when I == .Mul do return simd.mul(a, b)
    }

    i := 0
    length := len(self.data)
    size8 := length / 8
    for ; i + 3 < size8; i += 4 {
        t0 = op(a[i], b[i], OP)
        t1 = op(a[i + 1], b[i + 1], OP)
        t2 = op(a[i + 2], b[i + 2], OP)
        t3 = op(a[i + 3], b[i + 3], OP)

        a[i] = t0
        a[i + 1] = t1
        a[i + 2] = t2
        a[i + 3] = t3
    }
    
    for j in 0..<3 {
        if i >= size8 do break
        a[i] = op(a[i], b[i], OP)
        i += 1
    }

    i *= 8
    for k in 0..<7 {
        if i >= length do break
        when OP == .Add do a[i] += b[i]
        else when OP == .Sub do a[i] -= b[i]
        else when OP == .Mul do a[i] *= b[i]

        i += 1
    }
}