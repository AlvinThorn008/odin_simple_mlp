package network

import "core:simd"
import "base:intrinsics"
import mat "../mat"

f32x8 :: simd.f32x8

/*
Custom matrix operations to accelerate batched NN training
*/


// Add a column vector to a matrix by the vector to each column of the matrix
//
// NOTE: This procedure is optimized for matrix-vector addition. If `self` is a column vector, it 
broadcast_add :: proc(self, column: SMat) {
    assert(column.cols == 1, "column must be a column vector")
    assert(self.rows == column.rows, "Row dimension must match")

    t0, t1, t2, t3: f32x8

    vec_rows := column.rows
    mat_cols := self.cols

    // Degenerate case: self is a column vector
    // `smat_add` is vectorized whereas (1) will perform scalar addition
    // in this degenerate case.
    if self.cols == 1 { mat.smat_add(self, column); return }

    // (1)
    for i := uint(0); i < vec_rows; i += 1 {
        a := f32x8(column.data[i])
        c: uint
        for c = 0; c + 31 < mat_cols; c += 32 {
            // NOTE: Matrix rows are not guaranteed to be aligned
            // so unaligned access is used here
            t0 = mat.simd_from_slice(f32x8, self.data[i * self.cols + c:])
            t1 = mat.simd_from_slice(f32x8, self.data[i * self.cols + c + 8:])
            t2 = mat.simd_from_slice(f32x8, self.data[i * self.cols + c + 16:])
            t3 = mat.simd_from_slice(f32x8, self.data[i * self.cols + c + 24:])

            t0 = simd.add(t0, a)
            t1 = simd.add(t1, a) 
            t2 = simd.add(t2, a)
            t3 = simd.add(t3, a)

            intrinsics.unaligned_store((^f32x8)(&self.data[i * self.cols + c]), t0)
            intrinsics.unaligned_store((^f32x8)(&self.data[i * self.cols + c + 8]), t1)
            intrinsics.unaligned_store((^f32x8)(&self.data[i * self.cols + c + 16]), t2)
            intrinsics.unaligned_store((^f32x8)(&self.data[i * self.cols + c + 24]), t3)
        }
        for ; c + 7 < mat_cols; c += 8 {
            t0 = mat.simd_from_slice(f32x8, self.data[i * self.cols + c:])
            t0 = simd.add(t0, a)
            intrinsics.unaligned_store((^f32x8)(&self.data[i * self.cols + c]), t0)
        }
        for ; c < mat_cols; c += 1 {
            self.data[i * self.cols + c] += column.data[i] 
        }
    } 
}

/*
Reduce matrix `self` into column vector `column` by summing the columns of
`self`
*/
reduce_add :: proc(self, column: SMat) {
    assert(column.cols == 1, "column must be a column vector")
    assert(self.rows == column.rows, "Row dimension must match")

    if self.cols == 1 { mat.copy_smat_to(self, column); return }

    t0, t1, t2, t3: f32x8

    for r := uint(0); r < self.rows; r += 1 {
        cols32 := self.cols & ~uint(31)
        cols8  := self.cols & ~uint(7)
        row_sum: f32 = 0.0
        c: uint
        for c = 0; c < cols32; c += 32 {
            // NOTE: Matrix rows are not guaranteed to be aligned
            // so unaligned access is used here
            t0 = mat.simd_from_slice(f32x8, self.data[r * self.cols + c:])
            t1 = mat.simd_from_slice(f32x8, self.data[r * self.cols + c + 8:])
            t2 = mat.simd_from_slice(f32x8, self.data[r * self.cols + c + 16:])
            t3 = mat.simd_from_slice(f32x8, self.data[r * self.cols + c + 24:])

            t0 = simd.add(t0, t1)
            t2 = simd.add(t2, t3) 
            t0 = simd.add(t0, t2)

            row_sum += simd.reduce_add_bisect(t0)
        }
        for ; c < cols8; c += 8 {
            t0 = mat.simd_from_slice(f32x8, self.data[r * self.cols + c:])
            row_sum += simd.reduce_add_bisect(t0)
        }
        for ; c < self.cols; c += 1 {
            row_sum += self.data[r * self.cols + c]
        }
        column.data[r] = row_sum
    }
}