// This file contains some of my initial matmul implementations. The vectorized variants are here as well.
// In particular, they assume that the second input matrix is right column major
// So no, this is not a SIMDized reference counting implementation, whatever that could mean
//
// :^)

package mat

import "core:mem"
import "core:simd"

/* 
    Matmul right column major
*/
@(fast_math={.No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_rc :: proc(self, other, out: SMat) {
    for r in uint(0)..<self.rows {
        for c in uint(0)..<other.cols { // c(r, c) = a(r) . b(c)
            for k in uint(0)..<self.cols {
                out.data[r * out.cols + c] += self.data[r * self.cols + k] * other.data[c * other.rows + k]
            }
        }
    }
}

// Like `smat_matmul_rc_aligned` but without the alignment restrictions.
@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_rc_simd :: proc(A, B, out: SMat) {
    num_vecs := A.cols >> 3       // Number of full (256 bit) vectors per row
    num_elems := num_vecs << 3    // Number of row elements excluding the tail not in a vector
    rem := A.cols & 7            // Number of row elements in a vector

    for r in 0..<A.rows {
        for c in 0..<B.cols {
            a_row, b_col: []f32
            a_row = A.data[r * A.cols:][:num_elems]
            b_col = B.data[c * B.rows:][:num_elems]
            
            acc: f32x8
            for i := uint(0); i < num_elems; i += 8 {
                // The start of the matrix allocation is guaranteed 32-bit aligned but the individual rows
                // are not. We use unaligned loads here due to that. Note the proc used is `simd_from_slice` not
                // `simd.from_slice`. The former explicity performs an unaligned load.
                // a_row[i:] and b_col[i:] would work fine here but I've left the exact slice for clarity
                a_vec := simd_from_slice(f32x8, a_row[i:i+8])
                b_vec := simd_from_slice(f32x8, b_col[i:i+8])
                acc = simd.add(acc, simd.mul(a_vec, b_vec))
            }

            out.data[r * out.cols + c] += simd.reduce_add_bisect(acc)

            for i in num_elems..<(num_elems + rem) {
                out.data[r * out.cols + c] += A.data[r * A.cols + i] * B.data[c * B.rows + i]
            }
        }
    }
}

// Matmul but each row must be aligned to the vector width
// 
// Aligning the start of the data is not sufficient as this doesn't guarantee
// that each row starts at an aligned address since row lengths will not necessarily
// be multiples of the vector width
@(fast_math={.Allow_Reassoc, .No_NaNs, .No_Infs, .No_Signed_Zeros})
smat_matmul_rc_simd_aligned :: proc(A, B, out: SMat) {
    num_vecs := A.cols >> 3             // Number of full (256 bit) vectors per row
    num_elems := A.cols & ~uint(7)      // Number of row elements excluding the tail not in a vector
    rem := A.cols & 7                   // Number of row elements in a vector

    for r in 0..<A.rows {
        for c in 0..<B.cols {
            a_row, b_col: []f32x8
            a_row = mem.slice_data_cast([]f32x8, A.data[r * A.cols:][:num_elems])
            b_col = mem.slice_data_cast([]f32x8, B.data[c * B.rows:][:num_elems])
            
            acc: f32x8
            for n in 0..<num_vecs {
                acc = simd.add(acc, simd.mul(a_row[n], b_col[n]))
            }

            out.data[r * out.cols + c] += simd.reduce_add_bisect(acc)

            for i in num_elems..<(num_elems + rem) {
                out.data[r * out.cols + c] += A.data[r * A.cols + i] * B.data[c * B.rows + i]
            }
        }
    }
}