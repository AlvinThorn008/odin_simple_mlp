package mat

import "base:intrinsics"
import "core:mem"
import "core:math/rand"
import "core:math"
import "core:os"
import "core:fmt"

rand_mat :: proc(rows, cols: uint) -> SMat {
	out := new_smat(rows, cols)

	for &val in out.data {
		val = rand.float32_range(-1000.0, 1000.0)
	}
	return out
}

// Random matrix but column major because I didn't
// wanna use a transpose
rand_mat_cm :: proc(rows, cols: uint) -> SMat {
	out := new_smat(rows, cols)

    for r in 0..<out.rows {
        for c in 0..<out.cols {
            out.data[c * out.rows + r] = rand.float32_range(-1000, 1000.0)
        }   
    }
	return out
}

// Convert slice to SIMD vector.
//
// This method works the same as `simd.from_slice` except it's guaranteed
// to use an unaligned load to perform the copy. 
// 
// It is likely that `simd.from_slice` will generate similar assembly after optimization
// but I didn't check and now I don't have to. 
@(require_results)
simd_from_slice :: #force_inline proc($T: typeid/#simd[$LANES]$E, slice: []E) -> T {
    assert(len(slice) >= LANES, "slice length must be a least the number of lanes")
    return intrinsics.unaligned_load((^T)(raw_data(slice)))
}

// Source: https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
smat_rel_compare :: proc(a, b: SMat, rel_tol := 0.05) -> uint {
    matches := uint(0)
    for i in 0..<len(a.data) {
        diff := math.abs(a.data[i] - b.data[i])

        if (diff <= 0.01) {
            matches += 1
            continue
        }

        a_abs := math.abs(a.data[i])
        b_abs := math.abs(b.data[i])
        max := math.max(a_abs, b_abs)

        if (f64(diff) <= f64(max) * rel_tol) { matches += 1 }

    }
    return matches
}

// Export matrix to a file
// ```
// Format: ROWS COLS D0 D1 D2 ...
// ``` 
export_smat :: proc(mat: SMat, name: string) {
    header_size := size_of(mat.rows) + size_of(mat.cols)
    data_size := len(mat.data) * size_of(f32)
    buf := make([]u8, header_size + data_size) // [ (uint)rows, (uint)cols, (f32)data[0], (f32)data[1], ... ]
    defer delete(buf)

    header := [2]uint { mat.rows, mat.cols }
    header_bytes := mem.slice_to_bytes(header[:])
    data_bytes := mem.slice_to_bytes(mat.data)

    copy(buf, header_bytes)
    copy(buf[header_size:], data_bytes)

    err := os.write_entire_file_from_bytes(name, buf)
    if err != nil { fmt.eprintfln("IO Error: %v", os.error_string(err)) }
}

// Read the rows and cols data of an exported matrix
read_smat_header :: proc(name: string) -> (rows, cols: uint, file: ^os.File, err: os.Error) {
    file = os.open(name) or_return

    header: [2 * size_of(uint)]u8
    n := os.read(file, header[:]) or_return

    dims := mem.slice_data_cast([]uint, header[:])

    return dims[0], dims[1], file, os.General_Error.None
}

// Import a matrix from file. Meant to be used to `read_smat_header`
//
// Really just a utility for loading `naive.bin` (See main.odin for details)
import_smat_with_meta :: proc(rows, cols: uint, file: ^os.File) -> (mat: SMat) {
    mat = new_smat(rows, cols)
    n, err := os.read(file, mem.slice_to_bytes(mat.data))

    if err != nil { fmt.eprintfln("IO Error: %v", os.error_string(err)) }

    return
}



