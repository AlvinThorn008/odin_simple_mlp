#+private
package mat_test

import mat "../../mat"

import "core:testing"
import "core:log"

SMat :: mat.SMat

@(test)
transpose_in_block :: proc(t: ^testing.T) {
    a : = mat.new_smat(6, 9)
    a_trans := mat.new_smat(9, 6)
    defer mat.delete_smat(a)
    defer mat.delete_smat(a_trans)

    // Init with index as element value
    for &val, i in a.data do val = f32(i)

    b_arr := [9][6]f32 {
        {0, 9, 18, 27, 36, 45},
        {1, 10, 19, 28, 37, 46},
        {2, 11, 20, 29, 38, 47},
        {3, 12, 21, 30, 39, 48},
        {4, 13, 22, 31, 40, 49},
        {5, 14, 23, 32, 41, 50},
        {6, 15, 24, 33, 42, 51},
        {7, 16, 25, 34, 43, 52},
        {8, 17, 26, 35, 44, 53}
    }
    b := mat.new_smat_2d_array(&b_arr)

    mat.smat_transpose(a, &a_trans)

    testing.expect(t, mat.equal(a_trans, b))
}

@(test)
transpose_outside_block :: proc(t: ^testing.T) {
    a := mat.new_smat(234, 136)
    a_trans := mat.new_smat(136, 234)
    b := mat.new_smat(136, 234)
    defer mat.delete_smat(a)
    defer mat.delete_smat(a_trans)
    defer mat.delete_smat(b)

    // Init with index as element value
    for &val, i in a.data do val = f32(i)
    for i := uint(0); i < b.rows; i += 1 {
        for j := uint(0); j < b.cols; j += 1 do b.data[i * b.cols + j] = f32(i + b.rows * j)
    }

    mat.smat_transpose(a, &a_trans)

    testing.expect(t, mat.equal(a_trans, b))
}