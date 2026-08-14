package function

import "core:simd"
import "core:mem"
import "core:slice"
import bench "./pkg"
import mat "../src/mat"
import nn "../src/network"
import "core:math/rand"

SMat :: mat.SMat

main :: proc() {
    rand.reset(67)
    a := mat.rand_mat(800, 1920)
    b := mat.new_smat(800, 1920)
    defer { mat.delete_smat(a); mat.delete_smat(b) }
    data := [2]SMat{a, b}
    bench.time_it("relu_scalar", 100, raw_data(data[:]), relu)
    mem.zero_slice(b.data)
    bench.time_it("relu_simd", 100, raw_data(data[:]), relu_simd)
    mem.zero_slice(b.data)
}

relu_simd :: proc(data: rawptr) {
    data := ([^]SMat)(data)[:2]
    #force_inline nn.relu(data[0], data[1])
}

relu :: proc(data: rawptr) {
    data := ([^]SMat)(data)[:2]
    for val, i in data[0].data do data[1].data[i] = max(val, 0.0)
}

diff_relu_simd :: proc(data: rawptr) {
    data := ([^]SMat)(data)[:2]
    #force_inline nn.diff_relu(data[0], data[1])
}

diff_relu :: proc(data: rawptr) {
    data := ([^]SMat)(data)[:2]
    for val, i in data[0].data do data[1].data[i] = val > 0.0 ? 1.0 : 0.0
}





