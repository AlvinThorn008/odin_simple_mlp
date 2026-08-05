package nn

import "core:fmt"
import "core:math"
import "core:mem"
import smat "../src/mat"
import "core:simd"
import "core:math/rand"


// Configuration

NN_DEBUG :: #config(NN_DEBUG, false)

matmul :: smat.smat_matmul_blocking
SMat :: smat.SMat

main :: proc() {
    rand.reset(69_69_69)

    net: Network
    create_network(&net)
    defer destroy_network(&net)

    dataset: [16]Example
    create_dataset(&dataset)
    defer destroy_dataset(&dataset)

    train(&net, dataset[:], 1.21, len(dataset), 250)
}

// A 3 layer FCNN 
Network :: struct {
    w1, w2,   // Weights
    b1, b2,   // Bias
    z1, z2,   // Unactivated layers (stored for backward_prop)
    a1, a2,   // Activatived layers
    dw1, dw2, // Weight gradients
    db1, db2, // Bias gradients
    x,        // Network input

    acc_dw1, acc_dw2,
    acc_db1, acc_db2: smat.SMat,

    // Resizable temp matrix for out-of-place operations and temporaries
    mT: smat.DynSMat,
}

// Initialize a network
//
// Allocates correctly sized matrices for all fields
create_network :: proc(net: ^Network) {
    /*
    4 16 16

    x: 4x1
    w1, dw1: 16x4
    w2, dw2: 16x16

    z1, a1, b1, db1: 16x1 
    z2, a2, b2, db2: 16x1

    Require: data allocation to be aligned to 32 bytes
    */

    net.x       = smat.new_smat(4, 1)
    net.w1      = smat.new_smat(16, 4)
    net.dw1     = smat.new_smat(16, 4)
    net.acc_dw1 = smat.new_smat(16, 4)
    net.w2      = smat.new_smat(16, 16)
    net.dw2     = smat.new_smat(16, 16)
    net.acc_dw2 = smat.new_smat(16, 16)
    net.z1      = smat.new_smat(16, 1)
    net.a1      = smat.new_smat(16, 1)
    net.b1      = smat.new_smat(16, 1)
    net.db1     = smat.new_smat(16, 1)
    net.acc_db1 = smat.new_smat(16, 1)
    net.z2      = smat.new_smat(16, 1)
    net.a2      = smat.new_smat(16, 1)
    net.b2      = smat.new_smat(16, 1)
    net.db2     = smat.new_smat(16, 1)
    net.acc_db2 = smat.new_smat(16, 1)
    // Capacity of mT = size of biggest matrix
    net.mT      = smat.new_dyn_smat(16, 16, 16*16)

    for &val in net.w1.data do val = rand.float32_normal(0.0, 2.0/4.0)
    for &val in net.w2.data do val = rand.float32_normal(0.0, 2.0/4.0)
    for &val in net.b1.data do val = rand.float32_normal(0.0, 2.0/4.0)
    for &val in net.b2.data do val = rand.float32_normal(0.0, 2.0/4.0)
}

// Delete all network matrices
destroy_network :: proc(net: ^Network) {
    smat.delete_smat(net.x)
    smat.delete_smat(net.w1)
    smat.delete_smat(net.dw1)
    smat.delete_smat(net.w2)
    smat.delete_smat(net.dw2)
    smat.delete_smat(net.z1)
    smat.delete_smat(net.a1)
    smat.delete_smat(net.b1)
    smat.delete_smat(net.db1)
    smat.delete_smat(net.z2)
    smat.delete_smat(net.a2)
    smat.delete_smat(net.b2)
    smat.delete_smat(net.db2)
    smat.delete_smat(net.mT)
}


forward_prop :: proc(net: ^Network, input: SMat) -> SMat {
    smat.copy_smat_to(input, net.x)

    smat.copy_smat_to(net.b1, net.z1) // z1 = b1
    smat.copy_smat_to(net.b2, net.z2)

    // Hidden layer a1 = relu(z1) = relu(w1.x + b1)
    matmul(net.w1, net.x, net.z1) // z1 += w1.x
    relu(net.z1, net.a1)

    // Output layer a2 = softmax(z2) = softmax(w2.a1 + b2)
    matmul(net.w2, net.a1, net.z2)
    softmax(net.z2, net.a2)

    return net.a2
}

backward_prop :: proc(net: ^Network, target: SMat) {
    // db2 softmax + cross entropy
    // db2 = a2 - target
    smat.copy_smat_to(net.a2, net.db2)
    smat.smat_sub(net.db2, target)

    // dw2 = db2 . a1^T
    smat.reshape(&net.mT, net.a1.cols, net.a1.rows)
    smat.smat_transpose(net.a1, &net.mT)
    matmul(net.db2, net.mT, net.dw2)

    // db1 = f1'(z1) * (W1^T . db2)
    smat.reshape(&net.mT, net.w2.cols, net.w2.rows)
    smat.smat_transpose(net.w2, &net.mT)
    matmul(net.mT, net.db2, net.db1) // db1 = W1^T . db2
    smat.reshape(&net.mT, net.a1.rows, net.a1.cols)
    diff_relu(net.z1, net.mT)        // temp = f1'(z1)
    smat.smat_mul(net.db1, net.mT)

    smat.reshape(&net.mT, net.x.rows, net.x.cols)
    smat.smat_transpose(net.x, &net.mT)
    matmul(net.db1, net.mT, net.dw1)
}

// A training example for the network
Example :: struct { input: SMat, output: SMat }

// Compute network gradients
train_batch :: proc(net: ^Network, batch: []Example, eta: f32) {
    batch_cost: f64 = 0.0
    for example in batch {
        forward_prop(net, example.input)
        backward_prop(net, example.output)
        smat.smat_add(net.acc_db1, net.db1)
        smat.smat_add(net.acc_db2, net.db2)
        smat.smat_add(net.acc_dw1, net.dw1)
        smat.smat_add(net.acc_dw2, net.dw2)
        when NN_DEBUG do batch_cost += cross_entropy(net.a2, example.output)
    }

    scale := eta / f32(len(batch))

    smat.smat_scale(net.acc_db1, scale)
    smat.smat_sub(net.b1, net.acc_db1)
    smat.smat_scale(net.acc_db2, scale)
    smat.smat_sub(net.b2, net.acc_db2)
    smat.smat_scale(net.acc_dw1, scale)
    smat.smat_sub(net.w1, net.acc_dw1)
    smat.smat_scale(net.acc_dw2, scale)
    smat.smat_sub(net.w2, net.acc_dw2)

    when NN_DEBUG do fmt.printfln("Batch cost: %f", batch_cost/f64(len(batch)))
}

train :: proc(net: ^Network, dataset: []Example, eta: f32, batch_size, epochs: uint) {
    dataset_size := uint(len(dataset))
    step := min(dataset_size, batch_size)
    total_cost := 0
    for epoch in 0..<epochs {
        i: uint
        // Read Entire dataset as batch
        // OR every batch_size slice
        for i = 0; i + step - 1 < dataset_size; i += step {
            when NN_DEBUG do fmt.printf("Epoch: %d | ", epoch)
            clear_accumulators(net)
            train_batch(net, dataset[i:i+step], eta)
        }
        // Handle remainder
        if i < dataset_size {
            when NN_DEBUG do fmt.printf("Epoch: %d | ", epoch)
            clear_accumulators(net)
            train_batch(net, dataset[i:], eta)
        }
    }
}

clear_accumulators :: proc(net: ^Network) {
    mem.zero_slice(net.acc_db1.data)
    mem.zero_slice(net.acc_db2.data)
    mem.zero_slice(net.acc_dw1.data)
    mem.zero_slice(net.acc_dw2.data)  
}

create_dataset :: proc(arr: ^[16]Example) {
    for i in 0..<16 {
        input := smat.new_smat(4, 1)
        output := smat.new_smat(16, 1)
        output.data[i] = 1.0

        j := i
        for n in 0..<4 {
            input.data[3 - n] = f32(j & 1)
            j >>= 1
        }
        arr[i] = { input, output }
    }
}

destroy_dataset :: proc(dataset: ^[16]Example) {
    for example in dataset {
        smat.delete_smat(example.input)
        smat.delete_smat(example.output)
    }
}

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
    for i in 0..<len(a.data) do sum -= f64(b.data[i]) * math.ln(f64(a.data[i]))

    return sum
}



