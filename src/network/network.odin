package network

import mat "../mat"

SMat :: mat.SMat
matmul :: mat.smat_matmul_blocking

Layer :: struct {
    w, b, z, a, 
    dw, db, acc_dw, acc_db: SMat,
    act_fn: ActFn
}

ActFn :: #type proc(input, output: SMat)

Network :: struct {
    x: SMat,
    temp: SMat, 
    layers: [dynamic]Layer
}

forward_prop :: proc(net: ^Network, input: SMat) -> SMat {
    assert(len(net.layers) > 0, "Must have an output layer")
    mat.copy_smat_to(input, net.x)
    matmul(net.layers[0].w, input, net.layers[0].z)
    // TODO: perhaps a "broadcast_copy" might be a better
    // mat.copy_smat_to(net.layers[0].b, net.layers[0].z)
    broadcast_add(net.layers[0].z, net.layers[0].b)
    net.layers[0].act_fn(net.layers[0].z, net.layers[0].a)

    i: int
    for i = 1; i < len(net.layers); i += 1 {
        prev, current := &net.layers[i - 1], &net.layers[i]
        matmul(current.w, prev.a, current.z)
        broadcast_add(current.z, current.b)
        current.act_fn(current.z, current.a)
    }

    return net.layers[i].a
}

backward_prop :: proc(net: ^Network, target: SMat) {
    assert(len(net.layers) > 0, "Must have an output layer")

    num_layers := len(net.layers)
    out_layer := &net.layers[num_layers - 1]
    prev_layer := &net.layers[num_layers - 2]

    // Calculate output layer gradient
    mem.zero_slice(out_layer.dw.data)
    net.output_grad_proc(target, out_layer.db)

    // Calculate output layer weight gradients
    mat.reshape(&net.temp, prev_layer.a.cols, prev_layer.a.rows)
    mat.smat_transpose(prev_layer.a, &net.temp)
    matmul(out_layer.db, net.temp, out_layer.dw)

    // Zero out gradients - `matmul` adds(not overwrites) its result to output
    for &layer in net.layers {
        mem.zero_slice(layer.dw.data)
        mem.zero_slice(layer.db.data)
    }

    // Each iteration computes net.layers[i - 1] or `prev`'s gradients
    // net.layers doesn't hold the input layer so the pre_prev, prev, current won't work
    // (without some checks) so I handle it just after the loop
    for i := num_layers - 1; i > 1; i -= 1 {
        pre_prev, prev, current := &net.layers[i - 2], &net.layers[i - 1], &net.layers[i]
        compute_layer_gradients(net, prev, current, pre_prev.a)
    }

    // Compute the first HIDDEN layer's gradients
    compute_layer_gradients(net, &net.layers[0], &net.layers[1], net.x)
}

// This actually only computes the `prev`'s gradients
compute_layer_gradients :: #force_inline proc(net: ^Network, prev, current: ^Layer, pre_prev_act: SMat) {
    mat.reshape(&net.temp, current.w.cols, current.w.rows)
    mat.smat_transpose(current.w, &net.temp)
    matmul(net.temp, current.db, prev.db) // db_(l-1) = (W_l)^T . db_l

    mat.reshape(&net.temp, prev.a.rows, prev.a.cols)
    prev.act_fn(prev.z, net.temp)   // temp = f_(l-1)'(z_(l-1))
    mat.smat_mul(prev.db, net.temp) // db_(l-1) = temp * db_(l-1) 

    mat.reshape(&net.temp, pre_prev_act.cols, pre_prev_act.rows)
    mat.smat_transpose(pre_prev_act, &net.temp)
    matmul(prev.db, net.temp, prev.dw)  // dw_(l-1) = db_(l-1) . (a_(l-2))^T
}