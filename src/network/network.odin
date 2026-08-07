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