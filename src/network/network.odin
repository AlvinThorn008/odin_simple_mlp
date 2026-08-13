package network

import "core:log"
import "core:mem"
import "core:math/rand"
import mat "../mat"

SMat :: mat.SMat
matmul :: mat.smat_matmul_blocking

// A FCNN layer
//
// Weights are associated with the layer they feed into and as such, `Layer` doesn't model
// the input layer of a network.
Layer :: struct {
    w, b, acc_dw, acc_db, dw: SMat,
    z, a, db: mat.DynSMat,
    act_fn, diff_act_fn: ActProc
}

ActProc :: #type proc(input, output: SMat)
GradProc :: #type proc(net: ^Network, target, grad: SMat)
CostProc :: #type proc(output, target: SMat) -> f64
DcostProc :: #type proc(output, target, res: SMat)

ActFn :: enum { Null, ReLU, SoftMax, Sigmoid }
CostFn :: enum { CrossEntropy, SquaredError }
OutputType :: enum { Dist, General }

LayerDef :: struct { num_nodes: uint, activation_fun: ActFn }


Network :: struct {
    x: mat.DynSMat,
    temp: mat.DynSMat,
    layers: [dynamic]Layer,
    output_grad_proc: GradProc,
    cost_proc: CostProc,
    diff_cost_proc: DcostProc,
    max_batch_size: uint
}

create_network :: proc(net: ^Network, cost_fn: CostFn, output_type: OutputType, max_batch_size: uint, layers: ..LayerDef) {
    num_layers := len(layers)
    assert(num_layers >= 2, "At least an input and output layer must be defined")
    net.x = mat.new_dyn_smat(layers[0].num_nodes, max_batch_size)

    log.debugf("(Layer 0) Layer size: %d | X: %dx%d", net.x.rows, net.x.rows, net.x.cols)

    net.layers = make([dynamic]Layer, 0, num_layers - 1)

    use_soft_cross := cost_fn == .CrossEntropy && output_type == .Dist

    switch cost_fn {
        case .CrossEntropy:
            net.cost_proc, net.diff_cost_proc = cross_entropy, diff_cross_entropy
        case .SquaredError:
            net.cost_proc, net.diff_cost_proc = squared_error, diff_squared_error
    }

    temp_size: uint

    for i := 1; i < num_layers; i += 1 {
        acts: [2]ActProc
        switch layers[i].activation_fun {
            case .ReLU    : acts = { relu, diff_relu }
            case .Sigmoid : acts = { sigmoid, diff_sigmoid }
            case .SoftMax :
                assert(i + 1 == num_layers && use_soft_cross, "softmax currently only supported in the output layer with cross entropy and Distribution output")
                net.output_grad_proc = output_grad_cross_entropy
                acts = { softmax, softmax }           
            case .Null:
            case:
        }

        next_layer_nodes := i + 1 < num_layers ? layers[i+1].num_nodes : 0

        temp_size = max(
            temp_size,
            layers[i].num_nodes * max_batch_size, // current activation
            layers[i].num_nodes * next_layer_nodes, // next weight
            layers[i-1].num_nodes * max_batch_size // previous activation
        )

        inputs, outputs := layers[i-1].num_nodes, layers[i].num_nodes

        log.debugf("(Layer %d) Layer size: %d | Z: %dx%d | W: %dx%d | B: %dx%d | act: %v", i, outputs, outputs, 
            max_batch_size, outputs, inputs, outputs, 1, layers[i].activation_fun)

        w := mat.new_smat(outputs, inputs)
        b := mat.new_smat(outputs, 1)

        // He initialization
        for &val in w.data do val = rand.float32_normal(0.0, 2.0/f32(outputs))
        for &val in b.data do val = rand.float32_normal(0.0, 2.0/f32(outputs))

        append(&net.layers, Layer {
            w = w,
            acc_dw = mat.new_smat(outputs, inputs),
            dw = mat.new_smat(outputs, inputs),
            z = mat.new_dyn_smat(outputs, max_batch_size),
            a = mat.new_dyn_smat(outputs, max_batch_size),
            db = mat.new_dyn_smat(outputs, max_batch_size),
            b = b,
            acc_db = mat.new_smat(outputs, 1),
            act_fn = acts[0],
            diff_act_fn = acts[1]
        })
    }

    net.temp = mat.new_dyn_smat(1, temp_size)
    log.debugf("Temp matrix sized to %dx%d = %d", net.temp.rows, net.temp.cols, len(net.temp.data))
}

resize_matrices :: proc(net: ^Network, batch_size: uint) -> bool {
    if batch_size > net.max_batch_size do return false

    mat.reshape(&net.x, net.x.rows, batch_size)

    for &layer in net.layers {
        mat.reshape(&layer.z, layer.z.rows, batch_size)
        mat.reshape(&layer.a, layer.a.rows, batch_size)
        mat.reshape(&layer.db, layer.db.rows, batch_size)
    }

    return true
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
    prev_layer_act := num_layers > 1 ? net.layers[num_layers - 2].a :net.x

    // Zero out gradients - `matmul` adds(not overwrites) its result to output
    for &layer in net.layers {
        mem.zero_slice(layer.dw.data)
        mem.zero_slice(layer.db.data)
    }

    // Calculate output layer gradient
    net.output_grad_proc(net, target, out_layer.db)

    // Calculate output layer weight gradients
    mat.reshape(&net.temp, prev_layer_act.cols, prev_layer_act.rows)
    mat.smat_transpose(prev_layer_act, &net.temp)
    matmul(out_layer.db, net.temp, out_layer.dw)

    // Each iteration computes net.layers[i - 1] or `prev`'s gradients
    // net.layers doesn't hold the input layer so the pre_prev, prev, current won't work
    // (without some checks) so I handle it just after the loop
    for i := num_layers - 1; i > 1; i -= 1 {
        pre_prev, prev, current := &net.layers[i - 2], &net.layers[i - 1], &net.layers[i]
        compute_layer_gradients(net, prev, current, pre_prev.a)
    }

    // Compute the first HIDDEN layer's gradients
    if num_layers > 1 do compute_layer_gradients(net, &net.layers[0], &net.layers[1], net.x)
}

// This actually only computes the `prev`'s gradients
compute_layer_gradients :: #force_inline proc(net: ^Network, prev, current: ^Layer, pre_prev_act: SMat) {
    mat.reshape(&net.temp, current.w.cols, current.w.rows)
    mat.smat_transpose(current.w, &net.temp)
    matmul(net.temp, current.db, prev.db) // db_(l-1) = (W_l)^T . db_l

    mat.reshape(&net.temp, prev.a.rows, prev.a.cols)
    prev.diff_act_fn(prev.z, net.temp)   // temp = f_(l-1)'(z_(l-1))
    mat.smat_mul(prev.db, net.temp) // db_(l-1) = temp * db_(l-1) 

    mat.reshape(&net.temp, pre_prev_act.cols, pre_prev_act.rows)
    mat.smat_transpose(pre_prev_act, &net.temp)
    matmul(prev.db, net.temp, prev.dw)  // dw_(l-1) = db_(l-1) . (a_(l-2))^T
}

output_grad_cross_entropy :: proc(net: ^Network, target, grad: SMat) {
    mat.copy_smat_to(net.layers[len(net.layers) - 1].a, grad)
    mat.smat_sub(grad, target)
}

output_grad :: proc(net: ^Network, target, grad: SMat) {
    last_layer := &net.layers[len(net.layers) - 1]
    net.diff_cost_proc(last_layer.a, target, grad)
    last_layer.diff_act_fn(last_layer.z, last_layer.z)
    mat.smat_mul(grad, last_layer.z)
}