package decoder


/*
This is an example of a 6-bit decoder. The network has 6 inputs representing a 6-bit
binary number(call it N), and 64 outputs. The (N+1)th output should be set(1.0).
*/

import "core:fmt"
import "core:log"
import nn "../src/network"
import mat "../src/mat"

LayerDef :: nn.LayerDef
Example  :: nn.Example
SMat     :: mat.SMat

main :: proc() {
    // Enable matrix formatting
    fmt.set_user_formatters(new(map[typeid]fmt.User_Formatter))
    fmt.register_user_formatter(SMat, mat.Matrix_Formatter)

    // Console logging - needed to log current loss/cost during training
    console_logger := log.create_console_logger(.Debug, { .Terminal_Color, .Level, .Time, .Procedure })
    defer log.destroy_console_logger(console_logger)

    context.logger = console_logger

    net := new(nn.Network)

    layers: [3]LayerDef = {  // Network structure: 3 layer FCNN
        {6, .Null},
        {38, .ReLU},
        {64, .SoftMax}
    }

    nn.create_network(net, .CrossEntropy, .Dist, 64, expand_values(layers))
    defer nn.destroy_network(net)

    dataset: Example
    create_dataset(&dataset)
    defer destroy_dataset(&dataset)


    for i in 0..<400 {
        nn.clear_accumulators(net)
        nn.train_batch(net, dataset, 0.1)
        // Break early if batch_cost is sufficiently low
        batch_cost := net.cost_proc(net.layers[len(net.layers)-1].a, dataset.output) / f64(dataset.output.cols)
        if batch_cost < 0.001 do break
    }

    fmt.printfln("%#v", nn.forward_prop(net, dataset.input))
}

print_mat :: proc(a: ^SMat) {
    fmt.printf("%dx%d\n", a.rows, a.cols)
    for i in 0..<a.rows {
        for j in 0..<a.cols {
            fmt.printf("%6.2f ", a.data[i * a.cols + j])
        }
        fmt.print("\n")
    }
}

// Generate a 6x64 input matrix -> 64 input vectors (all 6-bit numbers)
// and a 64z64 output matrix -> 64 output vectors (each one with corresponding Nth element set)
create_dataset :: proc(example: ^Example) {
    example.input = mat.new_smat(6, 64)
    example.output = mat.new_smat(64, 64)

    for i in 0..<64 {
        example.output.data[i * 64 + i] = 1.0

        j := i
        for n in 0..<6 {
            example.input.data[(5 - n) * 64 + i] = f32(j & 1)
            j >>= 1
        }
    }
}

destroy_dataset :: proc(dataset: ^Example) {
    mat.delete_smat(dataset.input)
    mat.delete_smat(dataset.output)
}