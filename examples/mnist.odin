package mnist

import "core:slice"
import "core:os"
import "core:fmt"
import "core:log"
import "core:terminal/ansi"
import "core:strings"
import "core:sys/windows"
import nn "../src/network"
import mat "../src/mat"

LayerDef :: nn.LayerDef
Example  :: nn.Example
SMat     :: mat.SMat

BATCH_SIZE :: u32(400)
EPOCHS     :: 1000
ETA        :: 0.03
EVAL_EPOCH :: 50

main :: proc() {
    when ODIN_OS == .Windows { // Make terminal display stdout as utf8
        windows.SetConsoleOutputCP(windows.CODEPAGE.UTF8)
    }

    // Enable matrix formatting
    fmt.set_user_formatters(new(map[typeid]fmt.User_Formatter))
    fmt.register_user_formatter(SMat, mat.Matrix_Formatter)

    // Console logging - needed to log current loss/cost during training
    console_logger := log.create_console_logger(.Debug, { .Terminal_Color, .Level, .Time, .Procedure })
    defer log.destroy_console_logger(console_logger)

    context.logger = console_logger

    net := new(nn.Network)

    layers: [4]LayerDef = {  // Network structure: 4 layer FCNN
        {784, .Null},
        {16, .ReLU},
        {16, .ReLU},
        {10, .SoftMax}
    }

    // Initialize network
    nn.create_network(net, .CrossEntropy, .Dist, uint(BATCH_SIZE), expand_values(layers))
    defer nn.destroy_network(net)

    // Parse training data and testing data
    images, labels, num_images, image_rows, image_cols := read_dataset("resources/train-images.idx3-ubyte", "resources/train-labels.idx1-ubyte")
    test_images, test_labels, test_num_images := read_dataset("resources/t10k-images.idx3-ubyte", "resources/t10k-labels.idx1-ubyte")

    test_data := make_example(test_images, test_labels, uint(image_rows*image_cols), 0, BATCH_SIZE)

    num_examples := (num_images + BATCH_SIZE - 1)/BATCH_SIZE
    num_examples_full := num_images / BATCH_SIZE

    // Build training batches
    batches := make([dynamic]Example, 0, num_examples)
    {
        i: u32 = 0
        for ; i < num_examples_full; i += 1 {
            append(&batches, make_example(images, labels, uint(image_rows*image_cols), uint(i*BATCH_SIZE), uint(BATCH_SIZE)))
        }
        // Remainder batch
        if num_examples != num_examples_full {
            append(&batches, make_example(images, labels, uint(image_rows*image_cols), uint(i*BATCH_SIZE), uint(num_images % BATCH_SIZE)))
        }
    }


    // SGD training loop
    last_batch_cost := f64(1000.0)
    for i in 0..<EPOCHS {
        max_cost, min_cost, avg_cost: f64 = 0.0, 10000000000.0, 0.0
        for batch in batches {
            nn.clear_accumulators(net)
            last_batch_cost = nn.train_batch(net, batch, ETA)
            max_cost = max(max_cost, last_batch_cost)
            min_cost = min(min_cost, last_batch_cost)
            avg_cost += last_batch_cost
        }
        fmt.printfln("Epoch %v [Min=%f, Max=%f, Avg=%f]", i, min_cost, max_cost, avg_cost / f64(len(batches)))
        if (i + 1) % EVAL_EPOCH == 0 {
            output := nn.forward_prop(net, test_data.input)
        }
    }
}

read_dataset :: proc(images_path: string, labels_path: string) -> (images, labels: []u8, num_images, image_rows, image_cols: u32) {
    labels_file, err := os.open(labels_path)
    images_file, err2 := os.open(images_path)

    label_meta: [2]u32be
    image_meta: [4]u32be

    {
        os.read(labels_file, slice.to_bytes(label_meta[:]))
        fmt.println(label_meta)
        labels = make([]u8, u32(label_meta[1]))
        os.read(labels_file, labels[:])
    }

    {
        os.read(images_file, slice.to_bytes(image_meta[:]))
        fmt.println(image_meta)
        num_images, image_rows, image_cols = u32(image_meta[1]), u32(image_meta[2]), u32(image_meta[3])
        images = make([]u8, num_images * image_rows * image_cols)
        os.read(images_file, images[:])
    }

    return   
}

make_example :: proc(images: []u8, labels: []u8, image_size, start, length: uint) -> Example {
    input := mat.new_smat(image_size, length)
    output := mat.new_smat(10, length)

    // Populate input matrix - each column is an image(28x28)
    // Populate output matrix - one-hot column for each label
    for i in start..<start+length {
        output.data[uint(labels[i]) * length + (i - start)] = 1.0
        for pixel, idx in images[i*image_size:][:image_size] {
            input.data[uint(idx) * length + (i - start)] = f32(pixel) / 255.0
        }
    }

    return { input, output }
}

delete_example :: proc(exp: ^Example) {
    mat.delete_smat(exp.input)
    mat.delete_smat(exp.output)
}

print_image :: proc(images: SMat, image_idx: uint, rows, cols: uint) {
    img := strings.builder_make_len_cap(0, 18060)

    for r in 0..<rows {
        for c in 0..<cols {
            pixel_idx := r*cols+c
            pixel := u32(images.data[pixel_idx*images.cols + image_idx] * 255)
            fmt.sbprintf(&img, ansi.CSI + ansi.FG_COLOR_24_BIT + ";%v;%v;%v" + ansi.SGR + "█", pixel, pixel, pixel)
        }
        strings.write_byte(&img, '\n')
    }
    fmt.sbprintln(&img, ansi.CSI + ansi.RESET + ansi.SGR)

    fmt.print(strings.to_string(img))
}