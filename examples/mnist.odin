package mnist

import "core:mem"
import "core:image"
import "core:slice"
import "core:os"
import "core:fmt"
import "core:log"
import "core:terminal/ansi"
import "core:strings"
import "core:sys/windows"
import "core:image/bmp"
import nn "../src/network"
import mat "../src/mat"

LayerDef :: nn.LayerDef
Example  :: nn.Example
SMat     :: mat.SMat

BATCH_SIZE :: u32(400)
EPOCHS     :: 1000
ETA        :: 0.03
LOG_EPOCH  :: 25
EVAL_EPOCH :: 100

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
    test_images, test_labels, test_num_images, _, _ := read_dataset("resources/t10k-images.idx3-ubyte", "resources/t10k-labels.idx1-ubyte")

    test_data := make_example(test_images, test_labels, image_rows*image_cols, 0, BATCH_SIZE)

    num_examples := (num_images + BATCH_SIZE - 1)/BATCH_SIZE
    num_examples_full := num_images / BATCH_SIZE

    // print_image(test_data.input, 0, 28, 28)

    // Build training batches
    batches := make([dynamic]Example, 0, num_examples)
    {
        i: u32 = 0
        for ; i < num_examples_full; i += 1 {
            append(&batches, make_example(images, labels, image_rows*image_cols, i*BATCH_SIZE, BATCH_SIZE))
        }
        // Remainder batch
        if num_examples != num_examples_full {
            append(&batches, make_example(images, labels, image_rows*image_cols, i*BATCH_SIZE, num_images % BATCH_SIZE))
        }
    }

    prediction := make_aligned([]u32, BATCH_SIZE, 32)
    actual := make_aligned([]u32, BATCH_SIZE, 32)

    nn.train(net, batches[:], uint(num_images), uint(BATCH_SIZE), EPOCHS, {
        eta = ETA,
        log_rate = LOG_EPOCH
    })

    pass, total := evaluate_model(net, test_images, test_labels, image_rows*image_cols)
    fmt.printfln("Evaluation: %i/%i (%f%%)", pass, total, f64(pass)/f64(total) * 100.0)

    digits_paths, err := os.read_all_directory_by_path("resources/handwritten/", context.allocator)
    defer delete(digits_paths)
    my_images := make([dynamic]u8, 0, len(digits_paths)*784*3)
    my_labels := make([dynamic]u8, 0, len(digits_paths))
    for path in digits_paths {
        digit := path.name[0] - '0'

        img, err := bmp.load(path.fullpath)
        defer image.destroy(img)
        assert(len(img.pixels.buf) == 784*3, "Should be a 28x28 image")

        append(&my_labels, digit)
        pixels := mem.slice_data_cast([]image.RGB_Pixel, img.pixels.buf[:])
        for pixel in pixels {
            append(&my_images, 255 - pixel.r)
        }
    }
    my_test_data := make_example(my_images[:], my_labels[:], 28*28, 0, u32(len(digits_paths)))
    // for i in 0..<uint(9) do print_image(my_test_data.input, i, 28, 28)

    nn.resize_matrices(net, 9)
    n, d := evaluate_model(net, my_images[:], my_labels[:], 28*28)
    fmt.printfln("Evaluation: %i/%i (%f%%)", n, d, f64(n)/f64(d) * 100.0)

}

evaluate_model :: proc(net: ^nn.Network, test_images: []u8, test_labels: []u8, image_size: u32) -> (pass: u32, total: u32) {
    num_labels := u32(len(test_labels))
    batch_size := u32(nn.current_batch_size(net))
    prediction := make_aligned([]u32, batch_size, 32)
    actual := make_aligned([]u32, batch_size, 32)

    i: u32 = 0
    is_rem_batch := false
    for ; i < num_labels; i += batch_size {
        is_rem_batch = i + batch_size - 1 < num_labels
        example_size := is_rem_batch ? batch_size : num_labels - i

        example := make_example(test_images, test_labels, image_size, i, example_size)

        if is_rem_batch do nn.resize_matrices(net, uint(example_size))
        output := nn.forward_prop(net, example.input)
        nn.argmax_batched(output, prediction[:example_size])
        nn.argmax_batched(example.output, actual[:example_size])
        
        total += example_size
        for i in 0..<example_size do if prediction[i] == actual[i] do pass += 1
    }
    if is_rem_batch do nn.resize_matrices(net, uint(batch_size))

    return
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

make_example :: proc(images: []u8, labels: []u8, image_size, start, length: u32) -> Example {
    input := mat.new_smat(uint(image_size), uint(length))
    output := mat.new_smat(10, uint(length))

    // Populate input matrix - each column is an image(28x28)
    // Populate output matrix - one-hot column for each label
    for i in start..<start+length {
        output.data[u32(labels[i]) * length + (i - start)] = 1.0
        for pixel, idx in images[i*image_size:][:image_size] {
            input.data[u32(idx) * length + (i - start)] = f32(pixel) / 255.0
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