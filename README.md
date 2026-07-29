# Simple Neural Network in Odin

This is a rewrite of a [neural network I made in C++](https://github.com/AlvinThorn008/simple_mlp) for a university assignment.

## Getting Started

### Dependencies

* Odin compiler version dev-2026-07-nightly:819fdc7 or greater*
* Visual Studio Build tools
* make*
* CPU with AVX2 support at least*


<sup>*</sup>Newer odin releases might break stuff so be warned.

<sup>*</sup>Not required as the build is simple, see .

<sup>*</sup>Not required but the program is optimized with AVX2 in mind.

### Building code

```sh
git clone https://github.com/AlvinThorn008/odin_simple_mlp
cd odin_simple_mlp
make build # OR odin build ./src -microarch:native -o:speed -out:build/neural_net.exe -no-bounds-check -disable-assert
```

### Executing program
```sh
make run
```
You could also run the binary built earlier: `./build/neural_net.exe`

Odin supposedly supports MacOS, linux too so you could build on those platforms as well.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](/LICENSE.txt) file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [Algorithmica](https://en.algorithmica.org/hpc/algorithms/matmul/)
* [Godbolt](https://godbolt.org/)
* [Cache blocking Visualization](https://jukkasuomela.fi/cache-blocking-demo/)
* [Anatomy of High-Performance Matrix
Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)