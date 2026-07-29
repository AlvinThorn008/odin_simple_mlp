MODE ?= release

ifeq ($(MODE), debug)
build: build/neural_net_dbg.exe
mode_suffix = _dbg
else
build: build/neural_net.exe
endif

./build/neural_net.exe: ./src/*.odin ./src/**/*.odin
	odin build ./src -microarch:native -o:speed -out:build/neural_net.exe -no-bounds-check -disable-assert

./build/neural_net_dbg.exe: ./src/*.odin ./src/**/*.odin
	odin build ./src -microarch:native -debug -out:build/neural_net_dbg.exe

run: build
	./build/neural_net$(mode_suffix).exe

test:
	odin test ./src/$(pkg) -define:ODIN_TEST_NAMES=$(tests) -define:ODIN_TEST_THREADS=1

testall:
	odin test ./src/$(pkg) -define:ODIN_TEST_THREADS=1