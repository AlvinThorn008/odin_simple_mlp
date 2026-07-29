package main

import "core:mem"
import "core:time"
import "core:fmt"
import "mat"
import "core:terminal/ansi"

SMat :: mat.SMat

MatMulProc :: proc(a, b, c: SMat)

// Perform a matrix benchmark of `bench_func`, running `runs` times and printing metrics(total time, average time)
do_bench :: proc(bench_func: MatMulProc, runs: uint, name: string, a, b, c: SMat, check_mat: Maybe(^SMat)) {
	fmt.printfln("\n\n[%v]", name)

	elapsed: f64 = 0.0
	for i in 0..<runs {
		mem.zero_slice(c.data)
		start := time.tick_now()
		bench_func(a, b, c)
		elapsed += time.duration_seconds(time.tick_since(start))
	}

	fmt.printfln("%d runs completed", runs)
	fmt.printfln("Total time:       " + ansi.CSI + ansi.FG_BRIGHT_CYAN + ansi.SGR + "%fs" + ansi.CSI + ansi.RESET + ansi.SGR, elapsed)
	fmt.printfln("Average run time: " + ansi.CSI + ansi.FG_BRIGHT_YELLOW + ansi.SGR + "%fs" + ansi.CSI + ansi.RESET + ansi.SGR, elapsed / f64(runs))

	if c_mat, ok := check_mat.?; ok {
		matches := mat.smat_rel_compare(c, c_mat^)
		fmt.printfln("%d/%d matched", matches, len(c.data))
	}
}



