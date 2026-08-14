package bench

import "core:time"
import "core:fmt"
import "core:terminal/ansi"

BenchProc :: #type proc(data: rawptr)

bench_it :: proc(runs: uint, arg: rawptr, func: BenchProc) -> f64 {
    elapsed := f64(0.0)

    for i in 0..<runs {
        start := time.tick_now()
        func(arg)
        elapsed += time.duration_microseconds(time.tick_since(start))
    }

    return elapsed
}

time_it :: proc(name: string, runs: uint, arg: rawptr, func: BenchProc) {
    elapsed := bench_it(runs, arg, func)

    START :: ansi.CSI + ansi.FG_BRIGHT_YELLOW + ansi.SGR
    END   :: ansi.CSI + ansi.RESET + ansi.SGR

    fmt.printfln("bench %-20s:  %s% 10.2f%s us/runs", name, START, elapsed/f64(runs), END)
}