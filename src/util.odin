package main

import "core:os"
import "core:sys/windows"

enable_virtual_terminal :: proc() -> bool {
    when ODIN_OS == .Windows {
        HANDLE :: windows.HANDLE

        if os.stdout.impl == windows.INVALID_HANDLE_VALUE { return false }
        mode: windows.DWORD
        if !windows.GetConsoleMode(HANDLE(os.stdout.impl), &mode) { return false }

        mode |= windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if !windows.SetConsoleMode(HANDLE(os.stdout.impl), mode) { return false }

        return true
    }
}