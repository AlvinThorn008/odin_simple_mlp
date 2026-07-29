#+feature using-stmt

package main

import "core:os"
import "core:fmt"
import "core:math/rand"
import "mat"

NUM_MAT :: 3

MAT_SIZE :: 1911

main :: proc() {
	// enable_virtual_terminal()
	using mat
	
	ref_output: SMat
	defer delete_smat(ref_output)
	{
		rand.reset(45)
		a := rand_mat(MAT_SIZE, MAT_SIZE)
		b := rand_mat(MAT_SIZE, MAT_SIZE)
		c := new_smat(MAT_SIZE, MAT_SIZE)
		defer { delete_smat(a); delete_smat(b); delete_smat(c) }

		// Naive benchmark takes too  long but only the result is required
		// for testing the other impls. The following block of code loads in
		// the last result given that the old and new dimensions are equivalent, otherwise the bench is redone. 

		rows, cols, file, err := read_smat_header("./outputs/naive.bin")
		fmt.printfln("OLD Header: (%d, %d) | MAT_SIZE: %d | Err: %v", rows, cols, MAT_SIZE, err)
		if err == os.ERROR_NONE && rows == MAT_SIZE && cols == MAT_SIZE {
			delete_smat(c)
			c = import_smat_with_meta(rows, cols, file)
		} else {
			do_bench(smat_matmul_naive, 1, "Naive", a, b, c, nil)
			export_smat(a, "./outputs/a.bin")
			export_smat(b, "./outputs/b.bin")
			export_smat(c, "./outputs/naive.bin")
		}

		ref_output = copy_smat(c)
	}

	{
		rand.reset(45)
		a := rand_mat(MAT_SIZE, MAT_SIZE)
		b := rand_mat(MAT_SIZE, MAT_SIZE)
		c := new_smat(MAT_SIZE, MAT_SIZE)
		defer { delete_smat(a); delete_smat(b); delete_smat(c) }

		do_bench(smat_matmul_loop_order, 1, "Loop reordered", a, b, c, &ref_output)
		export_smat(c, "./outputs/loop.bin")
	}
	
	{
		rand.reset(45)
		a := rand_mat(MAT_SIZE, MAT_SIZE)
		b := rand_mat_cm(MAT_SIZE, MAT_SIZE)
		c := new_smat(MAT_SIZE, MAT_SIZE)
		defer { delete_smat(a); delete_smat(b); delete_smat(c) }

		do_bench(smat_matmul_rc_simd, 10, "Right CM Vectorized", a, b, c, &ref_output)
		export_smat(c, "./outputs/rcm_vec.bin")
	}

	{
		rand.reset(45)
		a := rand_mat(MAT_SIZE, MAT_SIZE)
		b := rand_mat(MAT_SIZE, MAT_SIZE)
		c := new_smat(MAT_SIZE, MAT_SIZE)
		defer { delete_smat(a); delete_smat(b); delete_smat(c) }
		
		do_bench(smat_matmul_with_kernel, 10, "Kernels", a, b, c, &ref_output)
		export_smat(c, "./outputs/kernel.bin")
	}

	{
		rand.reset(45)
		a := rand_mat(MAT_SIZE, MAT_SIZE)
		b := rand_mat(MAT_SIZE, MAT_SIZE)
		c := new_smat(MAT_SIZE, MAT_SIZE)
		defer { delete_smat(a); delete_smat(b); delete_smat(c) }
		
		do_bench(smat_matmul_blocking, 10, "Blocking", a, b, c, &ref_output)
		export_smat(c, "./outputs/blocking.bin")
	}
}