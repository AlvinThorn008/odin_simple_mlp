// assume x is 256−bit aligned
// assume n is a multiple of 16
float shift_reduce_avx(float* x, size_t n) {
    __m256 stack[60];
    size_t p = 0;

    for (size_t i = 0; i + 16 <= n; i += 16) {
        //unrolled shift+reduce
        __m256 v = _mm256_add_ps(_mm256_load_ps(x + i), _mm256_load_ps(x + i + 4))
        __m256 w = _mm256_add_ps(_mm256_load_ps(x + i + 8), _mm256_load_ps(x + i + 12))

        v = _mm256_add_ps(v, w);

        for (size_t bitmask=16; i & bitmask; bitmask <<= 1, --p) {
            v = _mm256_add_ps(v, stack[p-1]);
            stack[p++] = v;
        }
    }

    __m256 vsum = __mm256_setzero_ps();

    for (size_t i = p; i > 0; --i) {
        vsum = _mm256_add_ps(vsum, stack[i - 1]);
    }

    vsum = _mm256_hadd_ps(vsum, vsum);
    vsum = _mm256_hadd_ps(vsum, vsum);

    return _mm_cvtss_f32(
        _mm_add_ss(
            _mm256_castps256_ps128(vsum),
            _mm256_extractf128_ps(s, 1)
        )
    );
}