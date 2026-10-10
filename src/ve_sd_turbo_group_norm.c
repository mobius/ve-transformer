/* Internal contiguous center/square tile; all three arrays must not overlap.
 * Variance accumulation deliberately remains in the strict caller.
 */
void sd_ve_group_norm_center_square_f32(int n, float *restrict dst,
        float *restrict squares, const float *restrict src, float mean) {
    for (int i = 0; i < n; ++i) {
        float value = src[i] - mean;
        dst[i] = value;
        squares[i] = value * value;
    }
}
