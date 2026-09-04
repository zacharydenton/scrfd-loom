// C ABI for the resident det_10g (SCRFD-10GF) inference session.
//
// The same shape as dinov3-loom's host/dinov3.h so both models share one ctypes
// loader: a session created once, runs serialized on it, element counts in the
// ABI so an undersized caller allocation is rejected before the GPU sees it.
#ifndef SCRFD_LOOM_H
#define SCRFD_LOOM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Increment when the ABI changes incompatibly. Callers compare before creating.
#define SCRFD_ABI_VERSION 2u

enum {
    SCRFD_OK = 0,
    SCRFD_ERROR = 1,
    SCRFD_INVALID_ARGUMENT = 64,
};

// One candidate detection: x1, y1, x2, y2, score, then five (x, y) keypoints.
// Coordinates are pixels of the 640x640 letterboxed input.
#define SCRFD_CANDIDATE_FLOATS 15

typedef struct scrfd_session scrfd_session;

uint32_t scrfd_abi_version(void);
int scrfd_input_size(void);           // 640

// Creates an independent resident session with buffers sized for up to
// max_batch images. On failure returns non-zero, leaves *out_session null and
// writes a NUL-terminated message to error (when error_capacity is non-zero).
int scrfd_create(const char *weights_dir, const char *kernels_dir, int max_batch,
                 scrfd_session **out_session, char *error, size_t error_capacity);

// Runs one batch and decodes it. `input` is batch letterboxed BGR uint8 images,
// each 640 x 640 x 3, exactly what insightface's detect() builds before
// blobFromImage; the normalisation happens on the GPU. `input_bytes` must be
// batch * 640 * 640 * 3.
//
// Every anchor whose sigmoid score is >= det_thresh (0 < det_thresh < 1) becomes
// one row of SCRFD_CANDIDATE_FLOATS floats, insightface's distance2bbox /
// distance2kps already applied, in letterboxed-input pixels and not yet sorted
// or suppressed. Image b's rows start at candidates + b * max_candidates *
// SCRFD_CANDIDATE_FLOATS and counts[b] says how many were written, where
// max_candidates = candidates_elements / (batch * SCRFD_CANDIDATE_FLOATS)
// (candidates_elements must be a positive multiple of that divisor, and
// counts_elements must equal batch). Rows come level-major (stride 8, 16, 32),
// pixel-major, anchor-minor, the order insightface stacks them in. An image
// with more candidates than max_candidates fails the whole call with
// SCRFD_ERROR and a message naming the count; nothing is truncated silently.
//
// Calls on one session are serialized internally. Destruction must not race a run.
int scrfd_run(scrfd_session *session, const uint8_t *input, size_t input_bytes, int batch,
              float det_thresh, float *candidates, size_t candidates_elements,
              int32_t *counts, size_t counts_elements, char *error, size_t error_capacity);

int scrfd_max_batch(const scrfd_session *session);
void scrfd_destroy(scrfd_session *session);

#ifdef __cplusplus
}
#endif

#endif
