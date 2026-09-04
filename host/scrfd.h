// C ABI for the resident det_10g (SCRFD-10GF) inference session.
//
// Mirrors dinov3-loom's host/dinov3.h so both models share one ctypes loader.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Increment when the ABI changes incompatibly. Callers compare before creating.
#define SCRFD_ABI_VERSION 1u

typedef struct scrfd_session scrfd_session;

uint32_t scrfd_abi_version(void);

// Creates a resident session with buffers sized for up to max_batch images.
// On failure returns non-zero, leaves *out_session null, and writes a
// NUL-terminated message to `error` (when error_capacity is non-zero).
int scrfd_create(const char *weights_dir, const char *kernels_dir, int max_batch,
                 scrfd_session **out_session, char *error, size_t error_capacity);

// Runs one batch. `input` is the NCHW f32 blob insightface's blobFromImage
// produces, batch x 3 x 640 x 640, already normalised. The three outputs are
// the fused head tensors, NHWC f16 as uint16: [batch*H*W][64] with H = W =
// 640/8, 640/16, 640/32, channels [score(2) | box(8) | kps(20) | zero pad].
// Scores are pre-sigmoid. Element counts are part of the ABI so an undersized
// caller allocation is rejected before the GPU sees it.
int scrfd_run(scrfd_session *session, const float *input, size_t input_elements, int batch,
              uint16_t *head8, size_t head8_elements,
              uint16_t *head16, size_t head16_elements,
              uint16_t *head32, size_t head32_elements,
              char *error, size_t error_capacity);

int scrfd_max_batch(const scrfd_session *session);
int scrfd_input_size(void);           // 640
void scrfd_destroy(scrfd_session *session);

#ifdef __cplusplus
}
#endif
