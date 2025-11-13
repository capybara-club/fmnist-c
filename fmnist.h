#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
#define FMNIST_C_PUBLIC_DEC extern "C"
#define FMNIST_C_PUBLIC_DEF extern "C"
#else
#define FMNIST_C_PUBLIC_DEC extern
#define FMNIST_C_PUBLIC_DEF
#endif

typedef enum {
    FMNIST_C_RESULT_SUCCESS,
    FMNIST_C_ERROR_CANT_OPEN_FILE,
    FMNIST_C_ERROR_CANT_FSTAT_FILE,
    FMNIST_C_ERROR_CANT_MMAP_FILE,
    FMNIST_C_ERROR_CANT_MUNMAP_FILE,
    FMNIST_C_ERROR_INSUFFICIENT_DEST,
    FMNIST_C_ERROR_DIM_MISMATCH,
    FMNIST_C_ERROR_INVALID_DATA,
} FmnistCResult;

FMNIST_C_PUBLIC_DEC
FmnistCResult
fmnist_c_dims(
    int64_t* num_train,
    int64_t* num_test,
    int64_t* num_rows,
    int64_t* num_cols,
    int64_t* num_labels
);

// Load FMNIST images as float [0.0, 1.0].
FMNIST_C_PUBLIC_DEC
FmnistCResult
fmnist_c_load_images_f32(
    bool is_train, 
    int64_t num_samples, 
    float* dest, 
    int64_t dest_stride_N,
    int64_t dest_stride_H, 
    int64_t dest_stride_W
);

// Load FMNIST images as uint8_t [0, 255].
FMNIST_C_PUBLIC_DEC
FmnistCResult
fmnist_c_load_images_u8(
    bool is_train, 
    int64_t num_samples, 
    uint8_t* dest,
    int64_t dest_stride_N, 
    int64_t dest_stride_H,
    int64_t dest_stride_W
);

// Load FMNIST labels as uint8_t (0-9).
FMNIST_C_PUBLIC_DEC
FmnistCResult
fmnist_c_load_labels_u8(
    bool is_train, 
    int64_t num_samples,
    uint8_t* dest, 
    int64_t dest_stride_N
);

// Load FMNIST labels as float one-hot [N, 10].
FMNIST_C_PUBLIC_DEC
FmnistCResult
fmnist_c_load_labels_onehot_f32(
    bool is_train, 
    int64_t num_samples, 
    float* dest,
    int64_t dest_stride_N, 
    int64_t dest_stride_C
);
