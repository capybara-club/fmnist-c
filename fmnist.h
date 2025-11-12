#pragma once

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
    FMNIST_C_ERROR_INVALID_DATA,
} FmnistCResult;

typedef enum {
    FMNIST_C_PIXEL_FORMAT_FLOAT,   /* 32-bit float, range [0,1] */
    FMNIST_C_PIXEL_FORMAT_UINT8,   /* 8-bit unsigned, range [0,255] */
} FmnistCPixelFormat;

typedef enum {
    FMNIST_C_LABEL_FORMAT_ONEHOT_FLOAT,  /* float one-hot vector (num_labels) */
    FMNIST_C_LABEL_FORMAT_UINT8,         /* uint8_t class index (0 … num_labels-1) */
} FmnistCLabelFormat;

FMNIST_C_PUBLIC_DEC
FmnistCResult
fmnist_c_dims(
    int64_t* num_train,
    int64_t* num_test,
    int64_t* num_rows,
    int64_t* num_cols,
    int64_t* num_labels
);

FMNIST_C_PUBLIC_DEC
FmnistCResult
fmnist_c_load(
    FmnistCPixelFormat pixel_format,
    FmnistCLabelFormat label_format,
    int64_t num_train,
    int64_t num_test,
    int64_t num_rows,
    int64_t num_cols,
    int64_t num_labels,
    void* x_train, int64_t x_train_ld,
    void* y_train, int64_t y_train_ld,
    void* x_test,  int64_t x_test_ld,
    void* y_test,  int64_t y_test_ld
);
