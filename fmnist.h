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
fmnist_c(
    int64_t num_train,
    int64_t num_test,
    int64_t num_rows,
    int64_t num_cols,
    int64_t num_labels,
    float* x_train, int64_t x_train_ld,
    float* y_train, int64_t y_train_ld,
    float* x_test,  int64_t x_test_ld,
    float* y_test,  int64_t y_test_ld
);
