#include <fmnist.h>

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define _FMNIST_C_STRING(x) #x
#define _FMNIST_C_XSTRING(x) _FMNIST_C_STRING(x)

#define _FMNIST_C_MAX(a, b) ((a) > (b) ? (a) : (b))

static const char* _fmnist_x_train_data_path =  _FMNIST_C_XSTRING(TRAIN_IMAGES_FILE);
static const char* _fmnist_y_train_data_path =  _FMNIST_C_XSTRING(TRAIN_LABELS_FILE);
static const char* _fmnist_x_test_data_path =   _FMNIST_C_XSTRING(T10K_IMAGES_FILE);
static const char* _fmnist_y_test_data_path =   _FMNIST_C_XSTRING(T10K_LABELS_FILE);

typedef struct {
	uint8_t* mapped_data;
	size_t mapped_size;
} _FmnistCMRead;

static 
inline 
FmnistCResult
_fmnist_c_mmap_file_open_ro(
	const char* file_name,
    _FmnistCMRead* m_read_out
) {
	uint8_t *mapped_data;
	size_t mapped_size;

	int fd = open(file_name, O_RDONLY);
	if (fd < 0) {
        return FMNIST_C_ERROR_CANT_OPEN_FILE;
	}

	struct stat statbuf;
	int err = fstat(fd, &statbuf);
	if (err < 0) {
        close( fd );
		return FMNIST_C_ERROR_CANT_FSTAT_FILE;
	}

	mapped_size = statbuf.st_size;
	mapped_data = (uint8_t*)mmap( NULL, statbuf.st_size, PROT_READ, MAP_SHARED, fd, 0 );
	if (mapped_data == MAP_FAILED) {
		return FMNIST_C_ERROR_CANT_MMAP_FILE;
	}
	close( fd );

    *m_read_out = (_FmnistCMRead) {
        .mapped_data = mapped_data,
		.mapped_size = mapped_size
    };

	return FMNIST_C_RESULT_SUCCESS;
}

static 
inline 
FmnistCResult
_fmnist_c_mmap_file_close(
	_FmnistCMRead m_read
) {
	int err = munmap(m_read.mapped_data, m_read.mapped_size);
	if (err != 0) {
        return FMNIST_C_ERROR_CANT_MUNMAP_FILE;
	}
    return FMNIST_C_RESULT_SUCCESS;
}

static
inline
unsigned int 
_fmnist_c_swap_endian(
	unsigned int num
) {
	return 
		((num >> 24) & 0x000000FF) | // Move byte 3 to byte 0
		((num >>  8) & 0x0000FF00) | // Move byte 2 to byte 1
		((num <<  8) & 0x00FF0000) | // Move byte 1 to byte 2
		((num << 24) & 0xFF000000);  // Move byte 0 to byte 3
}

static
inline
FmnistCResult
__fmnist_load_data(
    _FmnistCMRead mr_data,
    int64_t* num_samples_ref,
    int64_t* num_rows_ref,
    int64_t* num_cols_ref,
    float* data,
    int64_t data_ld
) {
    uint8_t *mapped_data = mr_data.mapped_data;
    uint32_t *data_u32 = (uint32_t*)mapped_data;

    if (_fmnist_c_swap_endian(*data_u32++) != 2051) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }

    int64_t num_samples = _fmnist_c_swap_endian(*data_u32++);
    int64_t num_rows = _fmnist_c_swap_endian(*data_u32++);
    int64_t num_cols = _fmnist_c_swap_endian(*data_u32++);

    if (data == NULL) {
        *num_samples_ref = num_samples;
        *num_rows_ref = num_rows;
        *num_cols_ref = num_cols;
        return FMNIST_C_RESULT_SUCCESS;
    }

    if (*num_samples_ref > num_samples) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    if (*num_rows_ref != num_rows) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    if (*num_cols_ref != num_cols) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }

    uint8_t* data_ptr = (uint8_t*)data_u32;
    int64_t D = num_cols * num_rows;
    for (int64_t n = 0; n < *num_samples_ref; n++) {
        for (int64_t d = 0; d < D; d++) {
            uint8_t pixel = data_ptr[n * D + d];
            data[d * data_ld + n] = (float)pixel / 255.0f;
        }
    }

    return FMNIST_C_RESULT_SUCCESS;
}

static
inline
FmnistCResult
_fmnist_load_data(
    const char* data_path,
    int64_t* num_samples,
    int64_t* num_rows,
    int64_t* num_cols,
    float* data,
    int64_t data_ld
) {
    FmnistCResult result;
    _FmnistCMRead mr_data;

    result = _fmnist_c_mmap_file_open_ro(data_path, &mr_data);
    if (result != FMNIST_C_RESULT_SUCCESS) {
        return result;
    }

    FmnistCResult main_result = 
        __fmnist_load_data(
            mr_data, num_samples, num_rows, num_cols,
            data, data_ld
        );

    result = _fmnist_c_mmap_file_close(mr_data);
    if (main_result != FMNIST_C_RESULT_SUCCESS) {
        return main_result;
    }
    if (result != FMNIST_C_RESULT_SUCCESS) {
        return result;
    }
    return main_result;
}

static
inline
FmnistCResult
__fmnist_load_labels(
    _FmnistCMRead mr_labels,
    int64_t* num_samples_ref,
    int64_t* num_labels_ref,
    float* label_data,
    int64_t label_data_ld
) {
    uint8_t *mapped_data = mr_labels.mapped_data;
    uint32_t *labels_u32 = (uint32_t*)mapped_data;
    
    if (_fmnist_c_swap_endian(*labels_u32++) != 2049) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    
    int64_t num_samples = _fmnist_c_swap_endian(*labels_u32++);
    uint8_t* labels = (uint8_t*)labels_u32;
    int64_t num_labels = 0;
	for (size_t i = 0; i < num_samples; i++) {
		num_labels = _FMNIST_C_MAX(num_labels, labels[i]);
	}
	(num_labels)++;

    if (label_data == NULL) {
        *num_samples_ref = num_samples;
        *num_labels_ref = num_labels;
        return FMNIST_C_RESULT_SUCCESS;
    }

    if (*num_samples_ref > num_samples) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    if (*num_labels_ref != num_labels) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }

    for (int64_t l = 0; l < *num_labels_ref; l++) {
        for (int64_t n = 0; n < *num_samples_ref; n++) {
            int64_t label_v = labels[n];
            float v = l == label_v ? 1.0f : 0.0f;
            label_data[l * label_data_ld + n] = v;
        }
    }

    return FMNIST_C_RESULT_SUCCESS;
}

static
inline
FmnistCResult
_fmnist_load_labels(
    const char* data_path,
    int64_t* num_samples,
    int64_t* num_labels,
    float* label_data, 
    int64_t label_data_ld
) {
    FmnistCResult result;
    _FmnistCMRead mr_labels;

    result = _fmnist_c_mmap_file_open_ro(data_path, &mr_labels);
    if (result != FMNIST_C_RESULT_SUCCESS) {
        return result;
    }

    FmnistCResult main_result = 
        __fmnist_load_labels(
            mr_labels, num_samples, num_labels,
            label_data, label_data_ld
        );

    result = _fmnist_c_mmap_file_close(mr_labels);
    if (main_result != FMNIST_C_RESULT_SUCCESS) {
        return main_result;
    }
    if (result != FMNIST_C_RESULT_SUCCESS) {
        return result;
    }
    return main_result;
}

FmnistCResult
fmnist_c_dims(
    int64_t* num_train,
    int64_t* num_test,
    int64_t* num_rows,
    int64_t* num_cols,
    int64_t* num_labels
) {
    FmnistCResult result;
    int64_t x_train_num_samples, x_train_num_rows, x_train_num_cols; 
    result = 
        _fmnist_load_data(
            _FMNIST_C_XSTRING(TRAIN_IMAGES_FILE),
            &x_train_num_samples, 
            &x_train_num_rows,
            &x_train_num_cols,
            NULL, 0
        );
    if (result != FMNIST_C_RESULT_SUCCESS) {
        return result;
    }

    int64_t y_train_num_samples, y_train_num_labels; 
    result = 
        _fmnist_load_labels(
            _FMNIST_C_XSTRING(TRAIN_LABELS_FILE),
            &y_train_num_samples,
            &y_train_num_labels,
            NULL, 0
        );
    if (result != FMNIST_C_RESULT_SUCCESS) {
        return result;
    }

    int64_t x_test_num_samples, x_test_num_rows, x_test_num_cols; 
    result = 
        _fmnist_load_data(
            _FMNIST_C_XSTRING(T10K_IMAGES_FILE),
            &x_test_num_samples, 
            &x_test_num_rows, 
            &x_test_num_cols,
            NULL, 0
        );
    if (result != FMNIST_C_RESULT_SUCCESS) {
        return result;
    }

    int64_t y_test_num_samples, y_test_num_labels; 
    result = 
        _fmnist_load_labels(
            _FMNIST_C_XSTRING(T10K_LABELS_FILE),
            &y_test_num_samples, 
            &y_test_num_labels,
            NULL, 0
        );
    if (result != FMNIST_C_RESULT_SUCCESS) {
        return result;
    }

    if (x_train_num_samples != y_train_num_samples) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    if (x_test_num_samples != y_test_num_samples) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    if (x_train_num_rows != x_test_num_rows) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    if (x_train_num_cols != x_test_num_cols) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    if (y_train_num_labels != y_test_num_labels) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    *num_train = x_train_num_samples;
    *num_test = x_test_num_samples;
    *num_rows = x_train_num_rows;
    *num_cols = x_train_num_cols;
    *num_labels = y_train_num_labels;

    return result;
}

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
) {
    FmnistCResult result;

    if (x_train_ld < num_train) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    if (y_train_ld < num_train) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    if (x_test_ld < num_test) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }
    if (y_test_ld < num_test) {
        return FMNIST_C_ERROR_INVALID_DATA;
    }

    if (x_train != NULL) {
        result = 
            _fmnist_load_data(
                _fmnist_x_train_data_path,
                &num_train, 
                &num_rows,
                &num_cols,
                x_train, x_train_ld
            );
        if (result != FMNIST_C_RESULT_SUCCESS) {
            return result;
        }
    }

    if (y_train != NULL) {
        result = 
            _fmnist_load_labels(
                _fmnist_y_train_data_path,
                &num_train,
                &num_labels,
                y_train, y_train_ld
            );
        if (result != FMNIST_C_RESULT_SUCCESS) {
            return result;
        }
    }

    if (x_test != NULL) {
        result = 
            _fmnist_load_data(
                _fmnist_x_test_data_path,
                &num_test, 
                &num_rows,
                &num_cols,
                x_test, x_test_ld
            );
        if (result != FMNIST_C_RESULT_SUCCESS) {
            return result;
        }
    }

    if (y_test != NULL) {
        result = 
            _fmnist_load_labels(
                _fmnist_y_test_data_path,
                &num_test,
                &num_labels,
                y_test, y_test_ld
            );
        if (result != FMNIST_C_RESULT_SUCCESS) {
            return result;
        }
    }

    return FMNIST_C_RESULT_SUCCESS;
}