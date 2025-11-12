#include <stdio.h>
#include <fmnist.h>
#include <inttypes.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

static const char* label_strings[] = {
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
};

#define STRING(x) #x
#define XSTRING(x) STRING(x)

int
main() {
    int64_t num_train, num_test, num_rows, num_cols, num_labels;
    FmnistCResult result = 
        fmnist_c_dims(
            &num_train, &num_test, &num_rows, &num_cols, &num_labels
        );
    
    if (result != FMNIST_C_RESULT_SUCCESS) {
        printf("Failed to get fmnist dimensions\n");
    }

    printf(
        "N train:  %5" PRId64 "\n" 
        "N test:   %5" PRId64 "\n" 
        "N rows:   %5" PRId64 "\n" 
        "N cols:   %5" PRId64 "\n" 
        "N labels: %5" PRId64 "\n", 
        num_train, num_test, num_rows, num_cols, num_labels
    );

    uint8_t* x_train;
    uint8_t* y_train; 
    uint8_t* x_test;
    uint8_t* y_test;
    int64_t x_train_ld = num_train;
    int64_t y_train_ld = 1;
    int64_t x_test_ld = num_test;
    int64_t y_test_ld = 1;

    x_train = malloc(x_train_ld * num_rows * num_cols * sizeof(uint8_t));
    y_train = malloc(num_train * sizeof(uint8_t));
    x_test = malloc(x_test_ld * num_rows * num_cols * sizeof(uint8_t));
    y_test = malloc(num_test * sizeof(uint8_t));

    result = fmnist_c_load(
        FMNIST_C_PIXEL_FORMAT_UINT8,
        FMNIST_C_LABEL_FORMAT_UINT8,
        num_train, 
        num_test,
        num_rows, 
        num_cols, 
        num_labels,
        x_train, x_train_ld,
        y_train, y_train_ld,
        x_test, x_test_ld,
        y_test, y_test_ld
    );
    if (result != FMNIST_C_RESULT_SUCCESS) {
        printf("Failed to load fmnist data\n");
    }

    #define IMAGE_NUM 16
    int stb_result;

    uint8_t* single_image = (uint8_t*)malloc(num_rows * num_cols * sizeof(uint8_t));
    if (single_image == NULL) {
        printf("Failed to allocate memory for single image\n");
        // Cleanup and exit...
    }

    for (int64_t l = 0; l < num_rows * num_cols; l++) {
        single_image[l] = x_train[l * x_train_ld + IMAGE_NUM];  // For image n=0
    }

    // Now write the single image
    stb_result = stbi_write_png( XSTRING(CMAKE_SOURCE_PATH) "/output_uint8_x_train.png", num_cols, num_rows, 1, single_image, num_cols * 1);
    if (stb_result == 0) {
        printf("Failed to write PNG\n");
    }
    uint8_t train_label = y_train[IMAGE_NUM];
    printf("train label: %s\n", label_strings[train_label]);

    for (int64_t l = 0; l < num_rows * num_cols; l++) {
        single_image[l] = x_test[l * x_test_ld + IMAGE_NUM];  // For image n=0
    }

    // Now write the single image
    stb_result = stbi_write_png( XSTRING(CMAKE_SOURCE_PATH) "/output_uint8_x_test.png", num_cols, num_rows, 1, single_image, num_cols * 1);
    if (stb_result == 0) {
        printf("Failed to write PNG\n");
    }
    uint8_t test_label = y_test[IMAGE_NUM];
    printf("test label: %s\n", label_strings[test_label]);

    free(single_image);

    free(x_train);
    free(y_train);
    free(x_test);
    free(y_test);
}
