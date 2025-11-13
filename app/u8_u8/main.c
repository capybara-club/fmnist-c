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

    x_train = malloc(num_train * num_rows * num_cols * sizeof(uint8_t));
    y_train = malloc(num_train * sizeof(uint8_t));
    x_test = malloc(num_test * num_rows * num_cols * sizeof(uint8_t));
    y_test = malloc(num_test * sizeof(uint8_t));

    result = fmnist_c_load_images_u8(
        true,
        num_train, 
        x_train,
        num_rows * num_cols,
        num_cols,
        1
    );
    if (result != FMNIST_C_RESULT_SUCCESS) {
        printf("Failed to load fmnist data\n");
    }

    result = fmnist_c_load_labels_u8(
        true,
        num_train, 
        y_train,
        1
    );
    if (result != FMNIST_C_RESULT_SUCCESS) {
        printf("Failed to load fmnist data\n");
    }

    #define IMAGE_NUM 25
    int stb_result;

    uint8_t* image = x_train + IMAGE_NUM * num_cols * num_rows;
    stb_result = stbi_write_png( XSTRING(CMAKE_SOURCE_PATH) "/output_uint8_x_train.png", num_cols, num_rows, 1, image, num_cols * 1);
    if (stb_result == 0) {
        printf("Failed to write PNG\n");
    }
    uint8_t train_label = y_train[IMAGE_NUM];
    printf("train label: %s\n", label_strings[train_label]);

    free(x_train);
    free(y_train);
    free(x_test);
    free(y_test);
}
