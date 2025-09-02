#include <stdio.h>
#include <fmnist.h>
#include <inttypes.h>
#include <stdlib.h>


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

    float* x_train;
    float* y_train; 
    float* x_test;
    float* y_test;
    int64_t x_train_ld = num_train;
    int64_t y_train_ld = num_train;
    int64_t x_test_ld = num_test;
    int64_t y_test_ld = num_test;

    x_train = malloc(x_train_ld * num_rows * num_cols * sizeof(float));
    y_train = malloc(y_train_ld * num_labels * sizeof(float));
    x_test = malloc(x_test_ld * num_rows * num_cols * sizeof(float));
    y_test = malloc(y_test_ld * num_labels * sizeof(float));

    result = fmnist_c(
        num_train, num_test,
        num_rows, num_cols, num_labels,
        x_train, x_train_ld,
        y_train, y_train_ld,
        x_test, x_test_ld,
        y_test, y_test_ld
    );
    if (result != FMNIST_C_RESULT_SUCCESS) {
        printf("Failed to load fmnist data\n");
    }
#if 0
    for (int64_t n = 0; n < num_train; n++) {
        for (int64_t l = 0; l < num_rows * num_cols; l++) {
            printf("%g,", x_train[l * x_train_ld + n]);
        }
        printf("\n");
    }

    for (int64_t n = 0; n < num_train; n++) {
        for (int64_t l = 0; l < num_labels; l++) {
            printf("%1.0f", y_train[l * y_train_ld + n]);
        }
        printf("\n");
    }

    for (int64_t n = 0; n < num_test; n++) {
        for (int64_t l = 0; l < num_rows * num_cols; l++) {
            printf("%g,", x_test[l * x_test_ld + n]);
        }
        printf("\n");
    }

    for (int64_t n = 0; n < num_test; n++) {
        for (int64_t l = 0; l < num_labels; l++) {
            printf("%1.0f", y_test[l * y_test_ld + n]);
        }
        printf("\n");
    }
#endif

    free(x_train);
    free(y_train);
    free(x_test);
    free(y_test);
}
