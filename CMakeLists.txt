cmake_minimum_required(VERSION 3.22)

project(fmnist_c)

add_compile_options(
	-march=native
	-Wall
    -Werror
)

set(INPUT_FILES
    "${CMAKE_SOURCE_DIR}/thirdparty/fashion-mnist/data/fashion/train-images-idx3-ubyte.gz"
    "${CMAKE_SOURCE_DIR}/thirdparty/fashion-mnist/data/fashion/train-labels-idx1-ubyte.gz"
    "${CMAKE_SOURCE_DIR}/thirdparty/fashion-mnist/data/fashion/t10k-images-idx3-ubyte.gz"
    "${CMAKE_SOURCE_DIR}/thirdparty/fashion-mnist/data/fashion/t10k-labels-idx1-ubyte.gz"
)

set(OUTPUT_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/fmnist/train-images-idx3-ubyte"
    "${CMAKE_CURRENT_BINARY_DIR}/fmnist/train-labels-idx1-ubyte"
    "${CMAKE_CURRENT_BINARY_DIR}/fmnist/t10k-images-idx3-ubyte"
    "${CMAKE_CURRENT_BINARY_DIR}/fmnist/t10k-labels-idx1-ubyte"
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/fmnist)

# Create a loop to generate custom commands for each file
# Get the length of the input files list
list(LENGTH INPUT_FILES input_files_length)

# Loop through input and output files dynamically
foreach(i RANGE 0 ${input_files_length})
    # Subtract 1 from the length because list indices start at 0
    math(EXPR i "${i} - 1")

    # Access the files from the lists using the index
    list(GET INPUT_FILES ${i} input_file)
    list(GET OUTPUT_FILES ${i} output_file)

    # Add a custom command to unzip each file
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND gzip -d -c ${input_file} > ${output_file}
        DEPENDS ${input_file}
        COMMENT "Unzipping ${input_file} to ${output_file}..."
    )
endforeach()

set(TRAIN_IMAGES_FILE "${CMAKE_CURRENT_BINARY_DIR}/fmnist/train-images-idx3-ubyte")
set(TRAIN_LABELS_FILE "${CMAKE_CURRENT_BINARY_DIR}/fmnist/train-labels-idx1-ubyte")
set(T10K_IMAGES_FILE "${CMAKE_CURRENT_BINARY_DIR}/fmnist/t10k-images-idx3-ubyte")
set(T10K_LABELS_FILE "${CMAKE_CURRENT_BINARY_DIR}/fmnist/t10k-labels-idx1-ubyte")

# Add a custom target that depends on all the output files
add_custom_target(unzip_fashion_mnist ALL DEPENDS ${OUTPUT_FILES} ${CMAKE_CURRENT_BINARY_DIR}/fmnist.h)

add_library(
    ${PROJECT_NAME}
    fmnist.c
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    .
)

target_compile_definitions(${PROJECT_NAME} PRIVATE "TRAIN_IMAGES_FILE=${TRAIN_IMAGES_FILE}")
target_compile_definitions(${PROJECT_NAME} PRIVATE "TRAIN_LABELS_FILE=${TRAIN_LABELS_FILE}")
target_compile_definitions(${PROJECT_NAME} PRIVATE "T10K_IMAGES_FILE=${T10K_IMAGES_FILE}")
target_compile_definitions(${PROJECT_NAME} PRIVATE "T10K_LABELS_FILE=${T10K_LABELS_FILE}")

add_subdirectory(app)
