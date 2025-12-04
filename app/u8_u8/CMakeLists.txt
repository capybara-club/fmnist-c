project(app_fmnist_c_u8_u8)

add_executable(
    ${PROJECT_NAME}
    main.c
)

target_link_libraries(
    ${PROJECT_NAME}
    fmnist_c
)

target_compile_definitions(${PROJECT_NAME} PRIVATE "CMAKE_SOURCE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

