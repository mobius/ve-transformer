# Generated sources apply to both CPU and VE so comparisons use the same graph.
function(sd_project_overrides)
    set(sd_root "${CMAKE_SOURCE_DIR}")
    get_filename_component(sd_project_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    if(TARGET ggml-base)
        get_target_property(sd_base_sources ggml-base SOURCES)
        list(REMOVE_ITEM sd_base_sources "ggml.c" "${sd_root}/ggml/src/ggml.c" "gguf.cpp" "${sd_root}/ggml/src/gguf.cpp")
        list(APPEND sd_base_sources "${sd_project_root}/build/sd-overlay/sd-ggml.c" "${sd_project_root}/build/sd-overlay/sd-gguf.cpp")
        set_property(TARGET ggml-base PROPERTY SOURCES "${sd_base_sources}")
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ve")
            set_source_files_properties("${sd_project_root}/build/sd-overlay/sd-gguf.cpp"
                TARGET_DIRECTORY ggml-base PROPERTIES COMPILE_OPTIONS "-O0;-fno-inline")
            set_source_files_properties("${sd_root}/ggml/src/ggml-backend-meta.cpp"
                TARGET_DIRECTORY ggml-base PROPERTIES COMPILE_OPTIONS "-O0;-fno-inline")
        endif()
    endif()
    if(TARGET stable-diffusion)
        get_target_property(sd_sources stable-diffusion SOURCES)
        list(REMOVE_ITEM sd_sources "${sd_root}/src/pipeline/image.cpp" "${sd_root}/src/pipeline/diffusion_engine.cpp" "${sd_root}/src/core/ggml_extend.cpp")
        list(APPEND sd_sources "${sd_project_root}/build/sd-overlay/sd-image.cpp" "${sd_project_root}/build/sd-overlay/sd-diffusion-engine.cpp" "${sd_project_root}/build/sd-overlay/sd-ggml-extend.cpp")
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ve")
            # The inference build reads official FP32 components directly.
            list(REMOVE_ITEM sd_sources "${sd_project_root}/build/sd-overlay/sd-image.cpp")
            list(APPEND sd_sources "${sd_project_root}/src/ve_sd_image.cpp")
            # Explicitly reject the optional conversion API instead of compiling it.
            list(REMOVE_ITEM sd_sources "${sd_root}/src/convert.cpp")
            list(APPEND sd_sources "${sd_project_root}/build/sd-overlay/sd-convert-disabled.cpp")
            list(REMOVE_ITEM sd_sources "${sd_root}/src/detailer.cpp")
            list(APPEND sd_sources "${sd_project_root}/build/sd-overlay/sd-detailer-disabled.cpp")
            list(REMOVE_ITEM sd_sources "${sd_root}/src/pipeline/model_builders.cpp")
            list(APPEND sd_sources "${sd_project_root}/build/sd-overlay/sd-model-builders.cpp")
            list(REMOVE_ITEM sd_sources "${sd_root}/src/pipeline/video.cpp")
            list(APPEND sd_sources "${sd_project_root}/build/sd-overlay/sd-video-disabled.cpp")
            # Keep numerical kernels at O1; use conservative compilation for
            # template-heavy orchestration while validating the initial port.
            set_source_files_properties(
                "${sd_project_root}/src/ve_sd_image.cpp"
                "${sd_project_root}/build/sd-overlay/sd-diffusion-engine.cpp"
                "${sd_project_root}/build/sd-overlay/sd-model-builders.cpp"
                TARGET_DIRECTORY stable-diffusion PROPERTIES COMPILE_OPTIONS "-O0;-fno-inline")
        endif()
        set_property(TARGET stable-diffusion PROPERTY SOURCES "${sd_sources}")
        target_include_directories(stable-diffusion PRIVATE "${sd_root}/src/pipeline")
        target_include_directories(stable-diffusion BEFORE PRIVATE "${sd_project_root}/build/sd-overlay/include")
    endif()
endfunction()
cmake_language(DEFER CALL sd_project_overrides)
