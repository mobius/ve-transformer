# NCC optimization of the tokenizer's regex/template source can take tens of minutes.
# Limit only this non-tensor source; keep model and quantized math optimized.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ve")
    function(ve_tokenizer_compile_override)
        if(TARGET llama)
            get_target_property(ve_sources llama SOURCES)
            list(REMOVE_ITEM ve_sources "llama-model.cpp")
            list(APPEND ve_sources "${CMAKE_BINARY_DIR}/qwen-ve-model.cpp")
            set_property(TARGET llama PROPERTY SOURCES "${ve_sources}")
            set_source_files_properties("${CMAKE_BINARY_DIR}/qwen-ve-model.cpp"
                TARGET_DIRECTORY llama PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON COMPILE_OPTIONS "-O1;-fno-inline")
            set_source_files_properties("${CMAKE_SOURCE_DIR}/src/unicode.cpp"
                "${CMAKE_SOURCE_DIR}/src/llama-model.cpp"
                TARGET_DIRECTORY llama PROPERTIES COMPILE_OPTIONS "-O1;-fno-inline")
        endif()
    endfunction()
    cmake_language(DEFER CALL ve_tokenizer_compile_override)
endif()
