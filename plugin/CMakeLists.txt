ida_add_plugin(tensorlens
  SOURCES
    tensorlens.cpp
    idb_models.cpp
    runtime.cpp
    typelib.cpp
    views.cpp
  LIBRARIES
    tensorlens_core
  DEFINES
    USE_STANDARD_FILE_FUNCTIONS
    USE_DANGEROUS_FUNCTIONS
)
if(MSVC)
  target_compile_options(tensorlens PRIVATE /EHsc)
endif()
