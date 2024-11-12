#define DISPATCH_BOOL(expr, name, ...)                                    \
    [&] {                                                                 \
        if(expr){                                                         \
            constexpr bool name = true;                                   \
            __VA_ARGS__();                                                \
        }else{                                                            \
            constexpr bool name = false;                                  \
            __VA_ARGS__();                                                \
        }                                                                 \
    }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...)          \
    [&] {                                                                    \
        switch (pytorch_dtype) {                                             \
            case at::ScalarType::Float:{                                     \
                using c_type = float;                                        \
                __VA_ARGS__();                                               \
                break;                                                       \
            }                                                                \
            case at::ScalarType::Half: {                                     \
                using c_type = half;                                         \
                __VA_ARGS__();                                               \
                break;                                                       \
            }                                                                \
            default:                                                         \
                throw std::runtime_error(" failed to dispatch data type ");  \
        }                                                                    \
    }()
