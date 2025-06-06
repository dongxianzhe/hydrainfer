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

#define DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define DISPATCH_CASE_INTEGRAL_TYPES(...) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))