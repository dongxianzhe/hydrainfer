import os
import torch

def get_device_capability():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f'_sm_{major}{minor}'
    else:
        return '_null_'
    
def generate_header(kernel_name, generated_code):
    device_capability = get_device_capability()
    ptx_filename = kernel_name + device_capability + ".ptx"
    ptx_header_filename = ptx_filename + ".h"
    ptx_header_filename = os.path.join("../ptx_kernels", ptx_header_filename)
    with open(ptx_header_filename, 'a') as f:
        f.write(generated_code)


from jinja2 import Template
from itertools import product
def template_render(kernel_name,
                    signature_suffix,
                    all_shape_dict, 
                    block_m_array, 
                    block_n_array, 
                    block_k_array, 
                    block_size_array,
                    shared_mem_bytes,
                    ptx_array):
    kernel_signature = kernel_name + get_device_capability() + "_" + signature_suffix

    input_dims = []
    for combo in product(*all_shape_dict.values()):
        M, N, K = combo
        if N == 7168 and K == 7168:
            continue
        input_dims.append(combo)

    cpp_template = Template("""
#pragma once
#include <array>
constexpr const char* {{ kernel_signature }} = R"({{ kernel_name }})";
constexpr std::array<std::array<int, {{ input_dims[0]|length }}>, {{ input_dims|length }}> {{ kernel_signature }}_INPUT_DIM_ARRAY = {{ '{{' }} 
    {% for dim in input_dims -%}
    { {{ dim | map('string') | join(', ') }} }{% if not loop.last %},{% else %} {% endif %}  // Index {{ loop.index0 }}
    {% endfor -%}
{{ '}}' }};
{% if block_m_array is defined and block_m_array -%}
constexpr std::array<int, {{ block_m_array|length }}> {{ kernel_signature }}_BLOCK_M_ARRAY{ {{ block_m_array | join(', ') }} };
{% endif -%}
{% if block_n_array is defined and block_n_array -%}
constexpr std::array<int, {{ block_n_array|length }}> {{ kernel_signature }}_BLOCK_N_ARRAY{ {{ block_n_array | join(', ') }} };
{% endif -%}
{% if block_k_array is defined and block_k_array -%}
constexpr std::array<int, {{ block_k_array|length }}> {{ kernel_signature }}_BLOCK_K_ARRAY{ {{ block_k_array | join(', ') }} };
{% endif -%}
{% if block_size_array is defined and block_size_array -%}
constexpr std::array<int, {{ block_size_array|length }}> {{ kernel_signature }}_BLOCK_SIZE{ {{ block_size_array | join(', ') }} };
{% endif -%}
constexpr std::array<int, {{ shared_mem_bytes|length }}> {{ kernel_signature }}_SHARED_MEM_BYTES{ {{ shared_mem_bytes | join(', ') }} };
constexpr std::array<const char*, {{ ptx_array|length }}> {{ kernel_signature }}_PTX_ARRAY = {
    {% for ptx in ptx_array -%}
    R"({{ ptx }})" {% if not loop.last %},{% else %} {% endif %}  // Index {{ loop.index0 }}
    {% endfor -%}
};
    """)
    
    template_vars = {
        "kernel_name": kernel_name,
        "kernel_signature": kernel_signature,
        "input_dims": input_dims,
        "block_m_array": block_m_array,
        "block_n_array": block_n_array,
        "block_k_array": block_k_array,
        "block_size_array": block_size_array,
        "shared_mem_bytes": shared_mem_bytes,
        "ptx_array": ptx_array
    }
    
    generated_code = cpp_template.render(template_vars)
    # print(generated_code)
    return generated_code