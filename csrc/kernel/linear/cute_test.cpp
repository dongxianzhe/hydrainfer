#include<iostream>
#include<gtest/gtest.h>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

TEST(cute, tensor){
    torch::Tensor t = torch::randn({2, 3});
    std::cout << t << std::endl;

    cute::Tensor tensor = cute::make_tensor(static_cast<float*>(t.data_ptr()), cute::make_shape(cute::Int<2>{}, cute::Int<3>{}), cute::make_stride(cute::Int<3>{}, cute::Int<1>{}));
    cute::print(tensor);std::cout << std::endl;
    
    for (int i = 0; i < cute::size<0>(tensor); ++i) {
        for (int j = 0; j < cute::size<1>(tensor); ++j) {
            tensor(i, j) ++;
        }
    }

    std::cout << t << std::endl;
}

TEST(cute, local_tile){
    using namespace cute;
    torch::Tensor a = torch::randn({1024, 256});
    Tensor t = make_tensor(
        static_cast<float*>(a.data_ptr()), 
        make_shape(Int<1024>{}, Int<256>{}), 
        make_stride(Int<256>{}, Int<1>{})
    );
    print(t);puts("");
    Tensor s1 = local_tile(t, make_tile(Int<128>{}, Int<32>{}), make_coord(_, _));
    Tensor s2 = local_tile(t, make_tile(Int<128>{}, Int<32>{}), make_coord(2, _));
    Tensor s3 = local_tile(t, make_tile(Int<128>{}, Int<32>{}), make_coord(_, 3));
    Tensor s4 = local_tile(t, make_tile(Int<128>{}, Int<32>{}), make_coord(2, 3));
    print(s1);puts("");
    print(s2);puts("");
    print(s3);puts("");
    print(s4);puts("");
}

TEST(cute, slice){
    using namespace cute;
    torch::Tensor a = torch::randn({1024, 256});
    Tensor t = make_tensor(
        static_cast<float*>(a.data_ptr()), 
        make_shape(Int<1024>{}, Int<256>{}), 
        make_stride(Int<256>{}, Int<1>{})
    );
    print(t);puts("");
    Tensor s = local_tile(t, make_tile(Int<128>{}, Int<32>{}), make_coord(2, _));
    print(s);puts("");
    auto s_slice = s(make_coord(_, _, 1));
    print(s_slice);puts("");
}