#include "kernels.cuh"
using namespace at;

torch::Tensor similar_cuda_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_loc,
        const int kH, const int kW,
        const bool casual_mask
) {
    TypeCheck(x_ori);
    TypeCheck(x_loc);
    AT_ASSERTM(!casual_mask || (kH & 1 == 1 && kW & 1 == 1), "If casual_mask is true, the kernel size must be odd!");
    const int batch = x_ori.size(0);
    const int channels = x_ori.size(1);
    const int height = x_ori.size(2);
    const int width = x_ori.size(3);

    const int batch_loc = x_loc.size(0);
    AT_ASSERTM(batch % batch_loc == 0, "cannot use auto expand");

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = casual_mask ? (kH * kW >> 1) + 1: kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;
    const int per_output = height * width * patch;
    auto output = torch::empty({batch, height, width, patch}, x_ori.options());

    int start_inp = 0, start_out = 0, start_inp_loc = 0;
    for (int j = 0; j < batch_loc; ++j){
        for (int i = 0; i < batch / batch_loc; ++i) {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_ori.scalar_type(), "similar_cuda_forward", 
                ([&] {
                        f_cc2k<scalar_t, double>(
                            at::cuda::getCurrentCUDAStream(),
                            x_ori.data_ptr<scalar_t>() + start_inp,
                            x_loc.data_ptr<scalar_t>() + start_inp_loc,
                            kH, kW, rH, rW,
                            patch, channels, height, width,
                            per_channel,
                            output.data_ptr<scalar_t>() + start_out
                        );
                }
                )
            );
            start_inp += per_input;
            start_out += per_output;
        }
        start_inp_loc += per_input;
    }

    return output;
}

//////////////////////////////////////////////////////////////

torch::Tensor similar_cuda_backward(
        const torch::Tensor &x,
        const torch::Tensor &grad_out,
        const int kH, const int kW,
        const int batch_loc,
        const bool is_ori,
        const bool casual_mask
) {
    TypeCheck(x);
    const int batch = grad_out.size(0);
    const int channels = x.size(1);
    const int height = x.size(2);
    const int width = x.size(3);

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = casual_mask ? (kH * kW >> 1) + 1: kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;

    if (is_ori){ // x is loc
        auto grad_inp = torch::empty({batch, channels, height, width}, x.options());
        int start_inp = 0, start_inp_loc = 0;
        for (int j = 0; j < batch_loc; ++j) {
            for (int i = 0; i < batch / batch_loc; ++i) {
                auto grad_out_row = grad_out.select(0, i + j * batch / batch_loc);
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "similar_cuda_backward_ori", 
                    ([&] {
                            f_ck2c_ori<scalar_t, float>(
                                at::cuda::getCurrentCUDAStream(),
                                x.data_ptr<scalar_t>() + start_inp_loc,
                                grad_out_row.data_ptr<scalar_t>(),
                                kH, kW, rH, rW,
                                patch, channels,
                                height, width,
                                per_channel, per_input,
                                grad_inp.data_ptr<scalar_t>() + start_inp
                            );
                    }
                    )
                    );
                start_inp += per_input;
            }
            start_inp_loc += per_input;
        }
        return grad_inp;
    } else{ // x is ori
        auto grad_inp = torch::empty({batch_loc, channels, height, width}, x.options());
        int start_inp = 0, start_inp_loc = 0;
        for (int j = 0; j < batch_loc; ++j) {
            bool is_accumulate = false;
            for (int i = 0; i < batch / batch_loc; ++i) {
                auto grad_out_row = grad_out.select(0, i + j * batch / batch_loc);
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "similar_cuda_backward_loc", 
                ([&] {
                        f_ck2c_loc<scalar_t, float>(
                                at::cuda::getCurrentCUDAStream(),
                                x.data_ptr<scalar_t>() + start_inp,
                                grad_out_row.data_ptr<scalar_t>(),
                                kH, kW, rH, rW,
                                patch, channels,
                                height, width,
                                per_channel, per_input,
                                grad_inp.data_ptr<scalar_t>() + start_inp_loc,
                                is_accumulate
                        );
                }
                )
                );
                start_inp += per_input;
                is_accumulate = true;
            }
            start_inp_loc += per_input;
        }
        return grad_inp;
    }
    
}
