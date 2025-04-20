
/*
CREDITS : 
MADE BY ME
CORRECTED BY CLAUDE 3.7
*/


#include "ATen/Dispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/float_power_ops.h"
#include "torch/csrc/autograd/custom_function.h"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda() , #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous() , #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


template <typename scalar_t>
__global__ void ntxent_loss_frwd_kernel(
    const scalar_t* __restrict__ z_norm,
    scalar_t* __restrict__ logits,
    int batch_size,
    int feature_dim,
    scalar_t temp
){
    const int N = 2 * batch_size;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N){
        //positive pair similarity

        scalar_t positive_sim;

        if(idx < batch_size){
            positive_sim = 0.0;

            for(int d = 0 ; d < feature_dim ; d++){

                positive_sim += z_norm[idx * feature_dim + d] * z_norm[(batch_size + idx) * feature_dim + d];


            }
            
        }
        else{
            positive_sim = 0.0;
            for(int d = 0 ; d < feature_dim ; d++){
                positive_sim += z_norm[idx *feature_dim + d] * z_norm[(idx - batch_size) * feature_dim + d];
            }
        }
        
        logits[idx * (N -1 + 1)] = positive_sim / temp;

        int logit_index = 1;

        for(int j = 0 ; j < N ; j++){
            //skipping the self and positive pairs
            if(j == idx || (idx < batch_size && j == (batch_size + idx))|| (idx >= batch_size && j == (idx - batch_size))){
                continue;
            }

            scalar_t sim = 0.0f;

            for(int d = 0 ; d < feature_dim; d++){
                sim += z_norm[idx *feature_dim + d] * z_norm[j * feature_dim + d];

            }


            logits[idx * (N - 1 + 1) + logit_index] = sim / temp;
            logit_index ++;

        }

    }
    
}



template <typename scalar_t>
__global__ void ntxent_loss_bckwd_kernel(
    scalar_t* __restrict__ grad_z,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ z_norm,
    const scalar_t* __restrict__ logits,
    const scalar_t* __restrict__ probs,
    int batch_size,
    int feature_dim,
    scalar_t temp
){

    const int N = 2 * batch_size;

    //index
    const int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    const int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;


    if (idx < N && feature_idx < feature_dim){
        scalar_t grad_sum = 0.0f;

        int pos_idx;

        if(idx < batch_size){
            pos_idx = batch_size + idx;
        }
        else{
            pos_idx = idx - batch_size;
        }

        const scalar_t pos_prob = probs[idx * (N - 1 + 1)];
        grad_sum += (pos_prob -1.0) * z_norm[pos_idx * feature_dim + feature_idx] / temp;

        int logit_idx = 1;
        for(int j = 0 ; j < N ; j++){
            if(j == idx || j == pos_idx){
                continue;
            }

            const scalar_t neg_prob = probs[idx * (N -1 + 1) + logit_idx];

            grad_sum += neg_prob * z_norm[j * feature_dim + feature_idx] / temp;
            logit_idx ++;
        }

        grad_z[idx * feature_dim + feature_idx] = grad_sum * grad_out[0] / N;
    }
}


torch::Tensor normalize_features(torch::Tensor z) {
    auto norm = torch::norm(z, 2, 1, true);
    return z / norm;
}

torch::Tensor compute_softmax(torch::Tensor logits) {
    auto max_logits = std::get<0>(torch::max(logits, 1, true)).expand_as(logits);
    auto exp_logits = torch::exp(logits - max_logits);
    auto sum_exp_logits = torch::sum(exp_logits, 1, true);
    return exp_logits / sum_exp_logits;
}




// forward pass --
std::vector<torch::Tensor> ntxent_loss_frwd(
    torch::Tensor z_i,
    torch::Tensor z_j,
    float temp
){
    //safety check first
    CHECK_INPUT(z_i);
    CHECK_INPUT(z_j);

    auto batch_size = z_i.size(0);
    auto feature_dim = z_i.size(1);
    const int N = 2 * batch_size;



    auto z = torch::cat({z_i, z_j}, 0);
    auto z_normalized = normalize_features(z);


    //allocating space for logits
    auto logits = torch::empty({N, N-1+1}, z_i.options());

    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    AT_DISPATCH_FLOATING_TYPES(z_i.scalar_type(), "ntxent_loss_forward_kernel",([&] {
        ntxent_loss_frwd_kernel<scalar_t><<<numBlocks , threadsPerBlock>>>(
            z_normalized.data_ptr<scalar_t>(),
            logits.data_ptr<scalar_t>(),
            batch_size,
            feature_dim,
            static_cast<scalar_t>(temp)
        );
    }

    ));

    auto probs = compute_softmax(logits);

    auto loss = -torch::log(probs.select(1, 0)).sum() / N;
    
    return {loss, z_normalized, logits, probs};
}



torch::Tensor ntxent_loss_bckwd(
    torch::Tensor grad_output,
    torch::Tensor z_normalized,
    torch::Tensor logits,
    torch::Tensor probs,
    float temp
){
    CHECK_INPUT(grad_output);
    CHECK_INPUT(z_normalized);
    CHECK_INPUT(logits);
    CHECK_INPUT(probs);

    auto batch_size = z_normalized.size(0) / 2;
    auto feature_dim = z_normalized.size(1);
    const int N = 2 * batch_size;

    auto grad_z = torch::zeros_like(z_normalized);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (feature_dim + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    AT_DISPATCH_FLOATING_TYPES(z_normalized.scalar_type(), "ntxent_loss_backward_kernel", ([&] {
        ntxent_loss_bckwd_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            grad_z.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            z_normalized.data_ptr<scalar_t>(),
            logits.data_ptr<scalar_t>(),
            probs.data_ptr<scalar_t>(),
            batch_size,
            feature_dim,
            static_cast<scalar_t>(temp)
        );
    }));

    return grad_z;
}


class FusedNTXentLoss : public torch::autograd::Function<FusedNTXentLoss>{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor z_i,
        torch::Tensor z_j,
        float temperature) {
        
        // Run forward pass
        auto outputs = ntxent_loss_frwd(z_i, z_j, temperature);
        auto loss = outputs[0];
        auto z_normalized = outputs[1];
        auto logits = outputs[2];
        auto probs = outputs[3];
        
        // Save for backward
        ctx->save_for_backward({z_normalized, logits, probs});
        ctx->saved_data["temperature"] = temperature;
        
        return {loss};
    
    
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        
        // Get saved tensors
        auto saved = ctx->get_saved_variables();
        auto z_normalized = saved[0];
        auto logits = saved[1];
        auto probs = saved[2];
        auto temperature = ctx->saved_data["temperature"].toDouble();
        
        // Run backward pass
        auto grad_z = ntxent_loss_bckwd(
            grad_outputs[0],
            z_normalized,
            logits,
            probs,
            temperature
        );
        
        // Split gradient for z_i and z_j
        auto batch_size = grad_z.size(0) / 2;
        auto grad_z_i = grad_z.slice(0, 0, batch_size);
        auto grad_z_j = grad_z.slice(0, batch_size, 2 * batch_size);
        
        return {grad_z_i, grad_z_j, torch::Tensor()};
    }

};

//python binding
torch::Tensor ntxent_loss_cuda(
    torch::Tensor z_i,
    torch::Tensor z_j,
    float temperature) {
    
    return FusedNTXentLoss::apply(z_i, z_j, temperature)[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ntxent_loss", &ntxent_loss_cuda, "NTXentLoss CUDA implementation");
}