#include <cute/tensor.hpp>
#include <cstdio>

using namespace cute;

void run_layout_discussion() {
    auto shape  = make_shape(Int<4>{}, make_shape(Int<4>{}, Int<2>{}));
    auto stride = make_stride(Int<4>{}, make_shape(Int<1>{}, Int<16>{}));
    auto layout = make_layout(shape, stride);

    auto coord = make_coord(2, make_coord(3, 1));
    int index = (int)layout(coord);

    printf("Index at coord (2, (3, 1)): %d\n", index);
}

template <typename T, int kBlockM, int kBlockN, int kHeadDim>
__global__ void flash_attn_kernel(T const* Q, T const* K, T const* V, T* O, int M, int N, float scale) {
    auto g_layout = make_layout(make_shape(M, Int<kHeadDim>{}), GenRowMajor{});
    Tensor gQ = make_tensor(make_gmem_ptr(Q), g_layout);
    Tensor gK = make_tensor(make_gmem_ptr(K), g_layout);
    Tensor gV = make_tensor(make_gmem_ptr(V), g_layout);
    Tensor gO = make_tensor(make_gmem_ptr(O), g_layout);

    int idxM = blockIdx.x;
    auto bM = Int<kBlockM>{};
    auto bN = Int<kBlockN>{};
    auto bD = Int<kHeadDim>{};

    __shared__ T sQ_raw[kBlockM * kHeadDim];
    __shared__ T sK_raw[kBlockN * kHeadDim];
    __shared__ T sV_raw[kBlockN * kHeadDim];

    Tensor sQ = make_tensor(make_smem_ptr(sQ_raw), make_layout(make_shape(bM, bD), GenRowMajor{}));
    Tensor sK = make_tensor(make_smem_ptr(sK_raw), make_layout(make_shape(bN, bD), GenRowMajor{}));
    Tensor sV = make_tensor(make_smem_ptr(sV_raw), make_layout(make_shape(bN, bD), GenRowMajor{}));

    float m_i = -1e20f;
    float l_i = 0.0f;

    copy(local_tile(gQ, make_shape(bM, bD), make_coord(idxM, 0)), sQ);

    for (int j = 0; j < (N / kBlockN); ++j) {
        copy(local_tile(gK, make_shape(bN, bD), make_coord(j, 0)), sK);
        copy(local_tile(gV, make_shape(bN, bD), make_coord(j, 0)), sV);
        __syncthreads();

        // Core Math: S = QK^T, Online Softmax updates for m_i, l_i, O
        
        __syncthreads();
    }

    copy(sQ, local_tile(gO, make_shape(bM, bD), make_coord(idxM, 0))); 
}

int main() {
    run_layout_discussion();
    return 0;
}
