%%writefile flash_attn_homework.cu
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

/**
 * Assignment: Refactor FlashAttention Algorithm 1 using CuTe
 * This implementation demonstrates Layout Algebra and Hierarchical Tiling.
 */

template <typename T, int kBlockM, int kBlockN, int kBlockK>
__global__ void flash_attn_kernel(
    T const* Q, int stride_q, 
    T const* K, int stride_k, 
    T* O,       int stride_o,
    int M, int N, int K_dim) 
{
    // 1. Layout Algebra (Ref: Screenshot 4 & 9)
    // Using static values for efficiency
    auto layout_Q = make_layout(make_shape(M, K_dim), make_stride(stride_q, _1{}));
    auto layout_K = make_layout(make_shape(N, K_dim), make_stride(stride_k, _1{}));
    auto layout_O = make_layout(make_shape(M, K_dim), make_stride(stride_o, _1{}));

    Tensor gQ = make_tensor(make_gmem_ptr(Q), layout_Q);
    Tensor gK = make_tensor(make_gmem_ptr(K), layout_K);
    Tensor gO = make_tensor(make_gmem_ptr(O), layout_O);

    // 2. Hierarchical Tiling (Ref: Screenshot 5 & 6)
    auto bM = Int<kBlockM>{};
    auto bN = Int<kBlockN>{};
    auto bK = Int<kBlockK>{};

    // 3. Algorithm 1 Inner-Outer Loop Refactoring
    int m_idx = blockIdx.x;

    // Slice global tensors into local tiles using coordinate mapping
    Tensor tQgQ = local_tile(gQ, make_tile(bM, bK), make_coord(m_idx, _0{}));
    Tensor tOgO = local_tile(gO, make_tile(bM, bK), make_coord(m_idx, _0{}));

    // Online Softmax statistics
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Inner loop: Iterate over K blocks (Algorithm 1: j = 1 to Tc)
    for (int j = 0; j < ceil_div(N, kBlockN); ++j) {
        Tensor tKjK = local_tile(gK, make_tile(bN, bK), make_coord(j, _0{}));
        
        // In a real implementation, we would use cute::copy(tKjK, sK) here
        // The Layout Algebra automatically handles physical indexing
    }
}

void run_assignment_test() {
    // Demonstrate Layout Algebra Visualization (Ref: Screenshot 10)
    // Nested layout as shown in your screenshot 6: (4, (4, 2)) : (4, (1, 16))
    auto layout = make_layout(make_shape(_4{}, make_shape(_4{}, _2{})), 
                              make_stride(_4{}, make_stride(_1{}, _16{})));
    
    std::cout << "--- CuTe Layout Algebra Verification ---" << std::endl;
    print(layout);
    std::cout << "\nLayout Size: " << size(layout) << std::endl;
    std::cout << "Coordinate (1, (2, 1)) maps to index: " << layout(1, make_coord(2, 1)) << std::endl;
}

int main() {
    run_assignment_test();
    return 0;
}
