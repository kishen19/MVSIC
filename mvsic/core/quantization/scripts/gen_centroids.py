import numpy as np
import os
import sys
from scipy.cluster.vq import kmeans2, vq

def generate_vq_centroids(d, k, num_samples=2_000_000):
    """
    Generates optimized float and int8 centroids for D-dimensional N(0,1) data.
    
    Returns:
        attenuation: The (1 - MSE) scaling factor.
        C_float: The exact float centroids (K x D).
        C_int: The scaled Int8 centroids (K x D).
        Sq_int_norms: The scaled, squared integer norms (length K).
        scale: The multiplier used to map floats to the [-127, 127] range.
    """
    # 1. Generate independent N(0,1) in D dimensions
    X = np.random.randn(num_samples, d)
    
    # 2. Run k-means clustering
    if d == 1:
        # Enforce perfect symmetry for 1D Scalar Quantization
        X_abs = np.abs(X)
        centroids_half, _ = kmeans2(X_abs, k // 2, minit='++')
        # Ensure it's a 2D array of shape (K, 1) to match VQ output
        centroids_half = centroids_half.reshape(-1, 1)
        C_float = np.sort(np.concatenate([centroids_half, -centroids_half]), axis=0)
    else:
        # Standard k-means for Vector Quantization
        C_float, _ = kmeans2(X, k, minit='++')
    
    # 3. Scale to Int8 (Anchor absolute max coordinate across all dims to 127)
    max_val = np.max(np.abs(C_float))
    scale = 127.0 / max_val
    C_int = np.round(C_float * scale).astype(int)
    
    # 4. Compute Attenuation Factor (1 - MSE)
    labels, _ = vq(X, C_float)
    quantized_X = C_float[labels]
    
    # Expected squared norm: E[||x||^2] (Theoretical = d)
    E_x2 = np.mean(np.sum(X ** 2, axis=1))
    
    # Expected quantized squared norm: E[||x_hat||^2]
    E_qx2 = np.mean(np.sum(quantized_X ** 2, axis=1))
    
    attenuation = E_qx2 / E_x2
    
    # 5. Compute Scaled Squared Int8 Norms
    Sq_int_norms = np.sum(C_int ** 2, axis=1) * attenuation
    
    return attenuation, C_float, C_int, Sq_int_norms, scale


def emit_cpp_block(d, k, attenuation, C_float, C_int, Sq_int_norms, scale):
    """Emit one (D, K) block for the C++ header."""
    lines = []
    lines.append(f"// D = {d}, K = {k}")
    lines.append(f"static constexpr float kPQ_Attenuation_D{d}_K{k} = {attenuation:.8f}f;")
    lines.append(f"static constexpr float kPQ_Int8Scale_D{d}_K{k} = {scale:.8f}f;")
    # Centroids float [K][D] row-major
    lines.append(f"static constexpr float kPQ_CentroidsFloat_D{d}_K{k}[{k}][{d}] = {{")
    for i in range(k):
        row = ", ".join(f"{C_float[i, j]:.8f}f" for j in range(d))
        lines.append(f"  {{{row}}},")
    lines.append("};")
    # Centroids int8 [K][D]; clamp to [-127, 127]
    C_int_clamped = np.clip(C_int, -127, 127).astype(np.int32)
    lines.append(f"static constexpr int8_t kPQ_CentroidsInt8_D{d}_K{k}[{k}][{d}] = {{")
    for i in range(k):
        row = ", ".join(str(C_int_clamped[i, j]) for j in range(d))
        lines.append(f"  {{{row}}},")
    lines.append("};")
    lines.append(f"static constexpr float kPQ_SqIntNorms_D{d}_K{k}[{k}] = {{")
    norms_str = ", ".join(f"{s:.8f}f" for s in Sq_int_norms)
    lines.append(f"  {norms_str}")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def write_header_file(out_path, dimensions, k_values, num_samples=2_000_000):
    """Generate all (D,K) codebooks and write the C++ header."""
    preamble = """#pragma once
#include <cstdint>

namespace mvsic {
namespace turboquant_pq_4bit {

"""
    footer = """}  // namespace turboquant_pq_4bit
}  // namespace mvsic
"""
    blocks = []
    for d in dimensions:
        for k in k_values:
            attenuation, C_float, C_int, Sq_int_norms, scale = generate_vq_centroids(
                d, k, num_samples=num_samples
            )
            blocks.append(
                emit_cpp_block(d, k, attenuation, C_float, C_int, Sq_int_norms, scale)
            )
    with open(out_path, "w") as f:
        f.write(preamble)
        f.write("\n".join(blocks))
        f.write(footer)


if __name__ == "__main__":
    np.random.seed(42)
    
    dimensions = [1, 2, 4, 8, 16]
    k_values = [2, 4, 8, 16, 32]
    
    print(f"{'D':<4} | {'K':<4} | {'Bits/Dim':<10} | {'Attenuation':<12} | {'Int8 Scale':<12}")
    print("-" * 55)
    
    for d in dimensions:
        for k in k_values:
            attenuation, C_float, C_int, Sq_int_norms, scale = generate_vq_centroids(d, k)
            
            # Bits per dimension = log2(K) / D
            bits_per_dim = np.log2(k) / d
            
            print(f"{d:<4} | {k:<4} | {bits_per_dim:<10.3f} | {attenuation:<12.4f} | {scale:<12.4f}")
        print("-" * 55)

    # Write C++ header next to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    header_path = os.path.join(script_dir, "..", "turboquant_pq_codebooks.h")
    header_path = os.path.normpath(header_path)
    print(f"\nWriting header to {header_path} ...")
    write_header_file(header_path, dimensions, k_values)
    print("Done.")