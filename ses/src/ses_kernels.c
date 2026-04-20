/**
 * SES CPU kernels — C implementation of inference primitives.
 * Compiled as shared library for Python ctypes testing,
 * later integrated into the full inference engine.
 */
#include <math.h>
#include <string.h>

/* RMS Normalization: out = x * weight / sqrt(mean(x^2) + eps) */
void c_rms_norm(const float* x, const float* weight, float* out,
                int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++)
        sum_sq += x[i] * x[i];

    float rms = 1.0f / sqrtf(sum_sq / (float)dim + eps);

    for (int i = 0; i < dim; i++)
        out[i] = x[i] * rms * weight[i];
}

/* SwiGLU activation: out = gate * sigmoid(gate) * up */
void c_swiglu(const float* gate, const float* up, float* out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        /* Clip for numerical stability */
        if (g > 88.0f) g = 88.0f;
        if (g < -88.0f) g = -88.0f;
        float sigmoid = 1.0f / (1.0f + expf(-g));
        out[i] = g * sigmoid * up[i];
    }
}

/* MoE combine: out (initialized with residual) += sum(weight[k] * experts[k]) */
void c_moe_combine(const float* experts_flat, const float* weights,
                    float* out, int K, int dim) {
    /* experts_flat: [K * dim] contiguous, out already contains residual */
    for (int k = 0; k < K; k++) {
        float w = weights[k];
        const float* expert = experts_flat + k * dim;
        for (int i = 0; i < dim; i++)
            out[i] += w * expert[i];
    }
}
