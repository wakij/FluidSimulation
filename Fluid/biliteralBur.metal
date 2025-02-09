//
//  biliteralBur.metal
//  Fluid
//
//  Created by wakita tomoshige on 2025/02/09.
//

#include <metal_stdlib>
using namespace metal;

struct BiliteralVertexIn {
    float2 position;
    float2 texCoord;
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

struct BiliteralUniforms {
    int maxFilterSize;
    float2 blurDir;
    float projectedParticleConstant;
    float depthThreshold;
};

vertex VertexOut biliteral_vertex(
                                   uint vid [[vertex_id]],
                                   constant BiliteralVertexIn *vertices [[buffer(0)]]
                                   )
{
    VertexOut out;
    out.texCoord = vertices[vid].texCoord;
    out.position = float4(vertices[vid].position, 0.0, 1.0);
    return out;
}


fragment float4 biliteral_fragment(
                                   VertexOut in [[stage_in]],
                                   texture2d<float> texture [[ texture(0) ]],
                                   constant BiliteralUniforms &uniform [[ buffer(0) ]]
                                    )
{
    constexpr sampler depthSampler;
    float2 texCoord = in.texCoord;
    float depth = texture.sample(depthSampler, texCoord).r;

    int filterSize = min(uniform.maxFilterSize, int(ceil(uniform.projectedParticleConstant / abs(depth))));
    float sigma = float(filterSize) / 3.0;
    float two_sigma2 = 2.0 * sigma * sigma;

    float sigmaDepth = uniform.depthThreshold / 3.0;
    float two_sigmaDepth2 = 2.0 * sigmaDepth * sigmaDepth;

    float sum = 0.;
    float wsum = 0.;

    for (int x = -filterSize; x <= filterSize; ++x) {
        float2 coords = float2(x);
        float sampleDepthVel = texture.sample(depthSampler, texCoord + coords * uniform.blurDir).r;

        float r = dot(coords, coords);
        float w = exp(-r / two_sigma2);

        float rDepth = sampleDepthVel - depth;
        float wd = exp(-rDepth * rDepth / two_sigmaDepth2);

        sum += sampleDepthVel * w * wd;
        wsum += w * wd;
    }
    
    return sum / wsum;
}


