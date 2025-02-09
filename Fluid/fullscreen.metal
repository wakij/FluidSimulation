//
//  fullscreen.metal
//  Fluid
//
//  Created by wakita tomoshige on 2025/02/09.
//

#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 vMatrix;
    float4x4 pMatrix;
    float4x4 invProjectionMatrix;
    float2 texelSize;
    float size;
    float sphereRadius; //size/2
};

struct FullScreenVertexIn {
    float2 position;
    float2 texCoord;
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut fullscreen_vertex(
                                   uint vid [[vertex_id]],
                                   constant FullScreenVertexIn *vertices [[buffer(0)]]
                                   )
{
    VertexOut out;
    out.texCoord = vertices[vid].texCoord;
    out.position = float4(vertices[vid].position, 0.0, 1.0);
    return out;
}

float3 computeViewPosFromUVDepth(
                                 float2 texCoord,
                                 float depth,
                                 float4x4 projectionMatrix,
                                 float4x4 invProjectionMatrix)
{
    float4 ndc;
    ndc.xy = texCoord * 2.0 - 1.0;
    ndc.z = projectionMatrix[2].z + projectionMatrix[3].z / depth;
    ndc.w = 1.0;

    float4 eyePos = invProjectionMatrix * ndc;
    eyePos.xyz /= eyePos.w;
    return eyePos.xyz;
}

float3 getViewPosFromTexCoord(
                              float2 texCoord,
                              float4x4 projectionMatrix,
                              float4x4 invProjectionMatrix,
                              texture2d<float> texture,
                              sampler sampler
                              )
{
    float depth = texture.sample(sampler, texCoord).r;
    return computeViewPosFromUVDepth(texCoord, depth, projectionMatrix, invProjectionMatrix);
}

fragment float4 fullscreen_fragment(
                                    VertexOut in [[stage_in]],
                                    texture2d<float> texture [[ texture(0) ]],
                                    constant Uniforms &uniform [[ buffer(0)]]
                                    )
{
    constexpr sampler depthSampler;
    float2 texCoord = in.texCoord;
    texCoord.y = 1.0 - texCoord.y;

    float depth = texture.sample(depthSampler, texCoord).r;
                    
    float3 viewPos = computeViewPosFromUVDepth(texCoord, depth, uniform.pMatrix, uniform.invProjectionMatrix);
//
//    // calculate normal
    float3 ddx = getViewPosFromTexCoord(texCoord + float2(uniform.texelSize.x, 0.), uniform.pMatrix, uniform.invProjectionMatrix, texture, depthSampler) - viewPos;
    float3 ddy = getViewPosFromTexCoord(texCoord + float2(0., uniform.texelSize.y), uniform.pMatrix, uniform.invProjectionMatrix, texture, depthSampler) - viewPos;
//    
    float3 ddx2 = viewPos - getViewPosFromTexCoord(texCoord + float2(-uniform.texelSize.x, 0.), uniform.pMatrix, uniform.invProjectionMatrix, texture, depthSampler);
    float3 ddy2 = viewPos - getViewPosFromTexCoord(texCoord + float2(0., -uniform.texelSize.y), uniform.pMatrix, uniform.invProjectionMatrix, texture, depthSampler);
//
    if (abs(ddx.z) > abs(ddx2.z)) {
        ddx = ddx2;
    }
    if (abs(ddy.z) > abs(ddy2.z)) {
        ddy = ddy2;
    }
//                    // 法線
    float3 normal = normalize(cross(ddx, ddy));
    return float4(normal, 1.0);
}

