#include <metal_stdlib>
using namespace metal;

constant float2 offsets[4] = {
    float2(0.0, 0.0),
    float2(1.0, 0.0),
    float2(0.0, 1.0),
    float2(1.0, 1.0),
};

struct Uniforms {
    float4x4 vMatrix;
    float4x4 pMatrix;
    float4x4 invProjectionMatrix;
    float2 texelSize;
    float size;
    float sphereRadius; //size/2
};

struct VertexOut {
    float4 position [[position]];
    float3 viewPos;
    float2 uv;
};

vertex VertexOut depthMap_vertex(
                             uint vid [[vertex_id]],
                             constant Uniforms &uniforms [[buffer(0)]],
                             constant float3 *positions [[buffer(1)]]
                          )
{
    VertexOut out;
    int offsetIndex = vid % 4;
    int positionIndex = vid / 4;
    float3 position = positions[positionIndex];
    float3 cornerPos;
    float2 offset = offsets[offsetIndex];
    cornerPos.xy = float2(offset.x - 0.5, offset.y - 0.5) * uniforms.size;
    cornerPos.z = 0;
    
    float3 viewPos = (uniforms.vMatrix * float4(position, 1.0)).xyz;
    
    out.position = uniforms.pMatrix * float4(viewPos + cornerPos, 1.0);
    out.viewPos = viewPos;
    out.uv = offset;
    return out;
}


struct FragmentOutput {
    float color [[color(0)]];
    float depth [[depth(any)]];
};

fragment FragmentOutput depthMap_fragment(VertexOut in [[stage_in]],
                                      constant Uniforms &uniform [[buffer(0)]]
                              )
{
    FragmentOutput out;
    float3 normal;
    normal.xy = in.uv * 2.0 - 1.0;
    float r2 = dot(normal.xy, normal.xy);
    if (r2 > 1.0) {
        discard_fragment();
    }
    normal.z = sqrt(1.0 - r2);
    
    float4 realViewPos = float4(in.viewPos + normal * uniform.sphereRadius, 1.0);
    float4 clipSpacePos = uniform.pMatrix * realViewPos;
    
    out.color = realViewPos.z;
    out.depth = (clipSpacePos.z / clipSpacePos.w);
    
    return out;
}
