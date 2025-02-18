//
//  copyPosition.metal
//  Fluid
//
//  Created by wakita tomoshige on 2025/02/13.
//

//Particleからpositionだけ抜く

#include <metal_stdlib>
using namespace metal;

struct Particle {
    float3 position;
    float3 v;
    float3 force;
    float3 lastAcceration;
    float density;
    float nearDensity;
};

kernel void copyPosition(
                         device const Particle *particles [[buffer(0)]],
                         device float3 *positions [[buffer(1)]],
                         constant uint &particleNum [[buffer(2)]],
                         uint tid  [[ thread_position_in_grid ]]
)
{
    if (tid < particleNum) {
        positions[tid] = particles[tid].position;
    }
                               
}
