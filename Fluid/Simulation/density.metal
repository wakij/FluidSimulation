//
//  density.metal
//  Fluid
//
//  Created by wakita tomoshige on 2025/02/11.
//

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

struct Environment {
    int xGrids;
    int yGrids;
    int zGrids;
    float cellSize;
    float xHalf;
    float yHalf;
    float zHalf;
    float offset;
};

struct SPHParams {
    float mass;
    float kernelRadius;
    float kernelRadiusPow2;
    float kernelRadiusPow5;
    float kernelRadiusPow6;
    float kernelRadiusPow9;
    float dt;
    float stiffness;
    float nearStiffness;
    float restDensity;
    float viscosity;
    uint n;
};

float nearDensityKernel(float r, SPHParams params) {
    float scale = 15.0 / (3.1415926535 * params.kernelRadiusPow6);
    float d = params.kernelRadius - r;
    return scale * d * d * d;
}

float densityKernel(float r2, SPHParams params){
    float scale = 315.0 / (64. * 3.1415926535 * params.kernelRadiusPow9);
    float dd = params.kernelRadiusPow2 - r2;
    return scale * dd * dd * dd;
}

int3 cellPosition(float3 v, Environment env){
    int xi = int(floor((v.x + env.xHalf + env.offset) / env.cellSize));
    int yi = int(floor((v.y + env.yHalf + env.offset) / env.cellSize));
    int zi = int(floor((v.z + env.zHalf + env.offset) / env.cellSize));
    return int3(xi, yi, zi);
}

int cellNumberFromId(int xi,int yi,int zi, Environment env) {
    return xi + yi * env.xGrids + zi * env.xGrids * env.yGrids;
}

kernel void computeDensity(
                           device Particle *particles [[buffer(0)]],
                           device const Particle *sortedParticles [[buffer(1)]],
                           device const uint *prefixSum [[buffer(2)]],
                           constant Environment &env [[buffer(3)]],
                           constant SPHParams &params [[buffer(4)]],
                           uint tid  [[ thread_position_in_grid ]]
)
{
    if (tid < params.n) {
        
        particles[tid].density = 0.0;
        particles[tid].nearDensity = 0.0;
        float3 pos_i = particles[tid].position;

        int3 v = cellPosition(pos_i, env);
        if (v.x < env.xGrids && 0 <= v.x &&
            v.y < env.yGrids && 0 <= v.y &&
            v.z < env.zGrids && 0 <= v.z)
        {
            for (int dz = max(-1, -v.z); dz <= min(1, env.zGrids - v.z - 1); dz++) {
                for (int dy = max(-1, -v.y); dy <= min(1, env.yGrids - v.y - 1); dy++) {
                    int dxMin = max(-1, -v.x);
                    int dxMax = min(1, env.xGrids - v.x - 1);
                    int startCellNum = cellNumberFromId(v.x + dxMin, v.y + dy, v.z + dz, env);
                    int endCellNum = cellNumberFromId(v.x + dxMax, v.y + dy, v.z + dz, env);
                    uint start = prefixSum[startCellNum];
                    uint end = prefixSum[endCellNum + 1];
                    for (uint j = start; j < end; j++) {
                        float3 pos_j = sortedParticles[j].position;
                        float r2 = dot(pos_i - pos_j, pos_i - pos_j);
                        
                       if (r2 < params.kernelRadiusPow2) {
                           particles[tid].density += params.mass * densityKernel(r2, params);
                           particles[tid].nearDensity += params.mass * nearDensityKernel(sqrt(r2), params);
                       }
                   }
               }
           }
       }
   }
}
