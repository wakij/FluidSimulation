//
//  force.metal
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

static inline int3 cellPosition(float3 v, Environment env){
    int xi = int(floor((v.x + env.xHalf + env.offset) / env.cellSize));
    int yi = int(floor((v.y + env.yHalf + env.offset) / env.cellSize));
    int zi = int(floor((v.z + env.zHalf + env.offset) / env.cellSize));
    return int3(xi, yi, zi);
}

static inline int cellNumberFromId(int xi,int yi,int zi, Environment env) {
    return xi + yi * env.xGrids + zi * env.xGrids * env.yGrids;
}
//
float densityKernelGradient(float r, SPHParams params) {
    float scale = 45.0 / (3.1415926535 * params.kernelRadiusPow6);
    float d = params.kernelRadius - r;
    return scale * d * d;
}
//
float nearDensityKernelGradient(float r, SPHParams params) {
    float scale = 45.0 / (3.1415926535 * params.kernelRadiusPow5);
    float d = params.kernelRadius - r;
    return scale * d * d;
}

//粘性
float viscosityKernelLaplacian(float r, SPHParams params) {
    float scale = 45.0 / (3.1415926535 * params.kernelRadiusPow6);
    float d = params.kernelRadius - r;
    return scale * d;
}

kernel void computeForce(
                         device Particle *particles [[buffer(0)]],
                         device const Particle *sortedParticles [[buffer(1)]],
                         device const uint *prefixSum [[buffer(2)]],
                         constant Environment &env [[buffer(3)]],
                         constant SPHParams &params [[buffer(4)]],
                         uint tid  [[ thread_position_in_grid ]])
{
    if (tid < params.n) {
        float density_i = particles[tid].density;
        float nearDensity_i = particles[tid].nearDensity;
        float3 pos_i = particles[tid].position;
        float3 fPress = float3(0.0, 0.0, 0.0);
        float3 fVisc = float3(0.0, 0.0, 0.0);

        int3 v = cellPosition(pos_i, env);
        if (v.x < env.xGrids && 0 <= v.x &&
            v.y < env.yGrids && 0 <= v.y &&
            v.z < env.zGrids && 0 <= v.z)
        {
            if (v.x < env.xGrids && v.y < env.yGrids && v.z < env.zGrids) {
                for (int dz = max(-1, -v.z); dz <= min(1, env.zGrids - v.z - 1); dz++) {
                    for (int dy = max(-1, -v.y); dy <= min(1, env.yGrids - v.y - 1); dy++) {
                        int dxMin = max(-1, -v.x);
                        int dxMax = min(1, env.xGrids - v.x - 1);
                        int startCellNum = cellNumberFromId(v.x + dxMin, v.y + dy, v.z + dz, env);
                        int endCellNum = cellNumberFromId(v.x + dxMax, v.y + dy, v.z + dz, env);
                        uint start = prefixSum[startCellNum];
                        uint end = prefixSum[endCellNum + 1];
                        for (uint j = start; j < end; j++) {
                            float density_j = sortedParticles[j].density;
                            float nearDensity_j = sortedParticles[j].nearDensity;
                            float3 pos_j = sortedParticles[j].position;
                            float r2 = dot(pos_i - pos_j, pos_i - pos_j);
                            if (density_j == 0. || nearDensity_j == 0.) {
                                continue;
                            }
                            if (r2 < params.kernelRadiusPow2 && 1e-30 < r2) {
                                float r = sqrt(r2);
                                float pressure_i = params.stiffness * (density_i - params.restDensity);
                                float pressure_j = params.stiffness * (density_j - params.restDensity);
                                float nearPressure_i = params.nearStiffness * nearDensity_i;
                                float nearPressure_j = params.nearStiffness * nearDensity_j;
                                float sharedPressure = (pressure_i + pressure_j) / 2.0;
                                float nearSharedPressure = (nearPressure_i + nearPressure_j) / 2.0;
                                float3 dir = normalize(pos_j - pos_i);
                                fPress += -params.mass * sharedPressure * dir * densityKernelGradient(r, params) / density_j;
                                fPress += -params.mass * nearSharedPressure * dir * nearDensityKernelGradient(r, params) / nearDensity_j;
                                float3 relativeSpeed = sortedParticles[j].v - particles[tid].v;
                                fVisc += params.mass * relativeSpeed * viscosityKernelLaplacian(r, params) / density_j;
                            }
                        }
                    }
                }
            }
        }

        fVisc *= params.viscosity;
        float3 fGrv = density_i * float3(0.0, 9.8, 0.0);
        particles[tid].force = fPress + fVisc + fGrv;
    }
}
