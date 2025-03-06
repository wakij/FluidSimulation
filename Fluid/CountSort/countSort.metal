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

int cellId(float3 position, Environment env) {
//    positionが0<=position<=env.gridになるように調整している
    int xi = int(floor((position.x + env.xHalf + env.offset) / env.cellSize));
    int yi = int(floor((position.y + env.yHalf + env.offset) / env.cellSize));
    int zi = int(floor((position.z + env.zHalf + env.offset) / env.cellSize));

    return xi + yi * env.xGrids + zi * env.xGrids * env.yGrids;
}

kernel void gridClear(
                      device uint *cellParticleCount      [[ buffer(0) ]],
                      constant uint &n [[ buffer(1)]],
                      uint tid                       [[ thread_position_in_grid ]])
{
    if (tid < n) {
        cellParticleCount[tid] = 0;
    }
}


kernel void gridCount(
                      device Particle *particles [[buffer(0)]],
                      device atomic_uint *cellParticleCount [[buffer(1)]],
                      constant uint &numParticles         [[buffer(2)]],
                      device uint *particleCellOffset [[buffer(3)]],
                      constant Environment &env [[buffer(4)]],
                      uint gid  [[ thread_position_in_grid ]]
){
    if (gid < numParticles) {
        uint cellIndex = cellId(particles[gid].position, env);
        if (cellIndex < uint(env.xGrids * env.yGrids * env.zGrids)) {
            particleCellOffset[gid] = atomic_fetch_add_explicit(&cellParticleCount[cellIndex], 1, memory_order_relaxed);
        }
    }
}

kernel void countSort(
                       device const Particle *sourceParticles [[buffer(0)]],
                       device Particle *targetParticles [[buffer(1)]],
                       device uint *cellParticleCount [[buffer(2)]],
                       device uint *particleCellOffset [[buffer(3)]],
                       constant Environment &env,
                       constant SPHParams &params,
                       uint tid [[ thread_position_in_grid ]]
)
{
    if (tid < params.n) {
        uint cellIndex = cellId(sourceParticles[tid].position, env);
        if (cellIndex < uint(env.xGrids * env.yGrids * env.zGrids)) {
            // cellParticleCount[pCellId + 1]:pCellIdの終端
            // 逆順に配置している。
            uint targetIndex = cellParticleCount[cellIndex + 1] - particleCellOffset[tid] - 1;
            if (targetIndex < params.n) {
                targetParticles[targetIndex] = sourceParticles[tid];
            }
        }
    }
}
