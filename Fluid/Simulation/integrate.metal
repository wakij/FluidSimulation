//
//  integrate.metal
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

struct RealBoxSize {
    float xHalf;
    float yHalf;
    float zHalf;
};

kernel void integrate(
                      device Particle   *particles   [[ buffer(0) ]],
                      constant RealBoxSize &realBoxSize [[ buffer(1) ]],
                      constant SPHParams &params [[ buffer(2) ]],
                      uint tid [[ thread_position_in_grid ]]
) {
    if (tid < params.n) {
      // avoid zero division
        if (particles[tid].density != 0.) {
            float3 a = particles[tid].force / particles[tid].density;

            float xPlusDist = realBoxSize.xHalf - particles[tid].position.x;
            float xMinusDist = realBoxSize.xHalf + particles[tid].position.x;
            float yPlusDist = realBoxSize.yHalf - particles[tid].position.y;
            float yMinusDist = realBoxSize.yHalf + particles[tid].position.y;
            float zPlusDist = realBoxSize.zHalf - particles[tid].position.z;
            float zMinusDist = realBoxSize.zHalf + particles[tid].position.z;

            float wallStiffness = 8000.;

            float3 xPlusForce = float3(1., 0., 0.) * wallStiffness * min(xPlusDist, 0.);
            float3 xMinusForce = float3(-1., 0., 0.) * wallStiffness * min(xMinusDist, 0.);
            float3 yPlusForce = float3(0., 1., 0.) * wallStiffness * min(yPlusDist, 0.);
            float3 yMinusForce = float3(0., -1., 0.) * wallStiffness * min(yMinusDist, 0.);
            float3 zPlusForce = float3(0., 0., 1.) * wallStiffness * min(zPlusDist, 0.);
            float3 zMinusForce = float3(0., 0., -1.) * wallStiffness * min(zMinusDist, 0.);

            float3 xForce = xPlusForce + xMinusForce;
            float3 yForce = yPlusForce + yMinusForce;
            float3 zForce = zPlusForce + zMinusForce;

            a += xForce + yForce + zForce;
            
            particles[tid].v += params.dt * a;
            particles[tid].position += params.dt * particles[tid].v;
            particles[tid].lastAcceration = a;
      }
    }
}
