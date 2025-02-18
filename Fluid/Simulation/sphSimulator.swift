import Metal
    
class SPHSimluator {
    let densityPSO: MTLComputePipelineState
    let forcePSO: MTLComputePipelineState
    let integratePSO: MTLComputePipelineState
    
    let particlesBuffer: MTLBuffer
    let sortedParticlesBuffer: MTLBuffer
    let prefixSumBuffer: MTLBuffer
    let environmentBuffer: MTLBuffer
    let sphParamsBuffer: MTLBuffer
    let realBoxSizeBuffer: MTLBuffer
    
    let threadsPerThreadGroup = 256
    let numThreadGroups: Int
    
    init(
        device: MTLDevice,
        library: MTLLibrary,
        particleBuffer: MTLBuffer,
        sortedParticleBuffer: MTLBuffer,
        prefixSumBuffer: MTLBuffer,
        environmentBuffer: MTLBuffer,
        sphParamsBuffer: MTLBuffer,
        realBoxSizeBuffer: MTLBuffer,
        particleNum: UInt32
    ) throws {
        guard let densityFunction = library.makeFunction(name: "computeDensity"),
        let forceFunction = library.makeFunction(name: "computeForce"),
        let integrateFunction = library.makeFunction(name: "integrate") else {
            fatalError("シェーダ関数の取得に失敗しました")
        }
        self.densityPSO = try device.makeComputePipelineState(function: densityFunction)
        self.forcePSO = try device.makeComputePipelineState(function: forceFunction)
        self.integratePSO = try device.makeComputePipelineState(function: integrateFunction)
        
        self.particlesBuffer = particleBuffer
        self.sortedParticlesBuffer = sortedParticleBuffer
        self.prefixSumBuffer = prefixSumBuffer
        self.environmentBuffer = environmentBuffer
        self.sphParamsBuffer = sphParamsBuffer
        self.realBoxSizeBuffer = realBoxSizeBuffer
        
        self.numThreadGroups = (Int(particleNum) + threadsPerThreadGroup - 1) / threadsPerThreadGroup
    }
    
    func setInitDensity(commandBuffer: MTLCommandBuffer) {
        let threadsPerThreadgroup = MTLSize(width: Int(threadsPerThreadGroup), height: 1, depth: 1)
        let threadgroups = MTLSize(width: numThreadGroups, height: 1, depth: 1)
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(densityPSO)
        computeEncoder.setBuffer(particlesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedParticlesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(prefixSumBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(environmentBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(sphParamsBuffer, offset: 0, index: 4)
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
//        力を計算
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(forcePSO)
        computeEncoder.setBuffer(particlesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedParticlesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(prefixSumBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(environmentBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(sphParamsBuffer, offset: 0, index: 4)
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    func updateDensity(commandBuffer: MTLCommandBuffer) {
        let threadsPerThreadgroup = MTLSize(width: Int(threadsPerThreadGroup), height: 1, depth: 1)
        let threadgroups = MTLSize(width: numThreadGroups, height: 1, depth: 1)
        
//        密度を計算
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(densityPSO)
        computeEncoder.setBuffer(particlesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedParticlesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(prefixSumBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(environmentBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(sphParamsBuffer, offset: 0, index: 4)
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    func updateForce(commandBuffer: MTLCommandBuffer) {
        let threadsPerThreadgroup = MTLSize(width: Int(threadsPerThreadGroup), height: 1, depth: 1)
        let threadgroups = MTLSize(width: numThreadGroups, height: 1, depth: 1)
        
//        力を計算
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(forcePSO)
        computeEncoder.setBuffer(particlesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedParticlesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(prefixSumBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(environmentBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(sphParamsBuffer, offset: 0, index: 4)
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    func updatePosition(commandBuffer: MTLCommandBuffer) {
        let threadsPerThreadgroup = MTLSize(width: Int(threadsPerThreadGroup), height: 1, depth: 1)
        let threadgroups = MTLSize(width: numThreadGroups, height: 1, depth: 1)
        
//        座標を更新
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(integratePSO)
        computeEncoder.setBuffer(particlesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(realBoxSizeBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(sphParamsBuffer, offset: 0, index: 2)
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
}
