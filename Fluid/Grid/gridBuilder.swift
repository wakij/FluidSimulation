import Foundation
import Metal

class GridBuilder {
    let gridClearPSO: MTLComputePipelineState
    let gridCountPSO: MTLComputePipelineState
    
    let particleBuffer: MTLBuffer
    let gridNumBuffer: MTLBuffer
    let cellParticleCountBuffer: MTLBuffer
    let particleNumBuffer: MTLBuffer
    let particleCellOffsetBuffer: MTLBuffer
    let environmentBuffer: MTLBuffer
    
    let threadsPerThreadGroup = 256
    let cleanNumThreadGroups: Int
    let buildNumThreadGroups: Int
    
    init(
        device: MTLDevice,
        library: MTLLibrary,
        particleBuffer: MTLBuffer,
        cellParticleCountBuffer: MTLBuffer,
        particleCellOffsetBuffer: MTLBuffer,
        environmentBuffer: MTLBuffer,
        particleNum: Int,
        gridNum: Int
    ) throws {
        guard let gridClearFunc = library.makeFunction(name: "gridClear"),
              let gridCountFunc = library.makeFunction(name: "gridCount") else {
            fatalError("シェーダ関数の取得に失敗しました")
        }
        
        self.gridClearPSO = try device.makeComputePipelineState(function: gridClearFunc)
        self.gridCountPSO = try device.makeComputePipelineState(function: gridCountFunc)
        
        self.particleBuffer = particleBuffer
        var gridNum: UInt32 = UInt32(gridNum)
        self.gridNumBuffer = device.makeBuffer(bytes: &gridNum, length: MemoryLayout<UInt32>.stride, options: [])!
        self.cellParticleCountBuffer = cellParticleCountBuffer
        var particleNum: UInt32 = UInt32(particleNum)
        particleNumBuffer = device.makeBuffer(bytes: &particleNum, length: MemoryLayout<UInt32>.stride, options: [])!
        self.environmentBuffer = environmentBuffer
        self.particleCellOffsetBuffer = particleCellOffsetBuffer
        
        self.cleanNumThreadGroups = (Int(gridNum) + threadsPerThreadGroup - 1) / threadsPerThreadGroup
        self.buildNumThreadGroups = (Int(particleNum) + threadsPerThreadGroup - 1) / threadsPerThreadGroup
    }
    
//    累積部分を計算しているbufferを0クリアする
    func clean(commandBuffer: MTLCommandBuffer) {
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(gridClearPSO)
            computeEncoder.setBuffer(cellParticleCountBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(gridNumBuffer, offset: 0, index: 1)

            let threadsPerThreadgroup = MTLSize(width: threadsPerThreadGroup, height: 1, depth: 1)
            let threadgroups = MTLSize(width: cleanNumThreadGroups, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
    }
    
    func build(commandBuffer: MTLCommandBuffer) {
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(gridCountPSO)
            computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(cellParticleCountBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(particleNumBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(particleCellOffsetBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(environmentBuffer, offset: 0, index: 4)

            let threadsPerThreadgroup = MTLSize(width: threadsPerThreadGroup, height: 1, depth: 1)
            let threadgroups = MTLSize(width: buildNumThreadGroups, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
    }
}

