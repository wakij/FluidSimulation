import Foundation
import Metal

class ParticleUtil {
    let copyPSO: MTLComputePipelineState
    
    let particleBuffer: MTLBuffer
    let particleNumBuffer: MTLBuffer
    let positionBuffer: MTLBuffer
    
    let threadsPerThreadGroup = 256
    let numThreadGroups: Int
    
    init(
        device: MTLDevice,
        library: MTLLibrary,
        particleBuffer: MTLBuffer,
        positionBuffer: MTLBuffer,
        particleNum: Int
    ) throws {
        guard let copyFunc = library.makeFunction(name: "copyPosition") else {
            fatalError("シェーダ関数の取得に失敗しました")
        }
        
        self.copyPSO = try device.makeComputePipelineState(function: copyFunc)
        self.particleBuffer = particleBuffer
        self.positionBuffer = positionBuffer
    
        var particleNum: UInt32 = UInt32(particleNum)
        particleNumBuffer = device.makeBuffer(bytes: &particleNum, length: MemoryLayout<UInt32>.stride, options: [])!
        
        self.numThreadGroups = (Int(particleNum) + threadsPerThreadGroup - 1) / threadsPerThreadGroup
    }
    
    func copy(commandBuffer: MTLCommandBuffer) {
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(copyPSO)
            computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(positionBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(particleNumBuffer, offset: 0, index: 2)

            let threadsPerThreadgroup = MTLSize(width: threadsPerThreadGroup, height: 1, depth: 1)
            let threadgroups = MTLSize(width: numThreadGroups, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
    }
}

