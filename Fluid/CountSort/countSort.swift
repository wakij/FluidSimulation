import Foundation
import Metal

class CountSort {
   let sortPSO: MTLComputePipelineState
   
   let sourceParticlesBuffer: MTLBuffer
   let targetParticlesBuffer: MTLBuffer
   let cellParticleCountPrefixSumBuffer: MTLBuffer //排他的累積部分和が入っていることを期待
   let particleCellOffsetBuffer: MTLBuffer //ローカルオフセット
   let environmentBuffer: MTLBuffer
   let sphParamsBuffer: MTLBuffer
   
   let threadsPerThreadGroup = 256
   let numThreadGroups: Int
   
   init(
       device: MTLDevice,
       library: MTLLibrary,
       sourceParticlesBuffer: MTLBuffer,
       targetParticlesBuffer: MTLBuffer,
       cellParticleCountPrefixSumBuffer: MTLBuffer,
       particleCellOffsetBuffer: MTLBuffer,
       environmentBuffer: MTLBuffer,
       sphParamsBuffer: MTLBuffer,
       particleNum: UInt32
   ) throws {
       guard let sortFunction = library.makeFunction(name: "countSort") else {
           fatalError("シェーダ関数の取得に失敗しました")
       }
       
       self.sortPSO = try device.makeComputePipelineState(function: sortFunction)
       self.sourceParticlesBuffer = sourceParticlesBuffer
       self.targetParticlesBuffer = targetParticlesBuffer
       self.cellParticleCountPrefixSumBuffer = cellParticleCountPrefixSumBuffer
       self.particleCellOffsetBuffer = particleCellOffsetBuffer
       self.environmentBuffer = environmentBuffer
       self.sphParamsBuffer = sphParamsBuffer
       
       self.numThreadGroups = (Int(particleNum) + threadsPerThreadGroup - 1) / threadsPerThreadGroup
   }
   
   func excute(commandBuffer: MTLCommandBuffer) {
       guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
       computeEncoder.setComputePipelineState(sortPSO)
       computeEncoder.setBuffer(sourceParticlesBuffer, offset: 0, index: 0)
       computeEncoder.setBuffer(targetParticlesBuffer, offset: 0, index: 1)
       computeEncoder.setBuffer(cellParticleCountPrefixSumBuffer, offset: 0, index: 2)
       computeEncoder.setBuffer(particleCellOffsetBuffer, offset: 0, index: 3)
       computeEncoder.setBuffer(environmentBuffer, offset: 0, index: 4)
       computeEncoder.setBuffer(sphParamsBuffer, offset: 0, index: 5)

   //    スレッド数が必ず2の累乗の形になるようになっている
       let threadsPerThreadgroup = MTLSize(width: Int(threadsPerThreadGroup), height: 1, depth: 1)
       let threadgroups = MTLSize(width: numThreadGroups, height: 1, depth: 1)
       computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
       computeEncoder.endEncoding()
   }
}

