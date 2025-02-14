import Metal

class PrefixSum {
    let THREADGROUP_SIZE: UInt32 = 256
    let elementsPerGroup :Int
    let numGroups: Int
    var n: UInt32
    
    let blockScanPSO: MTLComputePipelineState
    let scanBlockSumsPSO: MTLComputePipelineState
    let addBlockOffsetsPSO: MTLComputePipelineState
    
    let inputBuffer: MTLBuffer
    let outputBuffer: MTLBuffer
    let blockSumBuffer: MTLBuffer
    let numElementsBuffer: MTLBuffer
    let numBlocksBuffer: MTLBuffer
    let nBuffer: MTLBuffer
    
    
    init(device: MTLDevice, library: MTLLibrary, gridNum: Int, inputBuffer: MTLBuffer, outputBuffer: MTLBuffer) throws {
        guard let blockScanFunction = library.makeFunction(name: "blockScanKernel"),
              let scanBlockSumsFunction = library.makeFunction(name: "scanBlockSumsKernel"),
              let addBlockOffsetsFunction = library.makeFunction(name: "addBlockOffsetsKernel") else {
            fatalError("シェーダ関数の取得に失敗しました")
        }
        
        self.blockScanPSO = try device.makeComputePipelineState(function: blockScanFunction)
        self.scanBlockSumsPSO = try device.makeComputePipelineState(function: scanBlockSumsFunction)
        self.addBlockOffsetsPSO = try device.makeComputePipelineState(function: addBlockOffsetsFunction)
        
        self.elementsPerGroup = Int(THREADGROUP_SIZE * 2)
        self.numGroups = (gridNum + elementsPerGroup - 1) / elementsPerGroup
        
        self.n = 1
        while n < numGroups {
            n *= 2
        }
        
        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
        self.blockSumBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * numGroups,
                                                                      options: [])!
        var gridNum = UInt32(gridNum)
        self.numElementsBuffer = device.makeBuffer(bytes: &gridNum,
                                                   length: MemoryLayout<UInt32>.stride,
                                                  options: [])!
        var numBlocksUInt = UInt32(numGroups)
        self.numBlocksBuffer = device.makeBuffer(bytes: &numBlocksUInt, length: MemoryLayout<UInt32>.stride, options: [])!
        self.nBuffer = device.makeBuffer(bytes: &n, length: MemoryLayout<UInt32>.stride, options: [])!
    }
    
    func execute(commandBuffer: MTLCommandBuffer) {
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(blockScanPSO)
            computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(blockSumBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(numElementsBuffer, offset: 0, index: 3)

        //    スレッド数が必ず2の累乗の形になるようになっている
            let threadsPerThreadgroup = MTLSize(width: Int(THREADGROUP_SIZE), height: 1, depth: 1)
            let threadgroups = MTLSize(width: numGroups, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }

        // ── 2. scanBlockSumsKernel のディスパッチ ──
//        一スレッドグループで収まる計算
        if numGroups > 0, let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(scanBlockSumsPSO)
            computeEncoder.setBuffer(blockSumBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(numBlocksBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(nBuffer, offset: 0, index: 2)

            let threadsPerThreadgroup = MTLSize(width: Int(n), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }

        // ── 3. addBlockOffsetsKernel のディスパッチ ──
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(addBlockOffsetsPSO)
            computeEncoder.setBuffer(outputBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(blockSumBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(numElementsBuffer, offset: 0, index: 2)

            let threadsPerThreadgroup = MTLSize(width: Int(THREADGROUP_SIZE), height: 1, depth: 1)
            let threadgroups = MTLSize(width: numGroups, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
    }
}

