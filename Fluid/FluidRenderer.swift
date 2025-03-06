
import Cocoa
import Metal
import MetalKit
import simd

struct FluidRenderParams {
    var blurMaxFilterSize: Int32 = 100
    var blurFilterSize: Int32 = 12
    var blurDepthScale: Float = 10
    let depthThreshold: Float
    let projectedParticleConstant: Float
    var particleNum: UInt32
    
    init(
        particleSize: Float,
        height: Float,
        particleNum: UInt32
    )
    {
        self.depthThreshold = (particleSize / 2.0) * blurDepthScale
        self.projectedParticleConstant = (Float(blurFilterSize) * particleSize * 0.05 * (height / 2.0)) /  tan(((45.0 * Float.pi) / 180.0) / 2.0)
        self.particleNum = particleNum
    }
}

class FluidRenderer {
//    debug用
    var depthMapPipeLineState: MTLRenderPipelineState!
    var biliteralPipeLineState: MTLRenderPipelineState!
    var resultPipeLineState: MTLRenderPipelineState!
    var depthStentcilState: MTLDepthStencilState!
    
//    bufferを関連
    var uniformsBuffer: MTLBuffer
    var positionBuffer: MTLBuffer
    var billboardIndexBuffer: MTLBuffer!
    
    var fullScreenVertexBuffer: MTLBuffer!
    var fullScreenIndexBuffer: MTLBuffer!
    
    var horizontalBiliterlUniformBuffer: MTLBuffer!
    var verticalBiliterlUniformBuffer: MTLBuffer!
    
//    descriptor
    var depthPassDescriptor: MTLRenderPassDescriptor!
    var horizontalBiliteralPassDescriptor: MTLRenderPassDescriptor!
    var verticalBiliteralPassDescriptor: MTLRenderPassDescriptor!
    
//    texture関連
    var depthMapTexture: MTLTexture!
    var depthTestTexture: MTLTexture!
//    中間テクスチャ
    var biliteralTexture: MTLTexture!
    var bgTexture: MTLTexture!
    
    var params: FluidRenderParams!
    
    var billboardIndices: [UInt16]
    let fullScreenIndices: [UInt16] = [
        0,1,2,
        2,1,3
    ]
    
    init(
        device: MTLDevice,
        library: MTLLibrary,
        metalView: MTKView,
        params: FluidRenderParams,
        uniformBuffer: MTLBuffer,
        positionBuffer: MTLBuffer
    ) throws {
        self.params = params
        
        self.uniformsBuffer = uniformBuffer
        self.positionBuffer = positionBuffer
        
        billboardIndices = []
        for i in 0..<params.particleNum {
            let index = UInt16(i * 4)
            billboardIndices.append(contentsOf: [
                index, index+1, index+2,
                index+2, index+1, index+3
            ])
        }
        
        setUpBuffer(device: device, metalView: metalView)
        try setUpTextures(device: device, metalView: metalView)
        setUpPipelineStates(device: device, library: library, metalView: metalView)
        setUpPassDescriptors()
    }
    
    private func setUpTextures(device: MTLDevice, metalView: MTKView) throws {
//        非線形深度を書き込んで深度テストに用いる
        let depthTestTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth16Unorm,
            width: Int(metalView.drawableSize.width),
            height: Int(metalView.drawableSize.height),
            mipmapped: false)
        depthTestTextureDescriptor.usage = [.renderTarget]
        depthTestTexture = device.makeTexture(descriptor: depthTestTextureDescriptor)
        
//        線形深度を書き込む
        let depthMapTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: Int(metalView.drawableSize.width),
            height: Int(metalView.drawableSize.height),
            mipmapped: false)
        depthMapTextureDescriptor.usage = [.renderTarget, .shaderRead]
        depthMapTexture = device.makeTexture(descriptor: depthMapTextureDescriptor)
        
        let biliteralTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: Int(metalView.drawableSize.width),
            height: Int(metalView.drawableSize.height),
            mipmapped: false)
        biliteralTextureDescriptor.usage = [.renderTarget, .shaderRead]
        biliteralTexture = device.makeTexture(descriptor: biliteralTextureDescriptor)
        
        let textureLoader = MTKTextureLoader(device: device)
        guard let url = Bundle.main.url(forResource: "bgImage", withExtension: "png") else { return }
        bgTexture = try textureLoader.newTexture(URL: url, options: nil)
    }
    
    private func setUpPipelineStates(device: MTLDevice, library: MTLLibrary, metalView: MTKView) {
//        非線形深度を深度バッファに書き込む+カラーバッファに線形深度を書き込む
        let depthMapdescriptor = MTLRenderPipelineDescriptor()
        depthMapdescriptor.vertexFunction = library.makeFunction(name: "depthMap_vertex")
        depthMapdescriptor.fragmentFunction = library.makeFunction(name: "depthMap_fragment")
        depthMapdescriptor.depthAttachmentPixelFormat = .depth16Unorm
        depthMapdescriptor.colorAttachments[0].pixelFormat = .r32Float //線形深度をrに書き込む
        self.depthMapPipeLineState = try! device.makeRenderPipelineState(descriptor: depthMapdescriptor)
        
//        深度を書き込むためのなんか
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .less  // 例: 小さい値が手前と判断
        depthStencilDescriptor.isDepthWriteEnabled = true      // 深度値の書き込みを有効にする
        self.depthStentcilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)!
        
        let biliteralDescriptor = MTLRenderPipelineDescriptor()
        biliteralDescriptor.vertexFunction = library.makeFunction(name: "biliteral_vertex")
        biliteralDescriptor.fragmentFunction = library.makeFunction(name: "biliteral_fragment")
        biliteralDescriptor.colorAttachments[0].pixelFormat = .r32Float //線形深度をrに書き込む
        self.biliteralPipeLineState = try! device.makeRenderPipelineState(descriptor: biliteralDescriptor)
        
        let fullscreenDescriptor = MTLRenderPipelineDescriptor()
        fullscreenDescriptor.vertexFunction = library.makeFunction(name: "fullscreen_vertex")
        fullscreenDescriptor.fragmentFunction = library.makeFunction(name: "fullscreen_fragment")
        fullscreenDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        self.resultPipeLineState = try! device.makeRenderPipelineState(descriptor: fullscreenDescriptor)
    }
    
    private func setUpPipelineStates2(device: MTLDevice, library: MTLLibrary, metalView: MTKView) {
//        非線形深度を深度バッファに書き込む+カラーバッファに線形深度を書き込む
        let spheredescriptor = MTLRenderPipelineDescriptor()
        spheredescriptor.vertexFunction = library.makeFunction(name: "sphere_vertex")
        spheredescriptor.fragmentFunction = library.makeFunction(name: "sphere_fragment")
        spheredescriptor.depthAttachmentPixelFormat = .depth16Unorm
        spheredescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        self.depthMapPipeLineState = try! device.makeRenderPipelineState(descriptor: spheredescriptor)
        
//        深度を書き込むためのなんか
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .less  // 例: 小さい値が手前と判断
        depthStencilDescriptor.isDepthWriteEnabled = true      // 深度値の書き込みを有効にする
        self.depthStentcilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)!
    }
    
    private func setUpPassDescriptors() {
        depthPassDescriptor = MTLRenderPassDescriptor()
        depthPassDescriptor.colorAttachments[0].texture = depthMapTexture
        depthPassDescriptor.depthAttachment.texture = depthTestTexture
        depthPassDescriptor.depthAttachment.clearDepth = 1.0
        depthPassDescriptor.depthAttachment.loadAction = .clear
        depthPassDescriptor.depthAttachment.storeAction = .store
        depthPassDescriptor.colorAttachments[0].loadAction = .clear
        depthPassDescriptor.colorAttachments[0].storeAction = .store
        depthPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
        
        horizontalBiliteralPassDescriptor = MTLRenderPassDescriptor()
        horizontalBiliteralPassDescriptor.colorAttachments[0].texture = biliteralTexture
        horizontalBiliteralPassDescriptor.colorAttachments[0].loadAction = .clear
        horizontalBiliteralPassDescriptor.colorAttachments[0].storeAction = .store
        horizontalBiliteralPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
        
        verticalBiliteralPassDescriptor = MTLRenderPassDescriptor()
        verticalBiliteralPassDescriptor.colorAttachments[0].texture = depthMapTexture
        verticalBiliteralPassDescriptor.colorAttachments[0].loadAction = .clear
        verticalBiliteralPassDescriptor.colorAttachments[0].storeAction = .store
        verticalBiliteralPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
    }
    
    private func setUpBuffer(device: MTLDevice, metalView: MTKView) {
        billboardIndexBuffer = device.makeBuffer(bytes: billboardIndices, length: MemoryLayout<UInt16>.stride * billboardIndices.count, options: [])!
        
        let fullScreenVertices: [FullScreenVertexIn] = [
            .init(position: .init(x: -1, y: 1), texCoord: .init(x: 0, y: 0)),
            .init(position: .init(x: 1, y: 1), texCoord: .init(x: 1, y: 0)),
            .init(position: .init(x: -1, y: -1), texCoord: .init(x: 0, y: 1)),
            .init(position: .init(x: 1, y: -1), texCoord: .init(x: 1, y: 1))
        ]
        
        fullScreenVertexBuffer = device.makeBuffer(bytes: fullScreenVertices, length: MemoryLayout<FullScreenVertexIn>.stride * fullScreenVertices.count, options: [])!
        
        fullScreenIndexBuffer = device.makeBuffer(bytes: fullScreenIndices, length: MemoryLayout<UInt16>.stride * fullScreenIndices.count, options: [])!
        
        var horizontalBiliterlUniform: BiliteralUniforms = .init(
            maxFilterSize: params.blurMaxFilterSize,
            blurDir: .init(x: 1.0 / Float(metalView.frame.width), y: 0.0),
            projectedParticleConstant: params.projectedParticleConstant,
            depthThreshold: params.depthThreshold)
        horizontalBiliterlUniformBuffer = device.makeBuffer(bytes: &horizontalBiliterlUniform, length: MemoryLayout<BiliteralUniforms>.stride, options: [])!
        
        var verticalBiliterlUniform: BiliteralUniforms = .init(
            maxFilterSize: params.blurMaxFilterSize,
            blurDir: .init(x: 0.0, y: 1.0 / Float(metalView.frame.height)),
            projectedParticleConstant: params.projectedParticleConstant,
            depthThreshold: params.depthThreshold)
        verticalBiliterlUniformBuffer = device.makeBuffer(bytes: &verticalBiliterlUniform, length: MemoryLayout<BiliteralUniforms>.stride, options: [])!
        
    }
    
    func render(commandBuffer: MTLCommandBuffer, metalView: MTKView) {
        guard let depthMapRenderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: depthPassDescriptor) else { return }
        
        depthMapRenderEncoder.setRenderPipelineState(depthMapPipeLineState)
        depthMapRenderEncoder.setVertexBuffer(uniformsBuffer, offset: 0, index: 0)
        depthMapRenderEncoder.setVertexBuffer(positionBuffer, offset: 0, index: 1)
        depthMapRenderEncoder.setFragmentBuffer(uniformsBuffer, offset: 0, index: 0)
//        深度書き込み用
        depthMapRenderEncoder.setDepthStencilState(depthStentcilState)
        depthMapRenderEncoder.drawIndexedPrimitives(type: .triangle, indexCount: billboardIndices.count, indexType: .uint16, indexBuffer: billboardIndexBuffer, indexBufferOffset: 0)
        depthMapRenderEncoder.endEncoding()
        
//        blurをかけていく
        for _ in 0..<4 {
            guard let horizontalBlurRenderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: horizontalBiliteralPassDescriptor) else { return }
            horizontalBlurRenderEncoder.setRenderPipelineState(biliteralPipeLineState)
            horizontalBlurRenderEncoder.setVertexBuffer(fullScreenVertexBuffer, offset: 0, index: 0)
            horizontalBlurRenderEncoder.setFragmentTexture(depthMapTexture, index: 0)
            horizontalBlurRenderEncoder.setFragmentBuffer(horizontalBiliterlUniformBuffer, offset: 0, index: 0)
            horizontalBlurRenderEncoder.drawIndexedPrimitives(type: .triangle, indexCount: 6, indexType: .uint16, indexBuffer: fullScreenIndexBuffer, indexBufferOffset: 0)
            horizontalBlurRenderEncoder.endEncoding()
            
            guard let verticalBlurRenderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: verticalBiliteralPassDescriptor) else { return }
            
            verticalBlurRenderEncoder.setRenderPipelineState(biliteralPipeLineState)
            verticalBlurRenderEncoder.setVertexBuffer(fullScreenVertexBuffer, offset: 0, index: 0)
            verticalBlurRenderEncoder.setFragmentTexture(biliteralTexture, index: 0)
            verticalBlurRenderEncoder.setFragmentBuffer(verticalBiliterlUniformBuffer, offset: 0, index: 0)
            verticalBlurRenderEncoder.drawIndexedPrimitives(type: .triangle, indexCount: 6, indexType: .uint16, indexBuffer: fullScreenIndexBuffer, indexBufferOffset: 0)
            verticalBlurRenderEncoder.endEncoding()
        }
        
//        深度マップの作成
//         レンダリングの開始
        guard let drawable = metalView.currentDrawable, let renderPassDescriptor = metalView.currentRenderPassDescriptor else { return }
        
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        renderEncoder?.setRenderPipelineState(resultPipeLineState)
        renderEncoder?.setVertexBuffer(fullScreenVertexBuffer, offset: 0, index: 0)
        renderEncoder?.setFragmentTexture(depthMapTexture, index: 0)
        renderEncoder?.setFragmentTexture(bgTexture, index: 1)
        renderEncoder?.setFragmentBuffer(uniformsBuffer, offset: 0, index: 0)
        renderEncoder?.drawIndexedPrimitives(type: .triangle, indexCount: 6, indexType: .uint16, indexBuffer: fullScreenIndexBuffer, indexBufferOffset: 0)
        renderEncoder?.endEncoding()
    }
}
