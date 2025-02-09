//
//  ViewController.swift
//  Fluid
//
//  Created by wakita tomoshige on 2025/02/08.
//

import Cocoa
import Metal
import MetalKit
import simd

struct Uniforms {
    var vMatrix: simd_float4x4
    var pMatrix: simd_float4x4;
    var invProjectionMatrix: simd_float4x4;
    var texelSize: simd_float2
    var size: Float
    var sphereRadius: Float //size/2
}

struct BiliteralUniforms {
    var maxFilterSize: Int32
    var blurDir: SIMD2<Float>
    var projectedParticleConstant: Float
    var depthThreshold: Float
}

struct FullScreenVertexIn {
    var position: SIMD2<Float>
    var texCoord: SIMD2<Float>
}

class ViewController: NSViewController {
    
    var metalView: MTKView!
    
    // Metal関連のプロパティ
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    
    var depthMapPipeLineState: MTLRenderPipelineState!
    var biliteralPipeLineState: MTLRenderPipelineState!
    var resultPipeLineState: MTLRenderPipelineState!
    
    var depthStentcilState: MTLDepthStencilState!
    
//    bufferを関連
    var uniformsBuffer: MTLBuffer!
    var positionBuffer: MTLBuffer!
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
    
    private var time: CGFloat = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Metalデバイスの取得
        device = MTLCreateSystemDefaultDevice()
        
        // MTKViewの設定
        metalView = MTKView(frame: CGRect(origin: .zero, size: .init(width: 512, height: 512)), device: device)
        metalView.layer?.isOpaque = false
        metalView.delegate = self
        metalView.clearDepth = 1
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        metalView.depthStencilPixelFormat = .invalid
        metalView.drawableSize = metalView.frame.size
        view.addSubview(metalView)
        
        // コマンドキューの作成
        commandQueue = device.makeCommandQueue()
        
        // シェーダーの設定
        setUpTextures()
        setUpPipelineStates()
        setUpPassDescriptors()
        
        setUpBuffer()
        
        self.view.wantsLayer = true
        self.view.layer?.backgroundColor = NSColor.lightGray.cgColor
    }
    
    private func setUpBuffer() {
        let camPostion: SIMD3<Float> = .init(x: 0, y: 0, z: -15)
        let camUpVec: SIMD3<Float> = .init(x: 0, y: 1, z: 0)
        let particleSize: Float = 0.6;
        
        let vMatrix = Matrix4.lookAt(eye: camPostion, center: .zero, up: camUpVec)
        let pMatrix = Matrix4.perspective(fovy: 45, aspect: Float(metalView.frame.width / metalView.frame.height), near: 5, far: 100)
        let invProjectionMatrix = pMatrix.inverse
        
        var uniforms = Uniforms(
            vMatrix: vMatrix,
            pMatrix: pMatrix,
            invProjectionMatrix: invProjectionMatrix,
            texelSize: .init(x: 1.0 / Float(metalView.frame.width), y: 1.0 / Float(metalView.frame.height)),
            size: particleSize,
            sphereRadius: particleSize/2.0)
        
        uniformsBuffer = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<Uniforms>.stride, options: [])!
        
        let billboardPositions: [SIMD3<Float>] = [
            .init(x: 0, y: 0, z: 0),
            .init(x: 0.2, y: 0, z: 0)
        ]
        
        let billboardIndices: [UInt16] = [
            0,1,2,
            2,1,3,
            
            4,5,6,
            6,5,7,
        ]
        
        positionBuffer = device.makeBuffer(bytes: billboardPositions, length: MemoryLayout<SIMD3<Float>>.stride * billboardPositions.count, options: [])!
        billboardIndexBuffer = device.makeBuffer(bytes: billboardIndices, length: MemoryLayout<UInt16>.stride * billboardIndices.count, options: [])!
        
        let fullScreenVertices: [FullScreenVertexIn] = [
            .init(position: .init(x: -1, y: 1), texCoord: .init(x: 0, y: 0)),
            .init(position: .init(x: 1, y: 1), texCoord: .init(x: 1, y: 0)),
            .init(position: .init(x: -1, y: -1), texCoord: .init(x: 0, y: 1)),
            .init(position: .init(x: 1, y: -1), texCoord: .init(x: 1, y: 1))
        ]
        
        fullScreenVertexBuffer = device.makeBuffer(bytes: fullScreenVertices, length: MemoryLayout<FullScreenVertexIn>.stride * fullScreenVertices.count, options: [])!
        
        let fullScreenIndices: [UInt16] = [
            0,1,2,
            2,1,3
        ]
        
        fullScreenIndexBuffer = device.makeBuffer(bytes: fullScreenIndices, length: MemoryLayout<UInt16>.stride * fullScreenIndices.count, options: [])!
        
        let blurMaxFilterSize: Int32 = 100
        let blurFilterSize: Int32 = 7
        let blurDepthScale: Float = 10
        let depthThreshold: Float = (particleSize / 2.0) * blurDepthScale;
        let projectedParticleConstant = (Float(blurFilterSize) * Float(particleSize) * 0.05 * (Float(metalView.frame.height) / 2.0)) /  tan(((45.0 * Float.pi) / 180.0) / 2.0)
        
        var horizontalBiliterlUniform: BiliteralUniforms = .init(
            maxFilterSize: blurMaxFilterSize,
            blurDir: .init(x: 1.0 / Float(metalView.frame.width), y: 0.0),
            projectedParticleConstant: projectedParticleConstant,
            depthThreshold: depthThreshold)
        horizontalBiliterlUniformBuffer = device.makeBuffer(bytes: &horizontalBiliterlUniform, length: MemoryLayout<BiliteralUniforms>.stride, options: [])!
        
        var verticalBiliterlUniform: BiliteralUniforms = .init(
            maxFilterSize: blurMaxFilterSize,
            blurDir: .init(x: 0.0, y: 1.0 / Float(metalView.frame.height)),
            projectedParticleConstant: projectedParticleConstant,
            depthThreshold: depthThreshold)
        verticalBiliterlUniformBuffer = device.makeBuffer(bytes: &verticalBiliterlUniform, length: MemoryLayout<BiliteralUniforms>.stride, options: [])!
        
    }
    
    private func setUpTextures() {
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
    }
    
    private func setUpPipelineStates() {
        guard let library = device.makeDefaultLibrary() else { return }
        
//        非線形深度を深度バッファに書き込む+カラーバッファに線形深度を書き込む
        let depthMapdescriptor = MTLRenderPipelineDescriptor()
        depthMapdescriptor.vertexFunction = library.makeFunction(name: "sphere_vertex")
        depthMapdescriptor.fragmentFunction = library.makeFunction(name: "sphere_fragment")
        depthMapdescriptor.depthAttachmentPixelFormat = .depth16Unorm
        depthMapdescriptor.colorAttachments[0].pixelFormat = .r32Float //線形深度をrに書き込む
        depthMapPipeLineState = try! device.makeRenderPipelineState(descriptor: depthMapdescriptor)
        
//        深度を書き込むためのなんか
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .less  // 例: 小さい値が手前と判断
        depthStencilDescriptor.isDepthWriteEnabled = true      // 深度値の書き込みを有効にする
        depthStentcilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)
        
        let biliteralDescriptor = MTLRenderPipelineDescriptor()
        biliteralDescriptor.vertexFunction = library.makeFunction(name: "biliteral_vertex")
        biliteralDescriptor.fragmentFunction = library.makeFunction(name: "biliteral_fragment")
        biliteralDescriptor.colorAttachments[0].pixelFormat = .r32Float //線形深度をrに書き込む
        biliteralPipeLineState = try! device.makeRenderPipelineState(descriptor: biliteralDescriptor)
        
        let fullscreenDescriptor = MTLRenderPipelineDescriptor()
        fullscreenDescriptor.vertexFunction = library.makeFunction(name: "fullscreen_vertex")
        fullscreenDescriptor.fragmentFunction = library.makeFunction(name: "fullscreen_fragment")
        fullscreenDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        resultPipeLineState = try! device.makeRenderPipelineState(descriptor: fullscreenDescriptor)
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
}

extension ViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // サイズ変更時の処理（必要に応じて実装）
        
    }

    func draw(in view: MTKView) {
        let commandBuffer = commandQueue.makeCommandBuffer()
        let depthMapRenderEncoder = commandBuffer?.makeRenderCommandEncoder(descriptor: depthPassDescriptor)
        depthMapRenderEncoder?.setRenderPipelineState(depthMapPipeLineState)
        depthMapRenderEncoder?.setVertexBuffer(uniformsBuffer, offset: 0, index: 0)
        depthMapRenderEncoder?.setVertexBuffer(positionBuffer, offset: 0, index: 1)
        depthMapRenderEncoder?.setFragmentBuffer(uniformsBuffer, offset: 0, index: 0)
//        深度書き込み用
        depthMapRenderEncoder?.setDepthStencilState(depthStentcilState)
        depthMapRenderEncoder?.drawIndexedPrimitives(type: .triangle, indexCount: 12, indexType: .uint16, indexBuffer: billboardIndexBuffer, indexBufferOffset: 0)
        depthMapRenderEncoder?.endEncoding()
        
//        blurをかけていく
        for _ in 0..<4 {
            let horizontalBlurRenderEncoder = commandBuffer?.makeRenderCommandEncoder(descriptor: horizontalBiliteralPassDescriptor)
            horizontalBlurRenderEncoder?.setRenderPipelineState(biliteralPipeLineState)
            horizontalBlurRenderEncoder?.setVertexBuffer(fullScreenVertexBuffer, offset: 0, index: 0)
            horizontalBlurRenderEncoder?.setFragmentTexture(depthMapTexture, index: 0)
            horizontalBlurRenderEncoder?.setFragmentBuffer(horizontalBiliterlUniformBuffer, offset: 0, index: 0)
            horizontalBlurRenderEncoder?.drawIndexedPrimitives(type: .triangle, indexCount: 6, indexType: .uint16, indexBuffer: fullScreenIndexBuffer, indexBufferOffset: 0)
            horizontalBlurRenderEncoder?.endEncoding()
            
            let verticalBlurRenderEncoder = commandBuffer?.makeRenderCommandEncoder(descriptor: verticalBiliteralPassDescriptor)
            verticalBlurRenderEncoder?.setRenderPipelineState(biliteralPipeLineState)
            verticalBlurRenderEncoder?.setVertexBuffer(fullScreenVertexBuffer, offset: 0, index: 0)
            verticalBlurRenderEncoder?.setFragmentTexture(biliteralTexture, index: 0)
            verticalBlurRenderEncoder?.setFragmentBuffer(verticalBiliterlUniformBuffer, offset: 0, index: 0)
            verticalBlurRenderEncoder?.drawIndexedPrimitives(type: .triangle, indexCount: 6, indexType: .uint16, indexBuffer: fullScreenIndexBuffer, indexBufferOffset: 0)
            verticalBlurRenderEncoder?.endEncoding()
        }
        
//        深度マップの作成
        // レンダリングの開始
        guard let drawable = metalView.currentDrawable, let renderPassDescriptor = metalView.currentRenderPassDescriptor else { return }
        
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture
        let renderEncoder = commandBuffer?.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        renderEncoder?.setRenderPipelineState(resultPipeLineState)
        renderEncoder?.setVertexBuffer(fullScreenVertexBuffer, offset: 0, index: 0)
        renderEncoder?.setFragmentTexture(biliteralTexture, index: 0)
        renderEncoder?.setFragmentBuffer(uniformsBuffer, offset: 0, index: 0)
        renderEncoder?.drawIndexedPrimitives(type: .triangle, indexCount: 6, indexType: .uint16, indexBuffer: fullScreenIndexBuffer, indexBufferOffset: 0)
        renderEncoder?.endEncoding()
        
        commandBuffer?.present(drawable)
        commandBuffer?.commit()
    }
}

