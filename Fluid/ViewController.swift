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

protocol FluidViewDelegate: NSObject {
    func onKeyDown(with event: NSEvent)
}

class FluidView: MTKView {
    override var acceptsFirstResponder: Bool { return true }
    weak var keyDownDelegate: FluidViewDelegate?
    
    override func keyDown(with event: NSEvent) {
        self.keyDownDelegate?.onKeyDown(with: event)
    }
}

class ViewController: NSViewController {
    
    var metalView: FluidView!
    
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
    
//    設定値
    var camPostion: SIMD3<Float> = .init(x: 0, y: 0, z: -15)
    var camUp: SIMD3<Float> = .init(x: 0, y: 1, z: 0)
    var cameraTarget: SIMD3<Float> = .zero
    var currentOrientation: simd_quatf = Quaternion.identity()
    var lastMouseLocation: CGPoint?
    let sensitivity: Float = 0.005
    
    let particleSize: Float = 0.6;
    
    let fullScreenIndices: [UInt16] = [
        0,1,2,
        2,1,3
    ]
    
    var billboardPositions: [SIMD3<Float>] = []
    var billboardIndices: [UInt16] = []
    
    func generateParticles(count: Int) {
        for i in 0..<count {
            let x = Float.random(in: -2...2)
            let y = Float.random(in: -2...2)
            let z = Float.random(in: -2...2)
            
            billboardPositions.append(.init(x: x, y: y, z: z))
            
            let index: UInt16 = UInt16(i) * 4
            
            billboardIndices.append(contentsOf: [
                index, index+1, index+2,
                index+2, index+1, index+3
            ])
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Metalデバイスの取得
        device = MTLCreateSystemDefaultDevice()
        
        // MTKViewの設定
        metalView = FluidView(frame: CGRect(origin: .zero, size: .init(width: 512, height: 512)), device: device)
        metalView.layer?.isOpaque = false
        metalView.delegate = self
        metalView.clearDepth = 1
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        metalView.depthStencilPixelFormat = .invalid
        metalView.drawableSize = metalView.frame.size
        metalView.keyDownDelegate = self
        view.addSubview(metalView)
        
        // コマンドキューの作成
        commandQueue = device.makeCommandQueue()
        
        generateParticles(count: 5000)
        // シェーダーの設定
        setUpTextures()
        setUpPipelineStates()
        setUpPassDescriptors()
        
        setUpBuffer()
        
        self.view.wantsLayer = true
        self.view.layer?.backgroundColor = NSColor.lightGray.cgColor
    }
    
    private func setUpBuffer() {
        
        let vMatrix = Matrix4.lookAt(eye: camPostion, center: cameraTarget, up: camUp)
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
        
        
        positionBuffer = device.makeBuffer(bytes: billboardPositions, length: MemoryLayout<SIMD3<Float>>.stride * billboardPositions.count, options: [])!
        billboardIndexBuffer = device.makeBuffer(bytes: billboardIndices, length: MemoryLayout<UInt16>.stride * billboardIndices.count, options: [])!
        
        let fullScreenVertices: [FullScreenVertexIn] = [
            .init(position: .init(x: -1, y: 1), texCoord: .init(x: 0, y: 0)),
            .init(position: .init(x: 1, y: 1), texCoord: .init(x: 1, y: 0)),
            .init(position: .init(x: -1, y: -1), texCoord: .init(x: 0, y: 1)),
            .init(position: .init(x: 1, y: -1), texCoord: .init(x: 1, y: 1))
        ]
        
        fullScreenVertexBuffer = device.makeBuffer(bytes: fullScreenVertices, length: MemoryLayout<FullScreenVertexIn>.stride * fullScreenVertices.count, options: [])!
        
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
    
    override func mouseDown(with event: NSEvent) {
        // ウィンドウ座標（通常、origin は左下ですが、アプリに合わせて調整）
        lastMouseLocation = event.locationInWindow
    }
    
    override func mouseDragged(with event: NSEvent) {
        guard let lastLocation = lastMouseLocation else { return }
        let currentLocation = event.locationInWindow
        
        // マウス移動量（ピクセル単位）
        let deltaX = Float(currentLocation.x - lastLocation.x)
        let deltaY = Float(currentLocation.y - lastLocation.y)
        
        // **横回転（ヨー）**
        // 横ドラッグでは、世界座標系の Y 軸（0,1,0）を中心に回転させます
        // マウスの X 方向の移動量に応じた回転角度（ラジアン）を計算
        let horizontalAngle = -deltaX * sensitivity
        let horizontalRotation = Quaternion.rotate(angle: horizontalAngle, axis: SIMD3<Float>(0, 1, 0))
        
        // **縦回転（ピッチ）**
        // 縦ドラッグでは、現在の回転状態から求めたカメラの右方向（ローカル X 軸）を中心に回転させます
        // マウスの Y 方向の移動量に応じた回転角度（ラジアン）を計算
        let verticalAngle = -deltaY * sensitivity
        // 現在の向きで右方向を求める（例えば、初期の右方向は (1,0,0)）
        let rightAxis = currentOrientation.act(SIMD3<Float>(1, 0, 0))
        let verticalRotation = Quaternion.rotate(angle: verticalAngle, axis: rightAxis)
        
        // **回転の合成**
        // どちらの回転も現在の向きに合成します
        // ※ クォータニオンの積は順序依存なので、今回は水平回転を先、縦回転を後に適用しています
        currentOrientation = Quaternion.multiply(horizontalRotation, currentOrientation)
        currentOrientation = Quaternion.multiply(verticalRotation, currentOrientation)
        
        // 現在のマウス位置を記憶（次回ドラッグとの差分計算に使用）
        lastMouseLocation = currentLocation
        
        // ※ ここで、ViewController に対して「回転が変わった」旨を通知し、カメラの view matrix を更新するなどの処理を行ってもよいでしょう
    }
    
    override func mouseUp(with event: NSEvent) {
        lastMouseLocation = nil
    }
}

extension ViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // サイズ変更時の処理（必要に応じて実装）
        
    }

    func draw(in view: MTKView) {
        let rotatedEye = currentOrientation.act(camPostion)
        let rotatedUp = currentOrientation.act(camUp)
        let newVMatrix = Matrix4.lookAt(eye: rotatedEye, center: cameraTarget, up: rotatedUp)
        let uniformsPointer = uniformsBuffer.contents().bindMemory(to: Uniforms.self, capacity: 1)
        uniformsPointer.pointee.vMatrix = newVMatrix
        
        let commandBuffer = commandQueue.makeCommandBuffer()
        let depthMapRenderEncoder = commandBuffer?.makeRenderCommandEncoder(descriptor: depthPassDescriptor)
        depthMapRenderEncoder?.setRenderPipelineState(depthMapPipeLineState)
        depthMapRenderEncoder?.setVertexBuffer(uniformsBuffer, offset: 0, index: 0)
        depthMapRenderEncoder?.setVertexBuffer(positionBuffer, offset: 0, index: 1)
        depthMapRenderEncoder?.setFragmentBuffer(uniformsBuffer, offset: 0, index: 0)
//        深度書き込み用
        depthMapRenderEncoder?.setDepthStencilState(depthStentcilState)
        depthMapRenderEncoder?.drawIndexedPrimitives(type: .triangle, indexCount: billboardIndices.count, indexType: .uint16, indexBuffer: billboardIndexBuffer, indexBufferOffset: 0)
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

extension ViewController: FluidViewDelegate {
    func onKeyDown(with event: NSEvent) {
        // キーコードは macOS 固有のものになります
        // 123: ←, 124: →, 125: ↓, 126: ↑
        let moveSpeed: Float = 0.5
        switch event.keyCode {
        case 123: // 左
            camPostion.x -= moveSpeed
            cameraTarget.x -= moveSpeed
        case 124: // 右
            camPostion.x += moveSpeed
            cameraTarget.x += moveSpeed
        case 125: // 下
            camPostion.y += moveSpeed
            cameraTarget.y += moveSpeed
        case 126: // 上
            camPostion.y -= moveSpeed
            cameraTarget.y -= moveSpeed
        default:
            break
        }
    }
}
