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

struct Particle {
    var position: SIMD3<Float>
    var v: SIMD3<Float>
    var force: SIMD3<Float>
    var lastAcceration: SIMD3<Float>
    var density: Float
    var nearDensity: Float
}

func initDambreak(initHalfBoxSize: SIMD3<Float>, numParticles: Int, kernelRadius: Float) -> [Particle] {
    var particles: [Particle] = []
    let distFactor: Float = 0.5  // 間隔係数
     
    // X方向の開始・終了位置（左右0.95倍）
    let startX = -initHalfBoxSize.x * 0.95
    let endX = initHalfBoxSize.x * 0.95
    
    // Y方向の開始位置（下側0.95倍） ※外側ループは粒子数に達するまで続ける
    let startY = initHalfBoxSize.y * 0.95
    
    // Z方向は下側から0まで（0 * initHalfBoxSize.z は0になります）
    let startZ = -initHalfBoxSize.z * 0.95
    let endZ: Float = 0.0
    
    var y = startY
    // 外側ループは粒子数に達するまでループします
    while particles.count < numParticles {
        var x = startX
        while x < endX && particles.count < numParticles {
            var z = startZ
            while z < endZ && particles.count < numParticles {
                // 少しだけランダムなジッターを加える
                let jitter = Float.random(in: 0..<0.001)
                let pos = SIMD3<Float>(x + jitter, y + jitter, z + jitter)
                let particle = Particle(position: pos,
                                        v: SIMD3<Float>(repeating: 0),
                                        force: SIMD3<Float>(repeating: 0),
                                        lastAcceration: .zero,
                                        density: 0,
                                        nearDensity: 0)
                particles.append(particle)
                z += distFactor * kernelRadius
            }
            x += distFactor * kernelRadius
        }
        y -= distFactor * kernelRadius
    }
    
    return particles
}

func initCube() -> [Particle] {
    var particles: [Particle] = []
    for _ in 0..<10000 {
        let x = Float.random(in: -0.2..<0.2)
        let y = Float.random(in: 1.6..<2.0)
        let z = Float.random(in: -0.2..<0.2)
        particles.append(Particle(position: .init(x: x, y: y, z: z), v: .zero, force: .zero, lastAcceration: .zero, density: 0, nearDensity: 0))
    }
    return particles
}

struct Uniforms {
    var vMatrix: simd_float4x4
    var pMatrix: simd_float4x4
    var invProjectionMatrix: simd_float4x4
    var texelSize: simd_float2
    var size: Float
    var sphereRadius: Float //size/2
    var dirLight: SIMD3<Float>
    var specularPower: Float
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

struct Environment {
    var xGrids: Int32
    var yGrids: Int32
    var zGrids: Int32
    var cellSize: Float
    var xHalf: Float
    var yHalf: Float
    var zHalf: Float
    var offset: Float
    
    init() {
        let kernelRadius = 0.07
        self.cellSize = Float(1.0 * kernelRadius)
        self.xHalf = 2.0
        self.yHalf = 2.0
        self.zHalf = 2.0
        let xLen = 2.0 * xHalf
        let yLen = 2.0 * yHalf
        let zLen = 2.0 * zHalf
        let sentinel = 4 * cellSize
        self.xGrids = Int32(ceil((xLen + sentinel) / cellSize))
        self.yGrids = Int32(ceil((yLen + sentinel) / cellSize))
        self.zGrids = Int32(ceil((zLen + sentinel) / cellSize))
        self.offset = sentinel / 2;
    }
    
    func gridNum() -> Int32 {
        return xGrids * yGrids * zGrids
    }
}

struct SPHParams {
    var mass: Float
    var kernelRadius: Float
    var kernelRadiusPow2: Float
    var kernelRadiusPow5: Float
    var kernelRadiusPow6: Float
    var kernelRadiusPow9: Float
    var dt: Float
    var stiffness: Float
    var nearStiffness: Float
    var restDensity: Float
    var viscosity: Float
    var surfaceTensionCoefficient: Float
    var n: UInt32
    
    init(n: UInt32) {
        self.mass = 1.0
        self.kernelRadius = 0.07
        self.kernelRadiusPow2 = pow(self.kernelRadius, 2)
        self.kernelRadiusPow5 = pow(self.kernelRadius, 5)
        self.kernelRadiusPow6 = pow(self.kernelRadius, 6)
        self.kernelRadiusPow9 = pow(self.kernelRadius, 9)
        self.stiffness = 20 //最終的な水面の厚みに影響を与えている 値が大きくと動きがダイナミックになって不安定になる
        self.nearStiffness = 1.0
        self.restDensity = 15000 //初期状態と大きく乖離していると不安定性の原因になる
        self.viscosity = 100 //粘性を上げれば数値的に安定する
        self.dt = 0.006
        self.surfaceTensionCoefficient = 2.0
        self.n = n
    }
}

struct RealBoxSize {
    var xHalf: Float
    var yHalf: Float
    var zHalf: Float
}

class ViewController: NSViewController {
    
    var metalView: FluidView!
    
    // Metal関連のプロパティ
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    
//    bufferを関連
    var uniformsBuffer: MTLBuffer!
    var particleBuffer: MTLBuffer!
    var sortedParticleBuffer: MTLBuffer!
    var prefixSumBuffer: MTLBuffer!
    var sphEnvironmentBuffer: MTLBuffer!
    var sphParamsBuffer: MTLBuffer!
    var realBoxSizeBuffer: MTLBuffer!
    var cellParticleCountBuffer: MTLBuffer!
    var particleCellOffsetBuffer: MTLBuffer!
    
//    設定値
    var camPostion: SIMD3<Float> = .init(x: 0, y: 0, z: -5)
    var camUp: SIMD3<Float> = .init(x: 0, y: 1, z: 0)
    var cameraTarget: SIMD3<Float> = .zero
    var currentOrientation: simd_quatf = Quaternion.identity()
    var lastMouseLocation: CGPoint?
    let sensitivity: Float = 0.005
    
    let particleSize: Float = 0.08;
    let particleNum: UInt32 = 10000;
    
    var sphEnv: Environment!
    var sphParams: SPHParams!
    
    
    var prefixSumKenel: PrefixSum!
    var countSortKenel: CountSort!
    
    var gridBuildKernel: GridBuilder!
    var fluidRender: FluidRenderer!
    var particleUtil: ParticleUtil!
    var sphSimulator: SPHSimluator!
    
    var realBoxSize: RealBoxSize!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Metalデバイスの取得
        device = MTLCreateSystemDefaultDevice()
        guard let library = device.makeDefaultLibrary() else { return }
        
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
        
        self.sphParams = SPHParams(n: particleNum)
        self.sphEnv = Environment()
        self.realBoxSize = RealBoxSize(xHalf: 0.7, yHalf: 2.0, zHalf: 0.7)
        
        let testParaticles = initDambreak(initHalfBoxSize: .init(x: realBoxSize.xHalf, y: realBoxSize.yHalf, z: realBoxSize.zHalf), numParticles: Int(particleNum), kernelRadius: 0.07)
//        let testParaticles = initCube()
        particleBuffer = device.makeBuffer(bytes: testParaticles, length: MemoryLayout<Particle>.stride * Int(particleNum), options: [])
        sortedParticleBuffer = device.makeBuffer(length: MemoryLayout<Particle>.stride * Int(particleNum), options: [])!
        prefixSumBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * Int(sphEnv.gridNum()), options: [])!
        sphEnvironmentBuffer = device.makeBuffer(bytes: &sphEnv, length: MemoryLayout<Environment>.stride, options: [])!
        sphParamsBuffer = device.makeBuffer(bytes: &sphParams, length: MemoryLayout<SPHParams>.stride, options: [])!
        realBoxSizeBuffer = device.makeBuffer(bytes: &realBoxSize, length: MemoryLayout<RealBoxSize>.stride, options: [])!
        cellParticleCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * Int(sphEnv.gridNum()), options: [])!
        particleCellOffsetBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * Int(particleNum), options: [])!
        
        let positionBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride * Int(particleNum), options: [])!
        
        let vMatrix = Matrix4.lookAt(eye: camPostion, center: cameraTarget, up: camUp)
        let pMatrix = Matrix4.perspective(fovy: 45, aspect: Float(metalView.frame.width / metalView.frame.height), near: 0.01, far: 10)
        let invProjectionMatrix = pMatrix.inverse
        
        var uniforms = Uniforms(
            vMatrix: vMatrix,
            pMatrix: pMatrix,
            invProjectionMatrix: invProjectionMatrix,
            texelSize: .init(x: 1.0 / Float(metalView.frame.width), y: 1.0 / Float(metalView.frame.height)),
            size: particleSize,
            sphereRadius: particleSize,
            dirLight: .init(x: 0, y: 10, z: -10),
            specularPower: 5
        )
        
        uniformsBuffer = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<Uniforms>.stride, options: [])!
        
        gridBuildKernel = try! GridBuilder(
            device: device,
            library: library,
            particleBuffer: particleBuffer,
            cellParticleCountBuffer: cellParticleCountBuffer,
            particleCellOffsetBuffer: particleCellOffsetBuffer,
            environmentBuffer: sphEnvironmentBuffer,
            particleNum: Int(particleNum),
            gridNum: Int(sphEnv.gridNum()))
        
        sphSimulator = try! .init(
            device: device,
            library: library,
            particleBuffer: particleBuffer,
            sortedParticleBuffer: sortedParticleBuffer,
            prefixSumBuffer: prefixSumBuffer,
            environmentBuffer: sphEnvironmentBuffer,
            sphParamsBuffer: sphParamsBuffer,
            realBoxSizeBuffer: realBoxSizeBuffer,
            particleNum: particleNum)
        
        let fluidRenderParams: FluidRenderParams = .init(
            particleSize: particleSize,
            height: Float(metalView.frame.height),
            particleNum: particleNum)
        
        fluidRender = try! FluidRenderer(
            device: device,
            library: library,
            metalView: metalView,
            params: fluidRenderParams,
            uniformBuffer: uniformsBuffer,
            positionBuffer: positionBuffer)
        
        particleUtil = try! .init(
            device: device,
            library: library,
            particleBuffer: particleBuffer,
            positionBuffer: positionBuffer,
            particleNum: Int(particleNum))
        
        prefixSumKenel = try! .init(
            device: device,
            library: library,
            gridNum: Int(sphEnv.gridNum()),
            inputBuffer: cellParticleCountBuffer,
            outputBuffer: prefixSumBuffer)
        
        countSortKenel = try! .init(
            device: device,
            library: library,
            sourceParticlesBuffer: particleBuffer,
            targetParticlesBuffer: sortedParticleBuffer,
            cellParticleCountPrefixSumBuffer: prefixSumBuffer,
            particleCellOffsetBuffer: particleCellOffsetBuffer,
            environmentBuffer: sphEnvironmentBuffer,
            sphParamsBuffer: sphParamsBuffer,
            particleNum: particleNum)
        
        self.view.wantsLayer = true
        self.view.layer?.backgroundColor = NSColor.lightGray.cgColor
    }
    
    func cellPosition(v: SIMD3<Float>, env: Environment) -> SIMD3<Int32> {
        let xi = Int32(floor((v.x + env.xHalf + env.offset) / env.cellSize));
        let yi = Int32(floor((v.y + env.yHalf + env.offset) / env.cellSize));
        let zi = Int32(floor((v.z + env.zHalf + env.offset) / env.cellSize));
        return SIMD3<Int32>(xi, yi, zi);
    }

    func cellNumberFromId(xi: Int32, yi: Int32, zi: Int32, env: Environment) -> Int32 {
        return xi + yi * env.xGrids + zi * env.xGrids * env.yGrids;
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
        
        let realBoxPointer = realBoxSizeBuffer.contents().bindMemory(to: RealBoxSize.self, capacity: 1)
        realBoxPointer.pointee.xHalf = realBoxSize.xHalf
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
                let drawable = metalView.currentDrawable else { return }
        
        for _ in 0..<2 {
    //        copyする
            gridBuildKernel.clean(commandBuffer: commandBuffer)
            gridBuildKernel.build(commandBuffer: commandBuffer)
            
    //        累積部分和を計算する -> ソートする
            prefixSumKenel.execute(commandBuffer: commandBuffer)
            countSortKenel.excute(commandBuffer: commandBuffer)
            
    //        シミュレーションを行う
            sphSimulator.updateDensity(commandBuffer: commandBuffer)
//            ソート済みparticleBufferに上のdensity計算の結果が書き込まれていないので書き込む
            countSortKenel.excute(commandBuffer: commandBuffer)
            sphSimulator.updateForce(commandBuffer: commandBuffer)
            sphSimulator.updatePosition(commandBuffer: commandBuffer)
        }
        
//        シミュレーション結果から描画に必要なものを取り出す
        particleUtil.copy(commandBuffer: commandBuffer)
        fluidRender.render(commandBuffer: commandBuffer, metalView: metalView)
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
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
        case 29:
            break
//            realBoxSize.xHalf *= 0.95
        case 49:
            break
//            realBoxSize.xHalf *= 1.1
        default:
            break
        }
    }
}
