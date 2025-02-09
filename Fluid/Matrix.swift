import simd
import Foundation

// MARK: - 4×4行列演算

struct Matrix4 {
    
    /// 新しい行列を生成（初期値は identity）
    static func create() -> simd_float4x4 {
        return matrix_identity_float4x4
    }
    
    /// 単位行列を返す
    static func identity() -> simd_float4x4 {
        return matrix_identity_float4x4
    }
    
    /// 行列同士の積（引数の順序は JavaScript版と同様に mat2 * mat1 を返す）
    static func multiply(_ mat1: simd_float4x4, _ mat2: simd_float4x4) -> simd_float4x4 {
        return mat2 * mat1
    }
    
    /// 与えられた行列の各列（x,y,z成分）にスケールをかけた行列を返す
    static func scale(_ mat: simd_float4x4, by vec: SIMD3<Float>) -> simd_float4x4 {
        var result = mat
        result.columns.0 *= vec.x
        result.columns.1 *= vec.y
        result.columns.2 *= vec.z
        return result
    }
    
    /// 与えられた行列に平行移動を加えた行列を返す
    /// （内部では、4列目（translation 成分）を「mat * (vec,1)」の形で更新します）
    static func translate(_ mat: simd_float4x4, by vec: SIMD3<Float>) -> simd_float4x4 {
        var result = mat
        result.columns.3 = mat.columns.0 * vec.x +
                           mat.columns.1 * vec.y +
                           mat.columns.2 * vec.z +
                           mat.columns.3
        return result
    }
    
    /// 指定軸（vec）周りに angle [rad] だけ回転させた行列を返す
    /// （ここでは、simd_quatf を用いて回転行列を生成し、右側から掛ける形としています）
    static func rotate(_ mat: simd_float4x4, angle: Float, axis: SIMD3<Float>) -> simd_float4x4 {
        let normalizedAxis = normalize(axis)
        let rotationMatrix = simd_float4x4(simd_quatf(angle: angle, axis: normalizedAxis))
        return mat * rotationMatrix
    }
    
    static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let z = normalize(center - eye)
        let x = normalize(cross(up, z))
        let y = cross(z, x)
        let translation = SIMD3<Float>(-dot(x, eye), -dot(y, eye), -dot(z, eye))
        return simd_float4x4(columns: (
            SIMD4<Float>(x.x, y.x, z.x, 0),
            SIMD4<Float>(x.y, y.y, z.y, 0),
            SIMD4<Float>(x.z, y.z, z.z, 0),
            SIMD4<Float>(translation.x, translation.y, translation.z, 1)
        ))
    }
    
    /// 透視投影行列を返す
    /// - Parameters:
    ///   - fovy: 垂直視野角（度単位；JavaScript版では fovy/2 をラジアンに変換して tan を取っています）
    ///   - aspect: アスペクト比
    ///   - near: ニアクリップ面
    ///   - far: ファークリップ面
    static func perspective(fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let ys = 1.0 / tan(fovy * .pi / 360.0)
        let xs = ys / aspect
        let zs = far / (far - near)
        
        return simd_float4x4(columns: (
            SIMD4<Float>(xs, 0,    0,   0),
            SIMD4<Float>(0,  ys,  0,   0),
            SIMD4<Float>(0,   0,   zs, 1),
            SIMD4<Float>(0,   0,   -near * zs,  0)
        ))
    }
    
    /// 正射影行列を返す
    static func ortho(left: Float, right: Float,
                      bottom: Float, top: Float,
                      near: Float, far: Float) -> simd_float4x4 {
        let w = right - left
        let h = top - bottom
        let d = far - near
        let m00 = 2 / w
        let m11 = 2 / h
        let m22 = -2 / d
        let m30 = -(left + right) / w
        let m31 = -(top + bottom) / h
        let m32 = -(far + near) / d
        return simd_float4x4(columns: (
            SIMD4<Float>(m00, 0,    0,   0),
            SIMD4<Float>(0,   m11,  0,   0),
            SIMD4<Float>(0,   0,   m22,  0),
            SIMD4<Float>(m30, m31, m32,  1)
        ))
    }
    
    /// 転置行列を返す
    static func transpose(_ mat: simd_float4x4) -> simd_float4x4 {
        return mat.transpose
    }
    
    /// 逆行列を返す
    static func inverse(_ mat: simd_float4x4) -> simd_float4x4 {
        return mat.inverse
    }
}

// MARK: - クォータニオン演算

struct Quaternion {
    // 内部では simd_quatf を用いる
    
    /// 新しいクォータニオン（単位クォータニオン）を返す
    static func create() -> simd_quatf {
        return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }
    
    /// 単位クォータニオン（＝回転なし）を返す
    static func identity() -> simd_quatf {
        return simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
    }
    
    /// 逆クォータニオン（共役）を返す
    /// （単位クォータニオンなら q⁻¹ = conjugate(q)）
    static func inverse(_ q: simd_quatf) -> simd_quatf {
        return q.inverse
    }
    
    /// クォータニオン同士の積（合成回転）を返す
    static func multiply(_ q1: simd_quatf, _ q2: simd_quatf) -> simd_quatf {
        return q1 * q2
    }
    
    /// 指定軸・角度の回転クォータニオンを返す
    /// （angle は [rad] 単位）
    static func rotate(angle: Float, axis: SIMD3<Float>) -> simd_quatf {
        let normalizedAxis = normalize(axis)
        return simd_quatf(angle: angle, axis: normalizedAxis)
    }
    
    /// クォータニオン q による回転を vec に適用して返す
    static func toVec3(_ vec: SIMD3<Float>, by q: simd_quatf) -> SIMD3<Float> {
        return q.act(vec)
    }
    
    /// クォータニオンから 4×4 の回転行列を生成して返す
    static func toMatrix(_ q: simd_quatf) -> simd_float4x4 {
        return simd_float4x4(q)
    }
    
    /// 球面線形補間（slerp）を行い、補間結果のクォータニオンを返す
    static func slerp(_ q1: simd_quatf, _ q2: simd_quatf, t: Float) -> simd_quatf {
        return simd_slerp(q1, q2, t)
    }
}

// MARK: - 幾何体生成（トーラス／スフィア／キューブ）

/// メッシュデータ（頂点座標，法線，色，テクスチャ座標，インデックス）
struct Mesh {
    var positions: [Float]  // 頂点座標（x,y,z のフラットな配列）
    var normals: [Float]    // 法線（x,y,z）
    var colors: [Float]     // RGBA
    var texCoords: [Float]  // テクスチャ座標（u,v）
    var indices: [UInt16]   // インデックス
}

/// hsva(h, s, v, a) --- JavaScript版と同様に HSV(A) から RGBA を求める
/// ※s, v, a が 1 を超える場合は nil を返す
func hsva(_ h: Float, _ s: Float, _ v: Float, _ a: Float) -> SIMD4<Float>? {
    if s > 1 || v > 1 || a > 1 { return nil }
    let th = fmod(h, 360)
    let i = floor(th / 60)
    let f = th / 60 - i
    let m = v * (1 - s)
    let n = v * (1 - s * f)
    let k = v * (1 - s * (1 - f))
    if s == 0 {
        return SIMD4<Float>(v, v, v, a)
    } else {
        let rValues: [Float] = [v, n, m, m, k, v]
        let gValues: [Float] = [k, v, v, n, m, m]
        let bValues: [Float] = [m, m, k, v, v, n]
        let index = Int(i) % 6
        return SIMD4<Float>(rValues[index], gValues[index], bValues[index], a)
    }
}

/// トーラス形状のメッシュを生成する
/// - Parameters:
///   - row: 周方向分割数
///   - column: 管状断面の分割数
///   - irad: 内側半径（チューブ半径）
///   - orad: 外側半径（トーラス全体の大きさ）
///   - color: 固定色（nil の場合は hsva により自動設定）
func torus(row: Int, column: Int, irad: Float, orad: Float, color: SIMD4<Float>? = nil) -> Mesh {
    var pos = [Float]()
    var nor = [Float]()
    var col = [Float]()
    var st  = [Float]()
    var idx = [UInt16]()
    
    for i in 0...row {
        let r = (Float.pi * 2 / Float(row)) * Float(i)
        let rr = cos(r)
        let ry = sin(r)
        for j in 0...column {
            let tr = (Float.pi * 2 / Float(column)) * Float(j)
            let tx = (rr * irad + orad) * cos(tr)
            let ty = ry * irad
            let tz = (rr * irad + orad) * sin(tr)
            let rx = rr * cos(tr)
            let rz = rr * sin(tr)
            let tc: SIMD4<Float>
            if let fixedColor = color {
                tc = fixedColor
            } else {
                tc = hsva(360 / Float(column) * Float(j), 1, 1, 1) ?? SIMD4<Float>(1, 1, 1, 1)
            }
            let rs = (1 / Float(column)) * Float(j)
            var rt = (1 / Float(row)) * Float(i) + 0.5
            if rt > 1.0 { rt -= 1.0 }
            rt = 1.0 - rt
            
            pos.append(contentsOf: [tx, ty, tz])
            nor.append(contentsOf: [rx, ry, rz])
            col.append(contentsOf: [tc.x, tc.y, tc.z, tc.w])
            st.append(contentsOf: [rs, rt])
        }
    }
    
    for i in 0..<row {
        for j in 0..<column {
            let r = UInt16((column + 1) * i + j)
            idx.append(r)
            idx.append(r + UInt16(column) + 1)
            idx.append(r + 1)
            
            idx.append(r + UInt16(column) + 1)
            idx.append(r + UInt16(column) + 2)
            idx.append(r + 1)
        }
    }
    
    return Mesh(positions: pos, normals: nor, colors: col, texCoords: st, indices: idx)
}

/// 球形のメッシュを生成する
func sphere(row: Int, column: Int, rad: Float, color: SIMD4<Float>? = nil) -> Mesh {
    var pos = [Float]()
    var nor = [Float]()
    var col = [Float]()
    var st  = [Float]()
    var idx = [UInt16]()
    
    for i in 0...row {
        let r = Float.pi / Float(row) * Float(i)
        let ry = cos(r)
        let rr = sin(r)
        for j in 0...column {
            let tr = (Float.pi * 2 / Float(column)) * Float(j)
            let tx = rr * rad * cos(tr)
            let ty = ry * rad
            let tz = rr * rad * sin(tr)
            let rx = rr * cos(tr)
            let rz = rr * sin(tr)
            let tc: SIMD4<Float>
            if let fixedColor = color {
                tc = fixedColor
            } else {
                tc = hsva(360 / Float(row) * Float(i), 1, 1, 1) ?? SIMD4<Float>(1, 1, 1, 1)
            }
            
            pos.append(contentsOf: [tx, ty, tz])
            nor.append(contentsOf: [rx, ry, rz])
            col.append(contentsOf: [tc.x, tc.y, tc.z, tc.w])
            st.append(contentsOf: [1 - (1 / Float(column)) * Float(j), (1 / Float(row)) * Float(i)])
        }
    }
    
    for i in 0..<row {
        for j in 0..<column {
            let r = UInt16((column + 1) * i + j)
            idx.append(r)
            idx.append(r + 1)
            idx.append(r + UInt16(column) + 2)
            
            idx.append(r)
            idx.append(r + UInt16(column) + 2)
            idx.append(r + UInt16(column) + 1)
        }
    }
    
    return Mesh(positions: pos, normals: nor, colors: col, texCoords: st, indices: idx)
}

/// キューブ（立方体）のメッシュを生成する
func cube(side: Float, color: SIMD4<Float>? = nil) -> Mesh {
    let hs = side * 0.5
    let pos: [Float] = [
        -hs, -hs,  hs,   hs, -hs,  hs,   hs,  hs,  hs,  -hs,  hs,  hs,
        -hs, -hs, -hs,  -hs,  hs, -hs,   hs,  hs, -hs,   hs, -hs, -hs,
        -hs,  hs, -hs,  -hs,  hs,  hs,   hs,  hs,  hs,   hs,  hs, -hs,
        -hs, -hs, -hs,   hs, -hs, -hs,   hs, -hs,  hs,  -hs, -hs,  hs,
         hs, -hs, -hs,   hs,  hs, -hs,   hs,  hs,  hs,   hs, -hs,  hs,
        -hs, -hs, -hs,  -hs, -hs,  hs,  -hs,  hs,  hs,  -hs,  hs, -hs
    ]
    let nor: [Float] = [
        -1, -1,  1,    1, -1,  1,    1,  1,  1,   -1,  1,  1,
        -1, -1, -1,   -1,  1, -1,    1,  1, -1,    1, -1, -1,
        -1,  1, -1,   -1,  1,  1,    1,  1,  1,    1,  1, -1,
        -1, -1, -1,    1, -1, -1,    1, -1,  1,   -1, -1,  1,
         1, -1, -1,    1,  1, -1,    1,  1,  1,    1, -1,  1,
        -1, -1, -1,   -1, -1,  1,   -1,  1,  1,   -1,  1, -1
    ]
    var col = [Float]()
    let vertexCount = pos.count / 3
    for i in 0..<vertexCount {
        let tc: SIMD4<Float>
        if let fixedColor = color {
            tc = fixedColor
        } else {
            tc = hsva(360 / Float(vertexCount) * Float(i), 1, 1, 1) ?? SIMD4<Float>(1, 1, 1, 1)
        }
        col.append(contentsOf: [tc.x, tc.y, tc.z, tc.w])
    }
    let st: [Float] = [
         0, 0,  1, 0,  1, 1,  0, 1,
         0, 0,  1, 0,  1, 1,  0, 1,
         0, 0,  1, 0,  1, 1,  0, 1,
         0, 0,  1, 0,  1, 1,  0, 1,
         0, 0,  1, 0,  1, 1,  0, 1,
         0, 0,  1, 0,  1, 1,  0, 1
    ]
    let idx: [UInt16] = [
         0,  1,  2,  0,  2,  3,
         4,  5,  6,  4,  6,  7,
         8,  9, 10,  8, 10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23
    ]
    return Mesh(positions: pos, normals: nor, colors: col, texCoords: st, indices: idx)
}



