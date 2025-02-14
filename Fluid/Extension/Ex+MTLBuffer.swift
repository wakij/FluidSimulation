//
//  Ex+MTLBuffer.swift
//  Fluid
//
//  Created by wakita tomoshige on 2025/02/14.
//
import Metal

extension MTLBuffer {
    /// バッファの内容を指定した要素数の配列として返す
    func readArray<T>(count: Int) -> [T] {
        let pointer = self.contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
