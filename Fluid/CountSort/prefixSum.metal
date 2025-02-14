#include <metal_stdlib>
using namespace metal;


constant uint THREADGROUP_SIZE = 256; // 各 threadgroup の半分の要素数
// 1ブロックあたりの要素数は 2 * THREADGROUP_SIZE になります。

// ───────────────────────────────
// 1. ブロックごとのスキャン（Blelloch スキャン）
//1スレッドで2つの値を読み込む
//256スレッドで512個の値を読み込める(maxで)
//スレッド0は0番目と256番目を読み込む and 書き込む
kernel void blockScanKernel(
    device const uint *inData         [[ buffer(0) ]],
    device uint       *outData        [[ buffer(1) ]],
    device uint       *blockSums      [[ buffer(2) ]],
    constant uint      &numElements    [[ buffer(3) ]],
    uint tid                           [[ thread_index_in_threadgroup ]],
    uint groupID                       [[ threadgroup_position_in_grid ]]
) {
    //2分木を確実に作成するためにこのようにする
    const uint elementsPerGroup = THREADGROUP_SIZE * 2;
    threadgroup uint sharedData[elementsPerGroup];

    
//    --------ただ入力配列を共有メモリに読み込んでいるだけ---------------
    uint startIndex = groupID * elementsPerGroup;
    uint index1 = startIndex + tid; //globalなindex
    uint index2 = startIndex + tid + THREADGROUP_SIZE;
    sharedData[tid] = (index1 < numElements) ? inData[index1] : 0;
    sharedData[tid + THREADGROUP_SIZE] = (index2 < numElements) ? inData[index2] : 0;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
//    -----------------------------------------------------------
    
    // Up-sweep（リダクション）フェーズ
    for (uint stride = 1; stride < elementsPerGroup; stride *= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < elementsPerGroup) {
            sharedData[index] += sharedData[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // 各ブロックの総和を blockSums に保存。さらに down-sweep のために最後の要素を 0 に設定。
    if (tid == 0) {
        blockSums[groupID] = sharedData[elementsPerGroup - 1];
        sharedData[elementsPerGroup - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Down-sweep フェーズ(交換と加算)
    for (uint stride = elementsPerGroup / 2; stride >= 1; stride /= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < elementsPerGroup) {
            uint temp = sharedData[index - stride];
            sharedData[index - stride] = sharedData[index];
            sharedData[index] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (index1 < numElements) {
        outData[index1] = sharedData[tid];
    }
    if (index2 < numElements) {
        outData[index2] = sharedData[tid + THREADGROUP_SIZE];
    }
}


// ───────────────────────────────
// 2. ブロック総和に対するスキャン
kernel void scanBlockSumsKernel(
    device uint   *blockSums   [[ buffer(0) ]],
    constant uint  &numBlocks   [[ buffer(1) ]],
    constant uint  &n           [[ buffer(2) ]],
    uint tid                     [[ thread_index_in_threadgroup ]]
) {
    // 仮に最大 1024 ブロックまで対応（numBlocks がこれ以下であるとする）
    threadgroup uint sharedData[1024];
    
    // まず、tid < n の範囲でブロック総和を共有メモリにロードします。
    if (tid < n) {
        sharedData[tid] = (tid < numBlocks) ? blockSums[tid] : 0;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Up-sweep
    for (uint stride = 1; stride < n; stride *= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            sharedData[index] += sharedData[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        sharedData[n - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Down-sweep
    for (uint stride = n / 2; stride >= 1; stride /= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            uint temp = sharedData[index - stride];
            sharedData[index - stride] = sharedData[index];
            sharedData[index] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid < numBlocks) {
        blockSums[tid] = sharedData[tid];
    }
}


// ───────────────────────────────
// 3. 各ブロックのスキャン結果にオフセットを加算
kernel void addBlockOffsetsKernel(
    device uint   *outData   [[ buffer(0) ]],
    device const uint *blockSums [[ buffer(1) ]],
    constant uint  &numElements  [[ buffer(2) ]],
    uint tid                     [[ thread_index_in_threadgroup ]],
    uint groupID                 [[ threadgroup_position_in_grid ]]
) {
    const uint elementsPerGroup = THREADGROUP_SIZE * 2;
    uint startIndex = groupID * elementsPerGroup;
    uint offset = blockSums[groupID];
    
    uint index1 = startIndex + tid;
    uint index2 = startIndex + tid + THREADGROUP_SIZE;
    
    if (index1 < numElements) {
        outData[index1] += offset;
    }
    if (index2 < numElements) {
        outData[index2] += offset;
    }
}

