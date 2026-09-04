// dst[i0, i1] = sum_k src0[i0, k] * src1[i1, k], batched over dims 2/3 with src0 broadcast
//
// tiled: a workgroup computes a TM x TN tile of dst, stepping over k in chunks of TK that are
// staged in shared memory. Thread (tx, ty) owns a 4x4 micro tile: 4 consecutive i0 (vec4) at 4
// consecutive i1, so each k step does 16 multiply-adds per 8 shared loads.
// The workgroup grid is flattened to 2D on the host, consecutive workgroups walk tile_m first.

struct Params {
    ne: u32,

    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    ne00: u32,
    ne01: u32,  // reduction dim, equal to src1 ne1
    ne02: u32,
    ne03: u32,

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    ne10: u32,

    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    stride_dst_0: u32,
    stride_dst_1: u32,
    stride_dst_2: u32,
    stride_dst_3: u32,

    n_tiles_m: u32,
    n_tiles_n: u32,
    n_wg: u32,
};

@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> src1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> tile_a: array<f32, TK * TM>;  // [k][m]
var<workgroup> tile_b: array<f32, TK * TN>;  // [k][n]

fn store_row(d: u32, i0: u32, v: vec4<f32>) {
    if (i0 < params.ne0) {
        dst[d + i0 * params.stride_dst_0] = v.x;
    }
    if (i0 + 1u < params.ne0) {
        dst[d + (i0 + 1u) * params.stride_dst_0] = v.y;
    }
    if (i0 + 2u < params.ne0) {
        dst[d + (i0 + 2u) * params.stride_dst_0] = v.z;
    }
    if (i0 + 3u < params.ne0) {
        dst[d + (i0 + 3u) * params.stride_dst_0] = v.w;
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id)        wid: vec3<u32>,
        @builtin(num_workgroups)      num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let wg = wid.x + num_wg.x * wid.y;
    if (wg >= params.n_wg) {
        return;
    }

    let tile_m = wg % params.n_tiles_m;
    let tile_n = (wg / params.n_tiles_m) % params.n_tiles_n;
    let batch  = wg / (params.n_tiles_m * params.n_tiles_n);

    let i2 = batch % params.ne2;
    let i3 = batch / params.ne2;

    // src0 is broadcast over dims 2 and 3
    let a_i2 = i2 / (params.ne2 / params.ne02);
    let a_i3 = i3 / (params.ne3 / params.ne03);

    let a_batch = params.offset_src0 + a_i3 * params.stride_src0_3 + a_i2 * params.stride_src0_2;
    let b_batch = params.offset_src1 + i3 * params.stride_src1_3 + i2 * params.stride_src1_2;

    let m0 = tile_m * u32(TM);
    let n0 = tile_n * u32(TN);

    let tx = lid.x % (u32(TM) / 4u);
    let ty = lid.x / (u32(TM) / 4u);

    var acc0 = vec4<f32>(0.0f);
    var acc1 = vec4<f32>(0.0f);
    var acc2 = vec4<f32>(0.0f);
    var acc3 = vec4<f32>(0.0f);

    for (var k0: u32 = 0; k0 < params.ne01; k0 += u32(TK)) {
        for (var e = lid.x; e < u32(TK * TM); e += u32(WG_SIZE)) {
            let k = e / u32(TM);
            let m = e % u32(TM);
            var v = 0.0f;
            if (k0 + k < params.ne01 && m0 + m < params.ne0) {
                v = src0[a_batch + (k0 + k) * params.stride_src0_1 + (m0 + m) * params.stride_src0_0];
            }
            tile_a[e] = v;
        }
        for (var e = lid.x; e < u32(TK * TN); e += u32(WG_SIZE)) {
            let k = e / u32(TN);
            let n = e % u32(TN);
            var v = 0.0f;
            if (k0 + k < params.ne01 && n0 + n < params.ne1) {
                v = src1[b_batch + (k0 + k) * params.stride_src1_1 + (n0 + n) * params.stride_src1_0];
            }
            tile_b[e] = v;
        }
        workgroupBarrier();

        for (var k: u32 = 0; k < u32(TK); k++) {
            let a_base = k * u32(TM) + tx * 4u;
            let a = vec4<f32>(tile_a[a_base], tile_a[a_base + 1u], tile_a[a_base + 2u], tile_a[a_base + 3u]);
            let b_base = k * u32(TN) + ty * 4u;
            acc0 += a * tile_b[b_base];
            acc1 += a * tile_b[b_base + 1u];
            acc2 += a * tile_b[b_base + 2u];
            acc3 += a * tile_b[b_base + 3u];
        }
        workgroupBarrier();
    }

    let i0 = m0 + tx * 4u;
    let i1 = n0 + ty * 4u;
    let d  = params.offset_dst + i3 * params.stride_dst_3 + i2 * params.stride_dst_2;
    if (i1 < params.ne1) {
        store_row(d + i1 * params.stride_dst_1, i0, acc0);
    }
    if (i1 + 1u < params.ne1) {
        store_row(d + (i1 + 1u) * params.stride_dst_1, i0, acc1);
    }
    if (i1 + 2u < params.ne1) {
        store_row(d + (i1 + 2u) * params.stride_dst_1, i0, acc2);
    }
    if (i1 + 3u < params.ne1) {
        store_row(d + (i1 + 3u) * params.stride_dst_1, i0, acc3);
    }
}
