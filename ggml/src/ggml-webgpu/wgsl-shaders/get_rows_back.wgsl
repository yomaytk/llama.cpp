// gradient of get_rows: dst[idx[i]] += grad[i], rows nobody points to are zero
//
// two dispatches, no f32 atomics needed:
//  - ZERO: clear dst
//  - default: one workgroup per index i. The workgroup of the first occurrence of a row value owns
//    that row and folds in every later duplicate, the others exit. The work is
//    O(n_idx^2 + n_idx * ne00 + n_rows * ne00) instead of O(n_rows * ne00 * n_idx) for a
//    per-element scan, which matters when dst is a vocab-sized embedding table.

struct Params {
    ne: u32,    // dst elements

    offset_grad: u32,
    offset_idx: u32,
    offset_dst: u32,

    ne00: u32,  // row size
    n_idx: u32, // number of indices, can be less than the rows of grad

    stride_grad_1: u32,
    stride_idx_0: u32,

    ne0: u32,
    ne1: u32,
};

#ifdef ZERO

// the clear pass only touches dst, and unused bindings are dropped from the inferred layout
@group(0) @binding(0)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups)       num_wg: vec3<u32>) {
    let idx = gid.x + (num_wg.x * u32(WG_SIZE)) * gid.y;
    if (idx >= params.ne) {
        return;
    }
    dst[params.offset_dst + idx] = 0.0f;
}

#else

@group(0) @binding(0)
var<storage, read_write> grad: array<f32>;

@group(0) @binding(1)
var<storage, read_write> row_idx: array<i32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

fn idx_at(i: u32) -> i32 {
    return row_idx[params.offset_idx + i * params.stride_idx_0];
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id)        wid: vec3<u32>,
        @builtin(num_workgroups)      num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = wid.x + num_wg.x * wid.y;
    if (i >= params.n_idx) {
        return;
    }

    let row = idx_at(i);

    // every thread reads the same indices, so this is uniform across the workgroup
    for (var j: u32 = 0; j < i; j++) {
        if (idx_at(j) == row) {
            return;
        }
    }

    let dst_row  = params.offset_dst + u32(row) * params.ne0;
    let grad_row = params.offset_grad + i * params.stride_grad_1;

    for (var col = lid.x; col < params.ne0; col += u32(WG_SIZE)) {
        var sum = grad[grad_row + col];
        for (var j = i + 1u; j < params.n_idx; j++) {
            if (idx_at(j) == row) {
                sum += grad[params.offset_grad + j * params.stride_grad_1 + col];
            }
        }
        dst[dst_row + col] = sum;
    }
}

#endif
