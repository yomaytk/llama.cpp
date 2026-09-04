// sum over the repeats of src0 that map to each dst element
//
// two variants, picked on the host from the number of repeats per dst element:
//  - default: one thread per dst element loops over its repeats (short reductions, wide dst)
//  - REDUCE:  a TX x TY workgroup covers TX consecutive dst elements, the TY threads sharing a
//             dst element split its repeats and are tree-reduced in shared memory (long reductions,
//             e.g. norm weight gradients where dst is a single row and the repeats span all tokens)

struct Params {
    ne: u32,

    offset_src0: u32,
    offset_dst: u32,

    ne00: u32,
    ne01: u32,
    ne02: u32,
    ne03: u32,

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
};

@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn dst_coords(idx: u32) -> vec4<u32> {
    var i = idx;
    let i0 = i % params.ne0;
    i = i / params.ne0;
    let i1 = i % params.ne1;
    i = i / params.ne1;
    let i2 = i % params.ne2;
    let i3 = i / params.ne2;
    return vec4<u32>(i0, i1, i2, i3);
}

#ifdef REDUCE

// scratch[ty * TX + tx]
var<workgroup> scratch: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id)        wid: vec3<u32>,
        @builtin(num_workgroups)      num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx  = lid.x % TX;
    let ty  = lid.x / TX;
    let idx = (wid.x + num_wg.x * wid.y) * TX + tx;

    let nr0   = params.ne00 / params.ne0;
    let nr1   = params.ne01 / params.ne1;
    let nr2   = params.ne02 / params.ne2;
    let nr3   = params.ne03 / params.ne3;
    let n_rep = nr0 * nr1 * nr2 * nr3;

    var acc = 0.0f;
    if (idx < params.ne) {
        let c = dst_coords(idx);
        for (var r = ty; r < n_rep; r += TY) {
            var t = r;
            let k0 = t % nr0;
            t = t / nr0;
            let k1 = t % nr1;
            t = t / nr1;
            let k2 = t % nr2;
            let k3 = t / nr2;
            acc += src0[params.offset_src0 +
                        (c.w + k3 * params.ne3) * params.stride_src0_3 +
                        (c.z + k2 * params.ne2) * params.stride_src0_2 +
                        (c.y + k1 * params.ne1) * params.stride_src0_1 +
                        (c.x + k0 * params.ne0) * params.stride_src0_0];
        }
    }

    scratch[lid.x] = acc;
    workgroupBarrier();

    var offset: u32 = TY / 2;
    while (offset > 0) {
        if (ty < offset) {
            scratch[lid.x] += scratch[lid.x + offset * TX];
        }
        offset = offset / 2;
        workgroupBarrier();
    }

    if (ty == 0 && idx < params.ne) {
        dst[params.offset_dst + idx] = scratch[tx];
    }
}

#else

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups)       num_wg: vec3<u32>) {
    let idx = gid.x + (num_wg.x * u32(WG_SIZE)) * gid.y;
    if (idx >= params.ne) {
        return;
    }

    let c = dst_coords(idx);

    var acc = 0.0f;
    for (var s3 = c.w; s3 < params.ne03; s3 += params.ne3) {
        for (var s2 = c.z; s2 < params.ne02; s2 += params.ne2) {
            for (var s1 = c.y; s1 < params.ne01; s1 += params.ne1) {
                for (var s0 = c.x; s0 < params.ne00; s0 += params.ne0) {
                    acc += src0[params.offset_src0 + s3 * params.stride_src0_3 + s2 * params.stride_src0_2 +
                                s1 * params.stride_src0_1 + s0 * params.stride_src0_0];
                }
            }
        }
    }

    dst[params.offset_dst + idx] = acc;
}

#endif
