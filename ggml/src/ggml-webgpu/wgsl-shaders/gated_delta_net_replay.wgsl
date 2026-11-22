// replay of the pending tokens of a gated delta net (see ggml_gated_delta_net_ext): runs the n_rep[seq] kept pending
// tokens from the initial state and writes the resulting state where gated_delta_net.wgsl (REPLAY) reads it.
// A separate pass, so that neither pass exceeds the default limit of 8 storage buffers per shader stage

@group(0) @binding(0)
var<storage, read_write> src_state: array<f32>;

@group(0) @binding(1)
var<storage, read_write> pend_k: array<f32>;

@group(0) @binding(2)
var<storage, read_write> pend_v: array<f32>;

@group(0) @binding(3)
var<storage, read_write> pend_gb: array<f32>;

@group(0) @binding(4)
var<storage, read_write> n_rep: array<i32>;

// dst of the gdn op, or the fused cache view
@group(0) @binding(5)
var<storage, read_write> dst_state: array<f32>;

struct Params {
    h: u32,
    neq1: u32,
    state_off: u32,
    n_pend: u32,

    spk1: u32,
    spk2: u32,
    spk3: u32,

    spv1: u32,
    spv2: u32,
    spv3: u32,

    spg1: u32,
    spg2: u32,
    spg3: u32,
};

@group(0) @binding(6)
var<uniform> params: Params;

var<workgroup> sh_k: array<f32, S_V>;
var<workgroup> sh_g: array<f32, S_V>;
var<workgroup> wg_n_rep: u32;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let head_id = workgroup_id.x;
    let seq_id = workgroup_id.y;
    let col = local_id.x;

    let iq1 = head_id % params.neq1;

    let state_base = (seq_id * params.h + head_id) * S_V * S_V;

    var state: array<f32, S_V>;
    for (var i = 0u; i < S_V; i++) {
        state[i] = src_state[state_base + col * S_V + i];
    }

    if (col == 0u) {
        wg_n_rep = u32(n_rep[seq_id]);
    }
    // uniform loop bound for the barriers below
    let n_r = workgroupUniformLoad(&wg_n_rep);

    for (var t = 0u; t < n_r; t++) {
        let k_off = seq_id * params.spk3 + t * params.spk2 + iq1 * params.spk1;
        let v_off = seq_id * params.spv3 + t * params.spv2 + head_id * params.spv1;
        let g_off = seq_id * params.spg3 + t * params.spg2 + head_id * params.spg1;

        sh_k[col] = pend_k[k_off + col];

#ifdef KDA
        sh_g[col] = exp(pend_gb[g_off + col]);
#endif

        workgroupBarrier();

        let v_val = pend_v[v_off + col];
        // the betas follow the n_pend gates
        let beta_val = pend_gb[g_off + params.n_pend * params.spg2];

        // same update as gated_delta_net.wgsl, without the attention output
        var kv_col = 0.0;
        var delta_col = 0.0;

#ifdef KDA
        for (var i = 0u; i < S_V; i++) {
            kv_col += (sh_g[i] * state[i]) * sh_k[i];
        }

        delta_col = (v_val - kv_col) * beta_val;

        for (var i = 0u; i < S_V; i++) {
            state[i] = sh_g[i] * state[i] + sh_k[i] * delta_col;
        }
#else
        let g_val = exp(pend_gb[g_off]);

        for (var i = 0u; i < S_V; i++) {
            kv_col += state[i] * sh_k[i];
        }

        delta_col = (v_val - g_val * kv_col) * beta_val;

        for (var i = 0u; i < S_V; i++) {
            state[i] = g_val * state[i] + sh_k[i] * delta_col;
        }
#endif

        workgroupBarrier();
    }

    for (var i = 0u; i < S_V; i++) {
        dst_state[params.state_off + state_base + col * S_V + i] = state[i];
    }
}
