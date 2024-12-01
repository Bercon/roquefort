function buildShadersVorticity({ device, computeShaders, source }) {
    computeShaders.vorticity = new ComputeShader("vorticity", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read> velocity_in : array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> vorticity : array<f32>;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let index = to_index(global_id);
        let id = vec3i(global_id);

        let vel_left   = velocity_in[clamp_to_edge(id - vec3i(1, 0, 0))];
        let vel_right  = velocity_in[clamp_to_edge(id + vec3i(1, 0, 0))];
        let vel_bottom = velocity_in[clamp_to_edge(id - vec3i(0, 1, 0))];
        let vel_top    = velocity_in[clamp_to_edge(id + vec3i(0, 1, 0))];
        let vel_back   = velocity_in[clamp_to_edge(id - vec3i(0, 0, 1))];
        let vel_front  = velocity_in[clamp_to_edge(id + vec3i(0, 0, 1))];

        let dvz_dy = (vel_top.z - vel_bottom.z) * 0.5 * u.rdx;
        let dvy_dz = (vel_front.y - vel_back.y) * 0.5 * u.rdx;
        let dvx_dz = (vel_front.x - vel_back.x) * 0.5 * u.rdx;
        let dvz_dx = (vel_right.z - vel_left.z) * 0.5 * u.rdx;
        let dvy_dx = (vel_right.y - vel_left.y) * 0.5 * u.rdx;
        let dvx_dy = (vel_top.x - vel_bottom.x) * 0.5 * u.rdx;

        let omega_x = dvz_dy - dvy_dz;
        let omega_y = dvx_dz - dvz_dx;
        let omega_z = dvy_dx - dvx_dy;

        let vorticity_magnitude = length(vec3(omega_x, omega_y, omega_z));
        vorticity[index] = vorticity_magnitude;
    }`);

    computeShaders.vorticityConfinment = new ComputeShader("vorticityConfinment", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read_write> velocity : array<vec4f>;
    @group(0) @binding(2) var<storage, read> vorticity : array<f32>;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let index = to_index(global_id);
        let id = vec3<i32>(global_id);

        let L  = vorticity[clamp_to_edge(id - vec3i(1, 0, 0))];
        let R  = vorticity[clamp_to_edge(id + vec3i(1, 0, 0))];
        let B  = vorticity[clamp_to_edge(id - vec3i(0, 1, 0))];
        let T  = vorticity[clamp_to_edge(id + vec3i(0, 1, 0))];
        let F  = vorticity[clamp_to_edge(id - vec3i(0, 0, 1))];
        let Ba = vorticity[clamp_to_edge(id + vec3i(0, 0, 1))];
        let C = vorticity[index];

        var force = vec3f(abs(R) - abs(L), abs(T) - abs(B), abs(F) - abs(Ba));
        const epsilon = 1e-10;
        force = force / sqrt(dot(force, force) + epsilon);
        force *= u.vorticity * u.dt * C * u.dx;

        velocity[index] += vec4f(force, 0);
    }`);
}
