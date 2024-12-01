async function main() {
    const configuration = new Configuration();

    createUI(configuration.config, configuration);
    configuration.loadConfig();

    const config = configuration.config;

    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
        throw new Error("<p>Your browser doesn't support WebGPU!</p><p>Try with Edge, Chrome or Chromium based browsers. Firefox nightly and Safari beta may also work.</p>");
    }

    // bgra8unorm as a storage texture is an optional feature so
    // if it's supported then we don't care if presentationFormat is
    // bgra8unorm or rgba8unorm but if the feature does not exist
    // then we must use rgba8unorm
    const presentationFormat = adapter.features.has('bgra8unorm-storage')
        ? navigator.gpu.getPreferredCanvasFormat()
        : 'rgba8unorm';

    const MAX_RESOLUTION = 384;
    const device = await adapter?.requestDevice({
        requiredFeatures: presentationFormat === 'bgra8unorm' ? ['bgra8unorm-storage'] : [],
        requiredLimits: {
            maxStorageBufferBindingSize: MAX_RESOLUTION * MAX_RESOLUTION * MAX_RESOLUTION * 4 * 4,
            maxBufferSize: MAX_RESOLUTION * MAX_RESOLUTION * MAX_RESOLUTION * 4 * 4
        },
    });
    if (!device) {
        throw new Error("<p>Couldn't get WebGPU device</p>");
    }

    // Get a WebGPU context from the canvas and configure it
    const canvas = document.querySelector('canvas');

    const context = canvas.getContext('webgpu');
    context.configure({
        device,
        format: presentationFormat,
        // This is what's required to be able to write to a texture from a compute shader
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
    });

    device.pushErrorScope('validation');

    const uniforms = new UniformBuffer(device, "uni", "U", {
        x: "f32",
        y: "f32",
        z: "f32",
        ux: "u32",
        uy: "u32",
        uz: "u32",
        dx: "f32",
        rdx: "f32",
        pressureDecay: "f32",
        vorticity: "f32",
        dt: "f32",
        t: "f32",
        mouseStartX: "f32",
        mouseStartY: "f32",
        mouseEndX: "f32",
        mouseEndY: "f32",
        brushSmokeAmount: "f32",
        brushFuelAmount: "f32",
        smokeDecay: "f32",
        temperatureDecay: "f32",
        ignitionTemperature: "f32",
        burnRate: "f32",
        burnHeatEmit: "f32",
        burnSmokeEmit: "f32",
        smokeR: "f32",
        smokeG: "f32",
        smokeB: "f32",
        boyancy: "f32",
        brushVelocityAmount: "f32",
        brushSize: "f32",
        velocityDecay: "f32",
        canvasX: "f32",
        canvasY: "f32",
        camPosX: "f32",
        camPosY: "f32",
        camPosZ: "f32",
        stepLength: "f32",
        combustionExpansion: "f32",
        enclosed: "u32",
        emitterR: "f32",
        emitterG: "f32",
        emitterB: "f32",
        blackbodyBrightness: "f32",
        brushTemperatureAmount: "f32",
    });

    const lightProps = {
        lightPrev: "f32",
        lightDir: "f32",
        lightR: "f32",
        lightG: "f32",
        lightB: "f32",
    }

    const lightingPrimaryUni = new UniformBuffer(device, "light", "Light", lightProps);
    const lightingFillUni = new UniformBuffer(device, "light", "Light", lightProps);
    const uniformEven = new UniformBuffer(device, "iter", "Iter", {  i: "u32" }, { i: 0 });
    const uniformOdd = new UniformBuffer(device, "iter", "Iter", {  i: "u32" }, { i: 1 });

    const uniformBuffer = uniforms.gpuBuffer;

    function createBuffers(config) {
        return {
            vorticity: createStorageBuffer(device, "vorticity", 1, config.gridX, config.gridY, config.gridZ),
            divergence: createStorageBuffer(device, "divergence", 1, config.gridX, config.gridY, config.gridZ),
            pressure: new DoubleStorageBuffer(device, "pressure", 1, config.gridX, config.gridY, config.gridZ),
            lighting: createStorageBuffer(device, "lighting", 4, config.gridX, config.gridY, config.gridZ),
            // Note: Velocity is XYZ and Fuel amount, none of which are required for rendering
            velocity: new DoubleStorageBuffer(device, "velocity", 4, config.gridX, config.gridY, config.gridZ),
            temperature: new DoubleStorageBuffer(device, "temperature", 1, config.gridX, config.gridY, config.gridZ),
            // Note: Smoke is RGB and Amount
            smoke: new DoubleStorageBuffer(device, "smoke", 4, config.gridX, config.gridY, config.gridZ),
        }
    } // (1 + 1 + 2 + 4 + 8 + 2 + 8) * 4

    function destroyBuffers(buffers) {
        if (!buffers) return;
        for (const [key, value] of Object.entries(buffers)) {
            value.destroy();
        }
    }

    let bufferResolution = 0;
    let buffers = null;
    let timer = new Timer();

    const computeShaders = {};
    const source = {};
    for (let shader of [
        'buildSourceCommon',
        'buildShadersLighting',
        'buildShadersRender',
        'buildShadersAdvect',
        'buildShadersGradientSubtract',
        'buildShadersPressure',
        'buildShadersVorticity',
        'buildShadersDivergence',
        'buildShadersEmitters'
    ]) {
        if (window[shader] === undefined) {
            throw new Error(`&quot;${shader}&#34; doesn't exist`);
        }
        window[shader]({
            device,
            uniformStruct: uniforms.struct,
            lightUniformStruct: lightingPrimaryUni.struct,
            presentationFormat,
            computeShaders,
            iterUniformStruct: uniformEven.struct,
            source});
    }

    const viewControls = new ViewControls(canvas, configuration);

    const resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
            const width = entry.contentBoxSize[0].inlineSize;
            const height = entry.contentBoxSize[0].blockSize;
            entry.target.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
            entry.target.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
        }
    });
    resizeObserver.observe(canvas);

    const TIMING_INTERVAL = 100;
    const javascriptTimer = new RollingAverage(TIMING_INTERVAL);

    async function render() {
        try {
            let [time, delta] = timer.getTimeAndDelta(config.speed);
            let startTime = performance.now();
            let emitterRgb = hsvToRgb([time * .1 % 1, .9, .7]);
            if (bufferResolution != config.gridSize) {
                config.gridX = config.gridSize;
                config.gridY = config.gridSize;
                config.gridZ = config.gridSize;

                config.rdx = config.gridSize * config.simulation_scale;
                config.dx = 1 / config.rdx;

                bufferResolution = config.gridSize;
                destroyBuffers(buffers);
                buffers = null;
                buffers = createBuffers(config);
            }

            if (config.cameraRotationEnabled) {
                configuration.addPhi(config.cameraRotationSpeed * delta);
            }

            const [camPosX, camPosY, camPosZ] = sphericalToCartesian([config.camPosRadius, config.camPosTheta, config.camPosPhi]);

            let updatedProps = {
                x: config.gridX,
                y: config.gridY,
                z: config.gridZ,
                ux: config.gridX,
                uy: config.gridY,
                uz: config.gridZ,
                dx: config.dx,
                rdx: config.rdx,
                pressureDecay: config.pressureDecay,
                vorticity: config.vorticity,
                dt: delta,
                t: time,
                brushSmokeAmount: config.brushSmokeAmount,
                brushSize: config.brushSize,
                brushFuelAmount: config.brushFuelAmount * Math.random(),
                smokeDecay: config.smokeDecay,
                brushVelocityAmount: config.brushVelocityAmount,
                velocityDecay: config.velocityDecay,
                canvasX: canvas.width,
                canvasY: canvas.height,
                camPosX,
                camPosY,
                camPosZ,
                temperatureDecay: config.temperatureDecay,
                ignitionTemperature: config.ignitionTemperature,
                burnRate: config.burnRate,
                burnHeatEmit: config.burnHeatEmit,
                burnSmokeEmit: config.burnSmokeEmit,
                smokeR: config.smokeRgb[0],
                smokeG: config.smokeRgb[1],
                smokeB: config.smokeRgb[2],
                boyancy: config.boyancy,
                stepLength: config.stepLength,
                combustionExpansion: config.combustionExpansion,
                enclosed: config.enclosed ? 1 : 0,
                mouseStartX: 0,
                mouseStartY: 0,
                mouseEndX: 0,
                mouseEndY: 0,
                emitterR: emitterRgb[0],
                emitterG: emitterRgb[1],
                emitterB: emitterRgb[2],
                blackbodyBrightness: config.blackbodyBrightness,
                brushTemperatureAmount: config.brushTemperatureAmount
            }

            let mouseUpdate = false;
            if (!viewControls.empty()) {
                const events = viewControls.getMouseLine();
                updatedProps = {
                    ...updatedProps,
                    mouseStartX: events[0].x / canvas.width,
                    mouseStartY: events[0].y / canvas.height,
                    mouseEndX: events[events.length - 1].x / canvas.width,
                    mouseEndY: events[events.length - 1].y / canvas.height,
                }
                mouseUpdate = true;
            }

            uniforms.update(device, updatedProps);

            lightingPrimaryUni.update(device, {
                lightPrev: 0, // Multiplier for previous light value
                lightDir: 2., // left, right, top, bottom, forward, backward
                lightR: config.primaryLightColor[0] * config.primaryLightBrightness,
                lightG: config.primaryLightColor[1] * config.primaryLightBrightness,
                lightB: config.primaryLightColor[2] * config.primaryLightBrightness,
            });
            lightingFillUni.update(device, {
                lightPrev: 1., // Multiplier for previous light value
                lightDir: 0., // left, right, top, bottom, forward, backward
                lightR: config.fillLightColor[0] * config.fillLightBrightness,
                lightG: config.fillLightColor[1] * config.fillLightBrightness,
                lightB: config.fillLightColor[2] * config.fillLightBrightness,
            });

            const encoder = device.createCommandEncoder();
            const passEncoder = encoder.beginComputePass()

            if (mouseUpdate) {
                computeShaders.updateMouse.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: buffers.velocity.read }, // Update in place
                    { buffer: buffers.smoke.read }, // Update in place
                    { buffer: buffers.temperature.read }, // Update in place
                ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
            }

            if (config.scene == "Rotating smoke emitter") {
                computeShaders.rotatingSmokeEmitter.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: buffers.velocity.read }, // Update in place
                    { buffer: buffers.smoke.read }, // Update in place
                ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
            } else  if (config.scene == "Rotating fire emitter") {
                computeShaders.rotatingFireEmitter.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: buffers.velocity.read }, // Update in place
                    { buffer: buffers.temperature.read }, // Update in place
                ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
            }

            computeShaders.advect.computePass(device, passEncoder, [
                { buffer: uniformBuffer },
                { buffer: buffers.velocity.read },
                { buffer: buffers.smoke.read },
                { buffer: buffers.temperature.read },
                { buffer: buffers.velocity.write },
                { buffer: buffers.smoke.write },
                { buffer: buffers.temperature.write },
                { buffer: buffers.divergence },
            ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
            buffers.velocity.swap();
            buffers.smoke.swap();
            buffers.temperature.swap();

            if (config.pressureDecay != 0.0) {
                computeShaders.pressureClear.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: buffers.pressure.read }, // Update in place
                ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
            }

            computeShaders.divergence.computePass(device, passEncoder, [
                { buffer: uniformBuffer },
                { buffer: buffers.velocity.read },
                { buffer: buffers.divergence },
            ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);

            if (config.useRedBlackJacobi) {
                // Double amount of iterations, but half amount of processing each time
                for (let i = 0; i < config.pressure_iterations * 2; i++) {
                    computeShaders.jacobiRedBlack.computePass(device, passEncoder, [
                        { buffer: uniformBuffer },
                        { buffer: buffers.pressure.read }, // Update in place
                        { buffer: buffers.divergence },
                        { buffer: (i % 2 == 0 ? uniformEven.gpuBuffer : uniformOdd.gpuBuffer) },
                    ], config.gridX / 4, config.gridY / 4, config.gridZ / 8); // half on z-axis
                }
            } else {
                for (let i = 0; i < config.pressure_iterations; i++) {
                    computeShaders.jacobi.computePass(device, passEncoder, [
                        { buffer: uniformBuffer },
                        { buffer: buffers.pressure.read },
                        { buffer: buffers.divergence },
                        { buffer: buffers.pressure.write },
                    ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
                    buffers.pressure.swap();
                }
            }

            computeShaders.gradientSubtract.computePass(device, passEncoder, [
                { buffer: uniformBuffer },
                { buffer: buffers.pressure.read },
                { buffer: buffers.velocity.read }, // read/write
            ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);

            if (config.enable_vorticity) {
                computeShaders.vorticity.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: buffers.velocity.read },
                    { buffer: buffers.vorticity },
                ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);

                computeShaders.vorticityConfinment.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: buffers.velocity.read }, // read/write
                    { buffer: buffers.vorticity },
                ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
            }

            if (config.renderMode == "Normal") {
                // LIGHTING

                computeShaders.lighting.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: lightingPrimaryUni.gpuBuffer },
                    { buffer: buffers.smoke.read },
                    { buffer: buffers.lighting },
                ], config.gridX / 8, config.gridY / 8);

                computeShaders.lighting.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: lightingFillUni.gpuBuffer },
                    { buffer: buffers.smoke.read },
                    { buffer: buffers.lighting },
                ], config.gridX / 8, config.gridY / 8);

                computeShaders.bakeLighting.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: buffers.smoke.read },
                    { buffer: buffers.temperature.read },
                    { buffer: buffers.lighting },
                ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);

                // RENDERING

                computeShaders.renderDefault.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    context.getCurrentTexture().createView(),
                    // { buffer: buffers.smoke.read },
                    { buffer: buffers.lighting },
                ], Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));

            } else if  (config.renderMode == "Fuel shaded") {
                // LIGHTING

                computeShaders.lightingFuel.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: lightingPrimaryUni.gpuBuffer },
                    { buffer: buffers.velocity.read },
                    { buffer: buffers.lighting },
                ], config.gridX / 8, config.gridY / 8);

                computeShaders.lightingFuel.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: lightingFillUni.gpuBuffer },
                    { buffer: buffers.velocity.read },
                    { buffer: buffers.lighting },
                ], config.gridX / 8, config.gridY / 8);

                // RENDERING

                computeShaders.renderFuel.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    context.getCurrentTexture().createView(),
                    { buffer: buffers.velocity.read },
                    { buffer: buffers.lighting },
                ], Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));


            } else if (config.renderMode == "Fuel temperature") {

                computeShaders.renderFuelTemperature.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    context.getCurrentTexture().createView(),
                    { buffer: buffers.velocity.read },
                    { buffer: buffers.temperature.read },
                ], Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));

            } else if (config.renderMode == "Pressure") {

                computeShaders.renderPressure.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    context.getCurrentTexture().createView(),
                    { buffer: buffers.pressure.read },
                ], Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));

            } else if (config.renderMode == "Divergence") {

                computeShaders.renderDivergence.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    context.getCurrentTexture().createView(),
                    { buffer: buffers.divergence },
                ], Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));

            } else {
                throw new Error(`<p>Unknown render mode:</p><p>${config.renderMode}</p>`);
            }

            passEncoder.end();
            const commandBuffer = encoder.finish();
            device.queue.submit([commandBuffer]);
            // console.log(time)

            const webgpuError = await device.popErrorScope();
            if (webgpuError) {
                displayError(`<p>WebGPU error:</p><p>${webgpuError.message}</p>`);
                throw new Error(`<p>WebGPU error:</p><p>${webgpuError.message}</p>`);
            } else {
                requestAnimationFrame(render)
            }
            device.pushErrorScope('validation');

            javascriptTimer.addSample(performance.now() - startTime);
            configuration.update("javascriptTime", javascriptTimer.get().toFixed(1));
        } catch (e) {
            displayError(e.message);
            throw e;
        }
    }

    render();
}

(async () => {
    try {
        await main();
    } catch (e) {
        displayError(e.message);
        throw e;
    }
})();