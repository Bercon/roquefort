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

    const MAX_GRID_SIZE = 384;
    // Cap at 4096 to get max 256MB viewport buffers, 2x 1GB seems excessive which you get at 8k or 8192x8192
    let MAX_TEX_DIM = 4096;
    const MAX_BUFFER_SIZE = Math.max(
        MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE * 4 * 4,
        MAX_TEX_DIM * MAX_TEX_DIM * 4 * 4
    );
    const device = await adapter?.requestDevice({
        requiredFeatures: presentationFormat === 'bgra8unorm' ? ['bgra8unorm-storage'] : [],
        requiredLimits: {
            maxStorageBufferBindingSize: MAX_BUFFER_SIZE,
            maxBufferSize: MAX_BUFFER_SIZE
        },
    });
    if (!device) {
        throw new Error("<p>Couldn't get WebGPU device</p>");
    }

    MAX_TEX_DIM = Math.min(MAX_TEX_DIM, device.limits.maxTextureDimension2D);

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
        canvasiX: "i32",
        canvasiY: "i32",
        camPosX: "f32",
        camPosY: "f32",
        camPosZ: "f32",
        stepLength: "f32",
        enclosed: "u32",
        emitterR: "f32",
        emitterG: "f32",
        emitterB: "f32",
        blackbodyBrightness: "f32",
        brushTemperatureAmount: "f32",
        orthoBlend: "f32",
        glowRadius: "i32",
        glowStrength: "f32",
        glowGamma: "f32",
    });

    const lightProps = {
        lightPrev: "f32",
        lightDir: "i32",
        lightR: "f32",
        lightG: "f32",
        lightB: "f32",
    }

    const pressureUniform = {
        i: "u32",
        offsetCurrent: "u32",
        offsetNext: "u32",
        current: "u32",
        next: "u32",
        dx: "f32"
    };

    const lightingPrimaryUni = new UniformBuffer(device, "light", "Light", lightProps);
    const lightingFillUni = new UniformBuffer(device, "light", "Light", lightProps);
    const uniformEven = new UniformBuffer(device, "iter", "Iter", pressureUniform, { i: 0, offsetCurrent: 0, offsetNext: 0, current: 0, next: 0, dx: 0 });
    const uniformOdd = new UniformBuffer(device, "iter", "Iter", pressureUniform, { i: 1, offsetCurrent: 0, offsetNext: 0, current: 0, next: 0, dx: 0 });

    const uniformBuffer = uniforms.gpuBuffer;

    function createBuffers(config) {
        // Only support multigrid levels where grid is still divisible by 4
        let s = config.gridSize;
        let level = 0;
        while (s >= 4 && (s / 2) % 4 == 0) {
            s /= 2;
            level++;
        }
        config.multigridMaxLevel = level;
        // console.log(`Max multigrid level ${config.multigridMaxLevel}`);

        function multigridOffset(level, baseSize) {
            let offset = 0;
            let size = baseSize;
            for (let i = 0; i < level; i++) {
                offset = offset + size * size * size;
                size = size / 2;
            }
            return offset;
        }

        let multigrid = [];
        let offsetCurrent = 0;
        let offsetNext = 0;
        for (let level = 0; level <= config.multigridMaxLevel; level++) {
            offsetCurrent = multigridOffset(level, config.gridSize);
            offsetNext = multigridOffset(level + 1, config.gridSize);
            size = config.gridSize / Math.pow(2, level);
            sizeNext = config.gridSize / Math.pow(2, level + 1);
            let rdx = size * config.simulation_scale;
            let dx = 1 / rdx;
            const shared = {
                offsetCurrent,
                offsetNext,
                current: size,
                next: sizeNext,
                current: size,
                dx: dx
            }
            multigrid.push(
                {
                    even: new UniformBuffer(device, "iter", "Iter", pressureUniform, { i: 0, ...shared }).gpuBuffer,
                    odd: new UniformBuffer(device, "iter", "Iter", pressureUniform, { i: 1, ...shared }).gpuBuffer,
                }
            );
            if (!Number.isInteger(shared.current)) throw new Error(`Not integer, multigrid cannot work: ${shared.current}`);
            // console.log("multigrid", level, "current", shared.current, "next", shared.next, "offsets", offsetCurrent, offsetNext);
        }
        return {
            // Notes: 2 to store multigrid levels, Jacobi solver uses previous pressure as starting point so can't use "velocity.write" as temp buffer
            pressure: createStorageBuffer(device, "pressure", 2, config.gridX, config.gridY, config.gridZ),
            // Note: Velocity is XYZ and Fuel amount, none of which are required for rendering
            velocity: new DoubleStorageBuffer(device, "velocity", 4, config.gridX, config.gridY, config.gridZ),
            temperature: new DoubleStorageBuffer(device, "temperature", 1, config.gridX, config.gridY, config.gridZ),
            // Note: Smoke is RGB and Amount
            smoke: new DoubleStorageBuffer(device, "smoke", 4, config.gridX, config.gridY, config.gridZ),
            original: createStorageBuffer(device, "original", 4, MAX_TEX_DIM, MAX_TEX_DIM, 1),
            blurredA: createStorageBuffer(device, "blurredA", 4, MAX_TEX_DIM, MAX_TEX_DIM, 1),
            blurredB: createStorageBuffer(device, "blurredB", 4, MAX_TEX_DIM, MAX_TEX_DIM, 1),
            multigrid: multigrid
        }
    }

    function destroyBuffers(buffers) {
        if (!buffers) return;
        for (const [key, value] of Object.entries(buffers)) {
            if (value.destroy) value.destroy();
        }
    }

    let bufferResolution = 0;
    let buffers = null;

    const computeShaders = {};
    const source = {};
    for (let shader of [
        'buildSourceCommon',
        'buildShadersBlur',
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
            entry.target.width = Math.max(1, Math.min(width, MAX_TEX_DIM));
            entry.target.height = Math.max(1, Math.min(height, MAX_TEX_DIM));
        }
    });
    resizeObserver.observe(canvas);

    const TIMING_INTERVAL = 100;
    const javascriptTimer = new RollingAverage(TIMING_INTERVAL);
    let timer = new Timer();
    async function render() {
        try {
            let startTime = performance.now();
            let [timeFullFrame, deltaFullFrame] = timer.getTimeAndDelta(config.speed);
            let renderOnThisFrame = false;

            let delta = deltaFullFrame / config.subframes;
            let time = timeFullFrame; // TODO: Not correct for subframes

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
                t: time, // TODO: Incorrect for subframes
                brushSmokeAmount: config.brushSmokeAmount,
                brushSize: config.brushSize,
                brushFuelAmount: config.brushFuelAmount * Math.random(), // TODO: Should take delta in account
                smokeDecay: config.smokeDecay,
                brushVelocityAmount: config.brushVelocityAmount,
                velocityDecay: config.velocityDecay,
                canvasX: canvas.width,
                canvasY: canvas.height,
                canvasiX: canvas.width,
                canvasiY: canvas.height,
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
                enclosed: config.enclosed ? 1 : 0,
                mouseStartX: 0,
                mouseStartY: 0,
                mouseEndX: 0,
                mouseEndY: 0,
                emitterR: emitterRgb[0],
                emitterG: emitterRgb[1],
                emitterB: emitterRgb[2],
                blackbodyBrightness: config.blackbodyBrightness,
                brushTemperatureAmount: config.brushTemperatureAmount,
                orthoBlend: config.orthoBlend,
                glowRadius: Math.round(config.glowRadius * canvas.width / 100) / 2, // / 2 because we run box blur twice
                glowStrength: config.glowStrength,
                glowGamma: config.glowGamma,
            }
            // console.log("glow radius", Math.round(config.glowRadius * canvas.width / 100));

            let mouseUpdate = false;
            if (!viewControls.empty()) {
                const events = viewControls.getMouseLine();
                // TODO: Not correct with subframes
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
                lightDir: 5,
                lightR: config.primaryLightColor[0] * config.primaryLightBrightness,
                lightG: config.primaryLightColor[1] * config.primaryLightBrightness,
                lightB: config.primaryLightColor[2] * config.primaryLightBrightness,
            });
            lightingFillUni.update(device, {
                lightPrev: 1., // Multiplier for previous light value
                lightDir: 1,
                lightR: config.fillLightColor[0] * config.fillLightBrightness,
                lightG: config.fillLightColor[1] * config.fillLightBrightness,
                lightB: config.fillLightColor[2] * config.fillLightBrightness,
            });

            // await device.queue.onSubmittedWorkDone();
            const encoder = device.createCommandEncoder();
            const passEncoder = encoder.beginComputePass()

            for (let subframe = 0; subframe < config.subframes; subframe++) {
                renderOnThisFrame = subframe === config.subframes - 1;

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
                } else if (config.scene == "Rotating fire emitter") {
                    computeShaders.rotatingFireEmitter.computePass(device, passEncoder, [
                        { buffer: uniformBuffer },
                        { buffer: buffers.velocity.read }, // Update in place
                        { buffer: buffers.temperature.read }, // Update in place
                    ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
                } else if (config.scene == "5 spheres (uses brush settings)") {
                    computeShaders.fiveSpheres.computePass(device, passEncoder, [
                        { buffer: uniformBuffer },
                        { buffer: buffers.velocity.read }, // Update in place
                        { buffer: buffers.smoke.read }, // Update in place
                        { buffer: buffers.temperature.read }, // Update in place
                    ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
                } else if (config.scene == "Vortex") {
                    computeShaders.vortex.computePass(device, passEncoder, [
                        { buffer: uniformBuffer },
                        { buffer: buffers.velocity.read }, // Update in place
                        { buffer: buffers.smoke.read }, // Update in place
                        { buffer: buffers.temperature.read }, // Update in place
                    ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
                } else if (config.scene == "Intro") {
                    computeShaders.intro.computePass(device, passEncoder, [
                        { buffer: uniformBuffer },
                        { buffer: buffers.velocity.read }, // Update in place
                        { buffer: buffers.smoke.read }, // Update in place
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
                    { buffer: buffers.pressure },
                ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
                buffers.velocity.swap();
                buffers.smoke.swap();
                buffers.temperature.swap();

                computeShaders.divergence.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: buffers.velocity.read },
                    { buffer: buffers.smoke.write }, // Reuse buffer
                ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);

                if (config.useRedBlackJacobi) {
                    for (let i = 0; i < config.pressure_iterations; i++) {
                        computeShaders.jacobiRedBlack.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: buffers.pressure }, // Update in place
                            { buffer: buffers.smoke.write }, // Reuse buffer
                            { buffer: (i % 2 == 0 ? uniformEven.gpuBuffer : uniformOdd.gpuBuffer) },
                        ], config.gridX / 4, config.gridY / 4, config.gridZ / 8); // half on z-axis
                    }
                } else {
                    function solveLevel(level, iter) {
                        let dispatch = config.gridSize / 4 / Math.pow(2, level);
                        for (let i = 0; i < iter; i++) {
                            let solver = computeShaders.solve;
                            if (dispatch == 1) {
                                solver = computeShaders.solveGroupSize442;
                            }
                            solver.computePass(device, passEncoder, [
                                // { buffer: uniformBuffer },
                                { buffer: buffers.pressure },
                                { buffer: buffers.smoke.write }, // Reuse buffer
                                { buffer: (i % 2 == 0 ? buffers.multigrid[level].even :buffers.multigrid[level].odd) },
                            ], dispatch, dispatch, Math.max(1, dispatch / 2));
                        }
                    }
                    function restrict(level, buffer) {
                        let dispatch = config.gridX / 4 / Math.pow(2, level + 1);
                        computeShaders.restrict.computePass(device, passEncoder, [
                            { buffer: buffer },
                            { buffer: buffers.multigrid[level].even },
                        ], dispatch, dispatch, dispatch);
                    }
                    function relax(level, buffer) {
                        let dispatch = config.gridX / 4 / Math.pow(2, level + 1);
                        computeShaders.relax.computePass(device, passEncoder, [
                            // { buffer: uniformBuffer },
                            { buffer: buffer },
                            { buffer: buffers.multigrid[level].even },
                        ], dispatch, dispatch, dispatch);
                    }
                    // "V"-cycle
                    for (let level = 0; level < config.multigridMaxLevel; level++) {
                        // TODO: Could solve couple times here, but unlikely to mean much?
                        restrict(level, buffers.smoke.write); // Reuse buffer
                    }
                    for (let level = config.multigridMaxLevel; level >= 0; level--) {
                        solveLevel(level, config.pressure_iterations);
                        if (level >= 1) relax(level - 1, buffers.pressure);
                    }
                }

                computeShaders.gradientSubtract.computePass(device, passEncoder, [
                    { buffer: uniformBuffer },
                    { buffer: buffers.pressure },
                    { buffer: buffers.velocity.read }, // read/write
                ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);

                if (config.enable_vorticity) {
                    computeShaders.vorticity.computePass(device, passEncoder, [
                        { buffer: uniformBuffer },
                        { buffer: buffers.velocity.read },
                        { buffer: buffers.temperature.write }, // Reuse temperature buffer
                    ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);

                    computeShaders.vorticityConfinment.computePass(device, passEncoder, [
                        { buffer: uniformBuffer },
                        { buffer: buffers.velocity.read }, // read/write
                        { buffer: buffers.temperature.write }, // Reuse temperature buffer
                    ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);
                }

                if (renderOnThisFrame) {
                    // console.log("Render");
                    if (config.renderMode == "Normal") {
                        computeShaders.lighting.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: lightingPrimaryUni.gpuBuffer },
                            { buffer: buffers.smoke.read },
                            { buffer: buffers.smoke.write }, // Reuse smoke buffer
                        ], config.gridX / 8, config.gridY / 8);

                        computeShaders.lighting.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: lightingFillUni.gpuBuffer },
                            { buffer: buffers.smoke.read },
                            { buffer: buffers.smoke.write }, // Reuse smoke buffer
                        ], config.gridX / 8, config.gridY / 8);

                        computeShaders.bakeLighting.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: buffers.smoke.read },
                            { buffer: buffers.temperature.read },
                            { buffer: buffers.smoke.write }, // Reuse smoke buffer
                        ], config.gridX / 4, config.gridY / 4, config.gridZ / 4);

                        computeShaders.renderDefault.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: buffers.smoke.write }, // Reuse smoke buffer
                            { buffer: buffers.original },
                            // context.getCurrentTexture().createView(),
                        ], Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));

                        // 2x box blur
                        computeShaders.boxBlurHorizontal.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: buffers.original },
                            { buffer: buffers.blurredA },
                        ], Math.ceil(canvas.height / 64));

                        computeShaders.boxBlurVertical.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: buffers.blurredA },
                            { buffer: buffers.blurredB },
                        ], Math.ceil(canvas.width / 64));

                        computeShaders.boxBlurHorizontal.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: buffers.blurredB },
                            { buffer: buffers.blurredA },
                        ], Math.ceil(canvas.height / 64));

                        computeShaders.boxBlurVertical.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: buffers.blurredA },
                            { buffer: buffers.blurredB },
                        ], Math.ceil(canvas.width / 64));

                        computeShaders.composite.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: buffers.original },
                            { buffer: buffers.blurredB },
                            context.getCurrentTexture().createView(),
                        ], Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));

                    } else if  (config.renderMode == "Fuel shaded") {
                        // LIGHTING

                        computeShaders.lightingFuel.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: lightingPrimaryUni.gpuBuffer },
                            { buffer: buffers.velocity.read },
                            { buffer: buffers.smoke.write }, // Reuse smoke buffer
                        ], config.gridX / 8, config.gridY / 8);

                        computeShaders.lightingFuel.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            { buffer: lightingFillUni.gpuBuffer },
                            { buffer: buffers.velocity.read },
                            { buffer: buffers.smoke.write }, // Reuse smoke buffer
                        ], config.gridX / 8, config.gridY / 8);

                        computeShaders.renderFuel.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            context.getCurrentTexture().createView(),
                            { buffer: buffers.velocity.read },
                            { buffer: buffers.smoke.write }, // Reuse smoke buffer
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
                            { buffer: buffers.pressure },
                            { buffer: buffers.multigrid[Math.min(config.multigridMaxLevel, config.debugMultigridLevel)].even },
                        ], Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));

                    } else if (config.renderMode == "Divergence") {

                        computeShaders.renderDivergence.computePass(device, passEncoder, [
                            { buffer: uniformBuffer },
                            context.getCurrentTexture().createView(),
                            { buffer: buffers.smoke.write }, // Reuse buffer
                            { buffer: buffers.multigrid[Math.min(config.multigridMaxLevel, config.debugMultigridLevel)].even },
                        ], Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));

                    } else {
                        throw new Error(`<p>Unknown render mode:</p><p>${config.renderMode}</p>`);
                    }


                }
            }

            passEncoder.end();
            const commandBuffer = encoder.finish();
            device.queue.submit([commandBuffer]);
            // await device.queue.onSubmittedWorkDone();
            // console.log(time)

            const webgpuError = await device.popErrorScope();
            if (webgpuError) {
                displayError(`<p>WebGPU error:</p><p>${webgpuError.message}</p>`);
                throw new Error(`<p>WebGPU error:</p><p>${webgpuError.message}</p>`);
            }
            device.pushErrorScope('validation');

            javascriptTimer.addSample(performance.now() - startTime);
            configuration.update("javascriptTime", javascriptTimer.get().toFixed(1));
            requestAnimationFrame(render)
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