function displayError(msg) {
    const canvas = document.getElementById("C");
    const errorBox = document.getElementById("errorBox");
    if (!(canvas && errorBox)) return;
    canvas.style = "display: none";
    errorBox.style.display = "flex";
    const warningSign = `
        <div style="width: 40px; height: 40px; margin: 0 auto;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 2c-.48 0-.94.26-1.18.7L1.64 20.3c-.24.44-.22.98.06 1.4.28.42.75.7 1.26.7h18.08c.5 0 .98-.27 1.26-.7.28-.42.3-.96.06-1.4L13.18 2.7A1.37 1.37 0 0 0 12 2zM12 16.5a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3zM12 7c.55 0 1 .45 1 1v5a1 1 0 0 1-2 0V8c0-.55.45-1 1-1z"/>
            </svg>
        </div>
    `;
    console.log(msg);
    errorBox.innerHTML = warningSign + msg;
}

function padder(element) {
    const padder = document.createElement("div");
    padder.appendChild(element);
    padder.className = "padder";
    return padder;
}

function createSlider({parent, configuration, key, labelText, minValue, maxValue, defaultValue, step=0.01}) {
    const container = document.createElement("div");
    container.className = "config-row";

    const numberInput = document.createElement("input");
    numberInput.id = uniqueId()
    numberInput.type = "number";
    numberInput.min = minValue;
    numberInput.max = maxValue;
    numberInput.step = step;

    const rangeInput = document.createElement("input");
    rangeInput.type = "range";
    rangeInput.min = minValue;
    rangeInput.max = maxValue;
    rangeInput.step = step;

    const label = document.createElement("label");
    label.htmlFor = numberInput.id
    label.textContent = labelText;

    container.appendChild(label);
    container.appendChild(padder(rangeInput));
    container.appendChild(padder(numberInput));
    parent.appendChild(container);

    configuration.addListener(key, value => {
        numberInput.value = value;
        rangeInput.value = value;
        const p = (rangeInput.value - rangeInput.min) / (rangeInput.max - rangeInput.min) * 100;
        rangeInput.style.setProperty('--slider-percentage', `${p}%`);
    });

    numberInput.addEventListener("change", function() {
        configuration.updateNumber(key, numberInput.value);
    });

    rangeInput.addEventListener("input", function(e) {
        configuration.updateNumber(key, rangeInput.value);
    });
    configuration.addUserConfigNumber(key, defaultValue);
}

function createCheckbox({parent, configuration, key, labelText, defaultValue}) {
    const container = document.createElement("div");
    container.className = "config-row";

    const checkboxInput = document.createElement("input");
    checkboxInput.id = uniqueId()
    checkboxInput.type = "checkbox";

    const label = document.createElement("label");
    label.htmlFor = checkboxInput.id;
    label.textContent = labelText;

    container.appendChild(label);
    container.appendChild(padder(checkboxInput));
    parent.appendChild(container);

    configuration.addListener(key, value => {
        checkboxInput.checked = value;
    });
    checkboxInput.addEventListener("change", function() {
        configuration.update(key, checkboxInput.checked);
    });
    configuration.addUserConfig(key, defaultValue);
}

function createSelect({parent, configuration, key, labelText, options, defaultValue}) {
    const values = options.map(x => x.value)
    if (values.findIndex((e) => e == defaultValue) == -1) {
        throw new Error("Default value \"" + defaultValue + "\" not found in options: "
            + values.toString());
    }

    const container = document.createElement("div");
    container.className = "config-row";

    const selectInput = document.createElement("select");
    selectInput.id = uniqueId()

    for (let option of options) {
        var optionElem = document.createElement("option");
        optionElem.value = option.value;
        optionElem.text = option.label;
        selectInput.appendChild(optionElem);
    }

    const label = document.createElement("label");
    label.htmlFor = selectInput.id
    label.textContent = labelText;

    container.appendChild(label);
    container.appendChild(padder(selectInput));
    parent.appendChild(container);

    configuration.addListener(key, value => {
        selectInput.value = value;
    });
    selectInput.addEventListener("change", function() {
        configuration.update(key, selectInput.value);

    });
    configuration.addUserConfig(key, defaultValue);
}

function createPreset({parent, configuration, labelText, presets}) {
    const container = document.createElement("div");
    container.className = "config-row";

    const selectInput = document.createElement("select");
    selectInput.id = uniqueId()

    for (let i = 0; i < presets.length; i++) {
        let p = presets[i];
        var optionElem = document.createElement("option");
        optionElem.value = p.label;
        optionElem.text = p.label;
        if (i === 0) optionElem.disabled = true;
        selectInput.appendChild(optionElem);
    }

    const label = document.createElement("label");
    label.htmlFor = selectInput.id
    label.textContent = labelText;

    container.appendChild(label);
    container.appendChild(padder(selectInput));
    parent.appendChild(container);

    let confPresets = new ConfigurationPresets(configuration, presets,
        preset => { selectInput.value = preset.label; }
    );
    selectInput.addEventListener("change", function() {
        confPresets.apply(selectInput.value);
    });
}

function createColorPicker({parent, configuration, key, labelText, defaultValue}) {
    function hexToRgb(hex) {
        hex = hex.replace(/^#/, '');
        const r = parseInt(hex.slice(0, 2), 16) / 255;
        const g = parseInt(hex.slice(2, 4), 16) / 255;
        const b = parseInt(hex.slice(4, 6), 16) / 255;
        return [r, g, b];
    }
    function rgbToHex(rgb) {
        const r = Math.min(1, Math.max(0, rgb[0]));
        const g = Math.min(1, Math.max(0, rgb[1]));
        const b = Math.min(1, Math.max(0, rgb[2]));
        const red = Math.round(r * 255).toString(16).padStart(2, '0');
        const green = Math.round(g * 255).toString(16).padStart(2, '0');
        const blue = Math.round(b * 255).toString(16).padStart(2, '0');
        return `#${red}${green}${blue}`;
    }

    const container = document.createElement("div");
    container.className = "config-row";

    const colorPickerInput = document.createElement("input");
    colorPickerInput.id = uniqueId()
    colorPickerInput.type = "color";

    const label = document.createElement("label");
    label.htmlFor = colorPickerInput.id
    label.textContent = labelText;

    container.appendChild(label);
    container.appendChild(padder(colorPickerInput));
    parent.appendChild(container);

    configuration.addListener(key, value => {
        colorPickerInput.value = rgbToHex(value);
    });
    colorPickerInput.addEventListener("input", function() {
        configuration.update(key, hexToRgb(colorPickerInput.value));
    });
    configuration.addUserConfig(key, defaultValue);
}

function createButton({parent, labelText, fn}) {
    const container = document.createElement("div");
    container.className = "config-row";

    const button = document.createElement("button");
    button.id = uniqueId()
    button.textContent = labelText;

    container.appendChild(padder(button));
    parent.appendChild(container);

    button.addEventListener("click", function() {
        fn();
    });
}

function createTextField({parent, configuration, key, labelText, disabled=false, userConfig=true, defaultValue}) {
    const container = document.createElement("div");
    container.className = "config-row";

    const textInput = document.createElement("input");
    textInput.id = uniqueId()
    textInput.type = "text";
    textInput.disabled = disabled;

    const label = document.createElement("label");
    label.htmlFor = textInput.id;
    label.textContent = labelText;

    container.appendChild(label);
    container.appendChild(padder(textInput));
    parent.appendChild(container);

    configuration.addListener(key, value => {
        textInput.value = value;
    });

    textInput.addEventListener("change", function() {
        configuration.update(key, textInput.value);
    });

    if (userConfig) {
        configuration.addUserConfig(key, defaultValue);
    } else {
        configuration.update(key, defaultValue);
    }
}

function createSection({parent, labelText, closed = false}) {
    const section = document.createElement("div");
    section.className = "menu-section";

    const title = document.createElement("div");
    title.className = "menu-section-title";
    title.innerHTML = labelText + (closed ? " +" : " -");

    const container = document.createElement("div");
    container.className = "menu-container";
    container.style.height = closed ? "0px" : "auto"; // Start collapsed or expanded

    section.appendChild(title);
    section.appendChild(container);
    parent.appendChild(section);

    title.addEventListener("click", function() {
        if (container.style.height === "0px") { // Expand to content height
            container.style.height = container.scrollHeight + "px";
            title.innerHTML = labelText + " -";

        } else { // Collapse
            // First compute current height since we start from auto, then
            // call requestAnimationFrame to collapse after height has first
            // been applied
            container.style.height = container.scrollHeight + "px";
            requestAnimationFrame(() => {
                container.style.height = "0px";
                title.innerHTML = labelText + " +";
            });
        }
    });

    return container;
}

function createUI(config, configuration) {
    const menuDiv = document.getElementById("menu");

    const settingsSection = createSection({parent: menuDiv, labelText: "General"});
    const renderingSection = createSection({parent: menuDiv, labelText: "Rendering"});
    const brushSection = createSection({parent: menuDiv, labelText: "Brush", closed: true});
    const cameraSection = createSection({parent: menuDiv, labelText: "Camera", closed: true});
    const simSection = createSection({parent: menuDiv, labelText: "Simulation", closed: true});
    const debugSection = createSection({parent: menuDiv, labelText: "Debug", closed: true});

    // CONFIG SECTION

    createButton({ parent: settingsSection, labelText: "Reset settings",
        fn: () => configuration.resetConfig()
    });
    createSelect({ parent: settingsSection, configuration,
        key: "scene", labelText: "Scene",
        options: ["None", "Rotating smoke emitter", "Rotating fire emitter", "5 spheres (uses brush settings)"].map(
            x => ({label: x, value: x})),
        defaultValue: "Rotating fire emitter"
    });
    createPreset({parent: settingsSection, configuration, labelText: "Brush presets", presets: [
        { label: "<Custom>" },
        { label: "Wind", props: { brushVelocityAmount: 3.0, brushSmokeAmount: 0, brushFuelAmount: 0, brushTemperatureAmount: 0 } },
        { label: "Smoke", props: { brushVelocityAmount: 3.0, brushSmokeAmount: 3, brushFuelAmount: 0, brushTemperatureAmount: 0 } },
        { label: "Fire", props: { brushVelocityAmount: 2, brushSmokeAmount: 0, brushFuelAmount: 0.15, brushTemperatureAmount: 1 } },
    ]});
    createCheckbox({ parent: settingsSection, configuration,
        key: "enclosed", labelText: "Enclosed volume",
        defaultValue: true
    });
    createSelect({ parent: settingsSection, configuration,
        key: "gridSize", labelText: "Voxel resolution",
        options: [16, 32, 64, 96, 128, 192, 256, 384].map(
            x => ({label: x + String.fromCharCode(179), value: x + ""})),
        defaultValue: 128
    });
    createSlider({ parent: settingsSection, configuration,
        key: "speed", labelText: "Speed",
        minValue: 0.001, maxValue: 5.0, defaultValue: 1.0, step: 0.001
    });

    // RENDERING SECTION

    createSlider({ parent: renderingSection, configuration,
        key: "stepLength", labelText: "Step length",
        minValue: 0.002, maxValue: 0.05, defaultValue: 0.01, step: 0.001
    });
    createColorPicker({ parent: renderingSection, configuration,
        key: "primaryLightColor", labelText: "Primary light color",
        defaultValue: [1.0, 0.928, 0.7142]
    });
    createSlider({ parent: renderingSection, configuration,
        key: "primaryLightBrightness", labelText: "Primary light brightness",
        minValue: 0.0, maxValue: 5.0, defaultValue: 1.4, step: 0.01
    });
    createColorPicker({ parent: renderingSection, configuration,
        key: "fillLightColor", labelText: "Fill light color",
        defaultValue: [0.4, 0.5, 0.7]
    });
    createSlider({ parent: renderingSection, configuration,
        key: "fillLightBrightness", labelText: "Fill light brightness",
        minValue: 0.0, maxValue: 5.0, defaultValue: 0.25, step: 0.01
    });
    createSlider({ parent: renderingSection, configuration,
        key: "blackbodyBrightness", labelText: "Fire brightness",
        minValue: 0.0, maxValue: 10.0, defaultValue: 1.0, step: 0.01
    });

    // SIM SECTION

    createSlider({ parent: simSection, configuration,
        key: "simulation_scale", labelText: "Simulation scale",
        minValue: 1, maxValue: 200, defaultValue: 4.0, step: 0.1
    });
    createSlider({ parent: simSection, configuration,
        key: "pressure_iterations", labelText: "Solver iterations",
        minValue: 2, maxValue: 200, defaultValue: 4, step: 1
    });
    createCheckbox({ parent: simSection, configuration,
        key: "enable_vorticity", labelText: "Enable vorticity",
        defaultValue: true
    });
    createSlider({ parent: simSection, configuration,
        key: "vorticity", labelText: "Vorticity amount",
        minValue: 0, maxValue: 50, defaultValue: 1., step: 0.001
    });
    createSlider({ parent: simSection, configuration,
        key: "velocityDecay", labelText: "Velocity decay",
        minValue: 0.0, maxValue: 10.0, defaultValue: 0.015, step: 0.001
    });
    createSlider({ parent: simSection, configuration,
        key: "smokeDecay", labelText: "Smoke decay",
        minValue: 0.0, maxValue: 10.0, defaultValue: 0.5, step: 0.001
    });
    createSlider({ parent: simSection, configuration,
        key: "temperatureDecay", labelText: "Temperature decay",
        minValue: 0.0, maxValue: 10.0, defaultValue: 1.0, step: 0.001
    });
    createSlider({ parent: simSection, configuration,
        key: "pressureDecay", labelText: "Pressure decay",
        minValue: 0.0, maxValue: 10.0, defaultValue: 2, step: 0.001
    });
    createSlider({ parent: simSection, configuration,
        key: "ignitionTemperature", labelText: "Ignition temperature",
        minValue: 0, maxValue: 6000, defaultValue: 400, step: 1
    });
    createSlider({ parent: simSection, configuration,
        key: "burnRate", labelText: "Burn rate",
        minValue: 0.0, maxValue: 5.0, defaultValue: 0.8, step: 0.01
    });
    createSlider({ parent: simSection, configuration,
        key: "burnHeatEmit", labelText: "Burn heat emit",
        minValue: 0, maxValue: 100000, defaultValue: 20000, step: 1
    });
    createColorPicker({ parent: simSection, configuration,
        key: "smokeRgb", labelText: "Smoke color",
        defaultValue: [0.3, 0.3, 0.3]
    });
    createSlider({ parent: simSection, configuration,
        key: "burnSmokeEmit", labelText: "Burn smoke emit",
        minValue: 0.0, maxValue: 10.0, defaultValue: 1, step: 0.01
    });
    createSlider({ parent: simSection, configuration,
        key: "boyancy", labelText: "Boyancy",
        minValue: 0, maxValue: 5, defaultValue: 1.0, step: 0.001
    });

    // CAMERA SECTION

    createCheckbox({ parent: cameraSection, configuration,
        key: "cameraRotationEnabled", labelText: "Camera rotation",
        defaultValue: true
    });
    createSlider({ parent: cameraSection, configuration,
        key: "cameraRotationSpeed", labelText: "Rotation speed",
        minValue: -1.0, maxValue: 1.0, defaultValue: 0.05, step: 0.001
    });
    createSlider({ parent: cameraSection, configuration,
        key: "camPosRadius", labelText: "Camera distance",
        minValue: 0.1, maxValue: 6.0, defaultValue: 2., step: 0.1
    });
    createSlider({ parent: cameraSection, configuration,
        key: "orthoBlend", labelText: "Orthographic blend",
        minValue: 0.0, maxValue: 1.0, defaultValue: 0., step: 0.01
    });

    // BRUSH SECTION

    createSlider({ parent: brushSection, configuration,
        key: "brushSize", labelText: "Brush size",
        minValue: 0.0, maxValue: 0.5, defaultValue: 0.1, step: 0.001
    });
    createSlider({ parent: brushSection, configuration,
        key: "brushVelocityAmount", labelText: "Mouse velocity strength",
        minValue: 0.0, maxValue: 10.0, defaultValue: 2.0, step: 0.1
    });
    createSlider({ parent: brushSection, configuration,
        key: "brushSmokeAmount", labelText: "Mouse smoke amount",
        minValue: 0.0, maxValue: 20.0, defaultValue: 0., step: 0.1
    });
    createSlider({ parent: brushSection, configuration,
        key: "brushFuelAmount", labelText: "Brush fuel amount",
        minValue: 0.0, maxValue: 3.0, defaultValue: .15, step: 0.001
    });
    createSlider({ parent: brushSection, configuration,
        key: "brushTemperatureAmount", labelText: "Brush temperature",
        minValue: 0.0, maxValue: 100.0, defaultValue: 1., step: 0.01
    });

    // DEBUG SECTION

    createSelect({ parent: debugSection, configuration,
        key: "renderMode", labelText: "Render mode",
        options: ["Normal", "Fuel shaded", "Fuel temperature", "Pressure", "Divergence"].map(
            x => ({label: x, value: x})),
        defaultValue: "Normal"
    });
    createSlider({ parent: debugSection, configuration,
        key: "debugMultigridLevel", labelText: "Multigrid level",
        minValue: 0, maxValue: 6, defaultValue: 0, step: 1
    });
    createCheckbox({ parent: debugSection, configuration,
        key: "useRedBlackJacobi", labelText: "Red/Black Jacobi",
        defaultValue: false
    });
    createTextField({ parent: debugSection, configuration,
        key: "javascriptTime", labelText: "Javascript (ms)",
        defaultValue: "",
        userConfig: false,
        disabled: true
    });
}
