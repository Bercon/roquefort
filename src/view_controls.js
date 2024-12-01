class ViewControls {
    constructor(canvas, configuration) {
        this.events = [];
        this.previous = null;

        canvas.addEventListener('contextmenu', (event) => {
            event.preventDefault(); // Disable right click menu over canvas
        });

        canvas.addEventListener('mousemove', (event) => {
            if (event.buttons == 0) return;
            let prev = this.previous;
            let now = [event.offsetX, event.offsetY];
            this.previous = now;

            const mouseEvent = {
                x: event.offsetX,
                y: event.offsetY,
                dx: prev ? event.offsetX - prev[0] : 0,
                dy: prev ? event.offsetY - prev[1] : 0,
            };

            if (event.buttons === 1) { // left
                this.events.push(mouseEvent);

            } else if (event.buttons === 2) { // right
                configuration.addTheta(-mouseEvent.dy * 0.004);
                configuration.addPhi(-mouseEvent.dx * 0.004);
            } else if (event.buttons === 4) { // middle
            }
        });

        canvas.addEventListener('wheel', (event) => {
            event.preventDefault();
            const delta = event.deltaY;
            configuration.addRadius(delta * 0.002);
        }, { passive: false });

        window.addEventListener('mouseup', () => {
            this.previous = null;
            this.events.length = 0;
        });
    }

    pop() {
        return this.events.shift();
    }

    popAll() {
        return this.events.splice(0, this.events.length);
    }

    getMouseLine() {
        const firstLast = [this.events[0], this.events[this.events.length - 1]];
        this.events.splice(0, this.events.length - 1);
        return firstLast;
    }

    empty() {
        return this.events.length == 0;
    }
}
