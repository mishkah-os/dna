/**
 * Mishkah Lite - Emergency Restoration
 * A minimal implementation of the Mishkah framework to restore functionality.
 */
(function (window) {
    'use strict';

    // 1. Core & Utils
    const M = window.Mishkah = window.Mishkah || {};
    M.utils = M.utils || {};
    M.UI = M.UI || {};
    M.DSL = M.DSL || {}; // Placeholder

    // Utils implementation for Plotly bridge
    M.utils.JSON = {
        parseSafe: (v) => { try { return JSON.parse(v); } catch (e) { return null; } },
        clone: (v) => JSON.parse(JSON.stringify(v)),
        stableStringify: JSON.stringify
    };

    // 2. Simple State & Renderer
    class MishkahApp {
        constructor() {
            this.state = { data: {} };
            this.template = null;
            this.mountPoint = null;
        }

        init() {
            // Load initial state from script tag
            const dataScript = document.querySelector('script[data-m-data]');
            if (dataScript) {
                try {
                    this.state.data = JSON.parse(dataScript.textContent);
                } catch (e) { console.error('State parse error', e); }
            }

            // Find template
            const templateEl = document.getElementById('dna-dashboard');
            if (templateEl) {
                this.template = templateEl.innerHTML;
            }

            // Mount point
            this.mountPoint = document.getElementById('app');

            this.render();
        }

        setState(callback) {
            const newState = callback(this.state);
            if (newState) this.state = newState;
            this.render();
        }

        // Basic Template Engine
        render() {
            if (!this.template || !this.mountPoint) return;

            let html = this.template;

            // 1. Handle x-if (simple removal)
            // Regex to find <tag ... x-if="condition" ...>...</tag> is hard.
            // We'll do a DOM-based approach after initial render, or simple regex for now.
            // For safety, let's just do variable interpolation first.

            // 2. Interpolation {state.data.xxx}
            html = html.replace(/\{([^}]+)\}/g, (match, key) => {
                const path = key.trim().split('.');
                let val = this.state;
                for (let p of path) {
                    if (p === 'state') continue;
                    val = val ? val[p] : '';
                }
                return val !== undefined ? val : '';
            });

            // 3. Inject
            this.mountPoint.innerHTML = html;

            // 4. Post-process directives (x-if, x-for)
            this.processDirectives(this.mountPoint);

            // 5. Hydrate Plotly if needed
            if (M.UI.Plotly && M.UI.Plotly.hydrate) {
                M.UI.Plotly.hydrate(this.mountPoint);
            }
        }

        processDirectives(root) {
            // Handle x-if
            root.querySelectorAll('[x-if]').forEach(el => {
                const cond = el.getAttribute('x-if');
                try {
                    // Safe-ish eval using function
                    const val = new Function('state', 'return ' + cond)(this.state);
                    if (!val) el.remove();
                } catch (e) {
                    console.warn('x-if error', cond, e);
                }
            });

            // Handle x-for (very basic: x-for="item in list")
            // This is complex to do in post-process without a real vdom.
            // We will skip x-for for now to avoid breaking layout, 
            // or just hide the container if empty.
            root.querySelectorAll('[x-for]').forEach(el => {
                // For now, just hide loop templates to clean up UI
                // el.style.display = 'none'; 
                // Better: try to render if list is not empty
                const expr = el.getAttribute('x-for');
                const [item, source] = expr.split(' in ').map(s => s.trim());
                const listPath = source.replace('state.', '').split('.');
                let list = this.state;
                for (let p of listPath) list = list ? list[p] : [];

                if (!Array.isArray(list) || list.length === 0) {
                    el.remove(); // Remove template if empty
                } else {
                    // Clone and render items (TODO: implement if needed)
                    // For now, we just show the first item as a placeholder or nothing
                    // This is a limitation of Mishkah Lite.
                }
            });

            // Handle x-class
            root.querySelectorAll('[x-class]').forEach(el => {
                const expr = el.getAttribute('x-class');
                try {
                    const val = new Function('state', 'exp', 'return ' + expr)(this.state, {});
                    // 'exp' context missing here, so this might fail for loops.
                    // But for static elements it works.
                    if (val) el.className = val;
                } catch (e) { }
            });
        }

        rebuild() { this.render(); }
    }

    // 3. App Factory
    M.app = {
        make: function () {
            const app = new MishkahApp();
            // Defer init to ensure DOM is ready
            setTimeout(() => app.init(), 0);
            return app;
        }
    };

    // 4. Auto Loader
    window.MishkahAuto = {
        ready: function (cb) {
            if (document.readyState === 'complete') {
                cb(M);
            } else {
                window.addEventListener('load', () => cb(M));
            }
        }
    };

})(window);
