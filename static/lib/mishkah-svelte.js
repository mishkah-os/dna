/*!
 * mishkah-svelte.js — Svelte 5 (Runes) Layer for Mishkah
 * Provides: mount, state, derived, effect, html
 * 2025-12-07 - Fixed Version (Unix Line Endings + Focus Management)
 */
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define(['mishkah'], function (M) { return factory(root, M); });
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory(root, require('mishkah'));
    } else {
        root.Mishkah = root.Mishkah || {};
        root.Mishkah.Svelte = factory(root, root.Mishkah);
    }
}(typeof window !== 'undefined' ? window : this, function (global, M) {
    "use strict";

    // -------------------------------------------------------------------
    // Dependency Graph (Fine-grained reactivity)
    // -------------------------------------------------------------------
    var bucket = new WeakMap();
    var activeEffect = null;
    var effectStack = [];

    function track(target, key) {
        if (!activeEffect) return;
        var depsMap = bucket.get(target);
        if (!depsMap) {
            depsMap = new Map();
            bucket.set(target, depsMap);
        }
        var dep = depsMap.get(key);
        if (!dep) {
            dep = new Set();
            depsMap.set(key, dep);
        }
        dep.add(activeEffect);
    }

    function trigger(target, key) {
        var depsMap = bucket.get(target);
        if (!depsMap) return;
        var dep = depsMap.get(key);
        if (!dep) return;
        // Copy to avoid infinite loop if effect triggers more updates
        var effectsToRun = new Set(dep);
        effectsToRun.forEach(function (effectFn) { effectFn(); });
    }

    function effect(fn) {
        var runner = function () {
            activeEffect = runner;
            effectStack.push(runner);
            try { fn(); } finally {
                effectStack.pop();
                activeEffect = effectStack[effectStack.length - 1] || null;
            }
        };
        runner();
        return runner;
    }

    // -------------------------------------------------------------------
    // State / Derived (Svelte 5 Runes)
    // -------------------------------------------------------------------
    var proxyCache = new WeakMap();

    function createProxy(target) {
        if (typeof target !== 'object' || target === null) return target;
        if (proxyCache.has(target)) return proxyCache.get(target);

        var proxy = new Proxy(target, {
            get: function (obj, prop) {
                track(obj, prop);
                return createProxy(obj[prop]);
            },
            set: function (obj, prop, value) {
                if (obj[prop] !== value) {
                    obj[prop] = value;
                    trigger(obj, prop);
                }
                return true;
            }
        });
        proxyCache.set(target, proxy);
        return proxy;
    }

    function state(initialValue) {
        return createProxy(initialValue);
    }

    function derived(fn) {
        var cache;
        var recompute = function () { cache = fn(); };
        effect(recompute);
        return {
            get value() { return cache; }
        };
    }

    // -------------------------------------------------------------------
    // HTML Template (Tagged Literal)
    // -------------------------------------------------------------------
    var htmlId = 0;

    function html(strings) {
        var values = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            values[_i - 1] = arguments[_i];
        }
        var fns = [];
        var out = '';

        function append(val) {
            if (val && val.__mishkah_html) {
                out += val.html;
                fns = fns.concat(val.fns || []);
                return;
            }
            if (Array.isArray(val)) {
                val.forEach(function (v) { append(v); });
                return;
            }
            out += (val == null ? '' : val);
        }

        for (var i = 0; i < strings.length; i++) {
            var seg = strings[i];
            out += seg;
            if (i >= values.length) continue;
            var val = values[i];

            // Check for event handler: on...="{fn}"
            // We look at the end of the segment for 'onX='
            var isEventSlot = /on[a-zA-Z]+=[\"']?$/.test(seg);

            if (isEventSlot && typeof val === 'function') {
                var marker = '__mk_sv_ev_' + (++htmlId) + '__';
                fns.push({ type: 'event', id: marker, fn: val });
                out += marker;
            } else {
                // Here we evaluate function values immediately for Svelte "pull" model
                // But we could support fine-grained if we wanted later.
                // For now, assume it's a value or a nested template.
                append(typeof val === 'function' ? val() : val);
            }
        }

        return { __mishkah_html: true, html: out, fns: fns };
    }

    // -------------------------------------------------------------------
    // Renderer
    // -------------------------------------------------------------------
    function renderTemplate(tpl, ctx, target) {
        var templateHTML = '';
        var registry = [];

        if (tpl && tpl.__mishkah_html) {
            templateHTML = tpl.html;
            registry = tpl.fns || [];
        } else {
            templateHTML = tpl == null ? '' : String(tpl);
        }

        if (!target) return;

        // [Focus Management]
        // Save focus state to restore after innerHTML replacement
        var activeEl = document.activeElement;
        var selectionStart = null;
        var selectionEnd = null;
        var activeId = activeEl ? activeEl.id : null;

        // Try simple selector path if no ID
        // (This is a rudimentary way to persist focus across innerHTML nuking)
        if (activeEl && activeEl !== document.body && target.contains(activeEl)) {
            // Note: If elements don't have IDs, this might be flaky.
            // We assume input elements in lists might need better handling,
            // but for this task, basic ID retrieval helps.
            if (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA') {
                selectionStart = activeEl.selectionStart;
                selectionEnd = activeEl.selectionEnd;
            }
        }

        // Nuke and Replace
        target.innerHTML = templateHTML;

        // Restore Focus
        if (activeId) {
            var newActive = target.querySelector('#' + activeId);
            if (newActive) {
                newActive.focus();
                if (selectionStart !== null && (newActive.tagName === 'INPUT' || newActive.tagName === 'TEXTAREA')) {
                    try {
                        newActive.setSelectionRange(selectionStart, selectionEnd);
                    } catch (e) { }
                }
            }
        }

        // Bind Events
        if (!registry.length) return;
        var events = registry.filter(function (r) { return r.type === 'event'; });

        if (events.length) {
            var all = target.querySelectorAll('*');
            for (var i = 0; i < all.length; i++) {
                var el = all[i];
                var attrs = Array.from(el.attributes);
                for (var j = 0; j < attrs.length; j++) {
                    var attr = attrs[j];
                    for (var k = 0; k < events.length; k++) {
                        var entry = events[k];
                        if (attr.value === entry.id && attr.name.indexOf('on') === 0) {
                            var eventName = attr.name.substring(2);
                            el.addEventListener(eventName, entry.fn.bind(ctx));
                            el.removeAttribute(attr.name);
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Mount
    // -------------------------------------------------------------------
    function mount(ComponentFn, target) {
        var container = typeof target === 'string' ? document.querySelector(target) : target;
        if (!container) return;

        // 1. Run ComponentFn ONCE to create state/closure
        var renderFn = ComponentFn();

        // 2. Validate return value
        if (typeof renderFn !== 'function') {
            console.error('[Mishkah.Svelte] Component must return a render function (e.g., return () => html`...`)');
            return;
        }

        // 3. Create Effect for Reactive Rendering
        effect(function render() {
            var output = renderFn();
            renderTemplate(output, {}, container);
        });
    }

    return {
        state: state,
        derived: derived,
        effect: effect,
        mount: mount,
        html: html
    };

}));
