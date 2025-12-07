/*!
 * mishkah-svelte.js — Svelte 5 (Runes) Layer for Mishkah
 * Provides: mount, state, derived, effect, html
 * 2025-12-03 (standalone, non-VDOM)
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
    // Tiny Dependency Graph (no VDOM, fine-grained)
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
        dep.forEach(function (effectFn) { effectFn(); });
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
    // State / Derived
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
                obj[prop] = value;
                trigger(obj, prop);
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
        var recompute = function () {
            cache = fn();
        };
        effect(recompute);
        return {
            get value() { return cache; }
        };
    }

    // -------------------------------------------------------------------
    // Template helper
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
            var isEventSlot = /on[a-zA-Z]+=["']?$/.test(seg);

            if (typeof val === 'function' && isEventSlot) {
                var marker = '__mk_sv_fn_' + (++htmlId) + '__';
                fns.push({ id: marker, fn: val });
                out += marker;
            } else {
                append(typeof val === 'function' ? val() : val);
            }
        }

        return { __mishkah_html: true, html: out, fns: fns };
    }

    function renderTemplate(tpl, ctx, target) {
        var templateHTML = '';
        var fnList = [];

        if (tpl && tpl.__mishkah_html) {
            templateHTML = tpl.html;
            fnList = tpl.fns || [];
        } else {
            templateHTML = tpl == null ? '' : String(tpl);
        }

        if (!target) return;
        target.innerHTML = templateHTML;

        if (!fnList.length) return;
        var all = target.querySelectorAll('*');
        for (var i = 0; i < all.length; i++) {
            var el = all[i];
            var attrs = Array.from(el.attributes);
            for (var j = 0; j < attrs.length; j++) {
                var attr = attrs[j];
                for (var k = 0; k < fnList.length; k++) {
                    var entry = fnList[k];
                    if (attr.value === entry.id && attr.name.indexOf('on') === 0) {
                        el.addEventListener(attr.name.substring(2), entry.fn.bind(ctx));
                        el.removeAttribute(attr.name);
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Mount (component rerenders when used state changes)
    // -------------------------------------------------------------------
    function mount(ComponentFn, target) {
        var container = typeof target === 'string' ? document.querySelector(target) : target;
        if (!container) return;

        effect(function render() {
            var output = ComponentFn();
            if (typeof output === 'function') output = output();
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
