/*!
 * mishkah-solid.js — SolidJS-like Signals Layer for Mishkah
 * Provides: createSignal, createEffect, createMemo, render, html, Show, For
 * 2025-12-12 (signal-driven, Mishkah-core template renderer)
 */
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define(['mishkah'], function (M) { return factory(root, M); });
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory(root, require('mishkah'));
    } else {
        root.Mishkah = root.Mishkah || {};
        root.Mishkah.Solid = factory(root, root.Mishkah);
    }
}(typeof window !== 'undefined' ? window : this, function (global, M) {
    "use strict";

    // -------------------------------------------------------------------
    // Signal graph (fine-grained reactivity)
    // -------------------------------------------------------------------
    var activeEffect = null;
    var effectStack = [];

    function createSignal(initialValue) {
        var value = initialValue;
        var subscribers = new Set();

        function getter() {
            if (activeEffect) subscribers.add(activeEffect);
            return value;
        }

        function setter(next) {
            value = (typeof next === 'function') ? next(value) : next;
            subscribers.forEach(function (fn) { fn(); });
        }

        return [getter, setter];
    }

    function createEffect(fn) {
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

    function createMemo(fn) {
        var cache;
        var compute = function () { cache = fn(); };
        createEffect(compute);
        return function () { return cache; };
    }

    // -------------------------------------------------------------------
    // Template helper (HTML + event wiring)
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
                var marker = '__mk_sd_fn_' + (++htmlId) + '__';
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
    // Control helpers
    // -------------------------------------------------------------------
    function Show(props) {
        var condition = typeof props.when === 'function' ? props.when() : props.when;
        return condition ? props.children : (props.fallback || null);
    }

    function For(props) {
        var list = typeof props.each === 'function' ? props.each() : props.each;
        if (!Array.isArray(list)) return null;
        if (typeof props.children === 'function') {
            return list.map(function (item, idx) { return props.children(item, idx); });
        }
        return null;
    }

    // -------------------------------------------------------------------
    // Root renderer (signal-aware)
    // -------------------------------------------------------------------
    function render(ComponentFn, target) {
        var container = typeof target === 'string' ? document.querySelector(target) : target;
        if (!container) return;

        createEffect(function redraw() {
            var output = ComponentFn();
            renderTemplate(output, {}, container);
        });
    }

    return {
        createSignal: createSignal,
        createEffect: createEffect,
        createMemo: createMemo,
        render: render,
        html: html,
        Show: Show,
        For: For
    };

}));
