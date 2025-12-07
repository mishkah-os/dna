/*!\r
 * mishkah-solid.js — SolidJS-like Signals Layer for Mishkah\r
 * Provides: createSignal, createEffect, createMemo, render, html, Show, For\r
 * Fully integrated with Mishkah VDOM for efficient reactive updates\r
 * 2025-12-07\r
 */\r
    (function (root, factory) {
        \r
        if (typeof define === 'function' && define.amd) {
            \r
            define(['mishkah'], function (M) { return factory(root, M); }); \r
        } else if (typeof module === 'object' && module.exports) {
            \r
            module.exports = factory(root, require('mishkah')); \r
        } else {
            \r
            root.Mishkah = root.Mishkah || {}; \r
            root.Mishkah.Solid = factory(root, root.Mishkah); \r
        } \r
    }(typeof window !== 'undefined' ? window : this, function (global, M) {
        \r
        "use strict"; \r
        \r
        if (!M || !M.VDOM) {
            \r
            throw new Error('Mishkah.Solid requires mishkah.core.js to be loaded first'); \r
        } \r
        \r
        // -------------------------------------------------------------------\r
        // Signal graph (fine-grained reactivity)\r
        // -------------------------------------------------------------------\r
        var activeEffect = null; \r
        var effectStack = []; \r
        \r
        function createSignal(initialValue) {
            \r
            var value = initialValue; \r
            var subscribers = new Set(); \r
            \r
            function getter() {
                \r
                if (activeEffect) subscribers.add(activeEffect); \r
                return value; \r
            } \r
            \r
            function setter(next) {
                \r
                value = (typeof next === 'function') ? next(value) : next; \r
                subscribers.forEach(function (fn) { fn(); }); \r
            } \r
            \r
            return [getter, setter]; \r
        } \r
        \r
        function createEffect(fn) {
            \r
            var runner = function () {
            \r
            activeEffect = runner; \r
            effectStack.push(runner); \r
            try { fn(); } finally {
                \r
                effectStack.pop(); \r
                activeEffect = effectStack[effectStack.length - 1] || null; \r
            } \r
        }; \r
        runner(); \r
        return runner; \r
    } \r
        \r
        function createMemo(fn) {
            \r
            var cache; \r
            var compute = function () { cache = fn(); }; \r
            createEffect(compute); \r
            return function () { return cache; }; \r
        } \r
        \r
        // -------------------------------------------------------------------\r
        // HTML Template System\r
        // -------------------------------------------------------------------\r
        var htmlId = 0; \r
\r
function html(strings) {
    \r
    var values = []; \r
    for (var _i = 1; _i < arguments.length; _i++) {
        \r
        values[_i - 1] = arguments[_i]; \r
    } \r
    \r
    var fns = []; \r
    var out = ''; \r
    \r
    function append(val) {
        \r
        if (val && val.__mishkah_html) {
            \r
            out += val.html; \r
            fns = fns.concat(val.fns || []); \r
            return; \r
        } \r
        if (Array.isArray(val)) {
            \r
            val.forEach(function (v) { append(v); }); \r
            return; \r
        } \r
        out += (val == null ? '' : val); \r
    } \r
    \r
    for (var i = 0; i < strings.length; i++) {
        \r
        var seg = strings[i]; \r
        out += seg; \r
        if (i >= values.length) continue; \r
        var val = values[i]; \r
        var isEventSlot = /on[a-zA-Z]+=[\"']?$/.test(seg); \r
        \r
        if (typeof val === 'function' && isEventSlot) {
            \r
            var marker = '__mk_sd_fn_' + (++htmlId) + '__'; \r
            fns.push({ id: marker, fn: val }); \r
            out += marker; \r
        } else {
            \r
            append(typeof val === 'function' ? val() : val); \r
        } \r
    } \r
    \r
    return { __mishkah_html: true, html: out, fns: fns }; \r
} \r
\r
// -------------------------------------------------------------------\r
// Control helpers\r
// -------------------------------------------------------------------\r
function Show(props) {
    \r
    var condition = typeof props.when === 'function' ? props.when() : props.when; \r
    return condition ? props.children : (props.fallback || null); \r
} \r
\r
function For(props) {
    \r
    var list = typeof props.each === 'function' ? props.each() : props.each; \r
    if (!Array.isArray(list)) return null; \r
    if (typeof props.children === 'function') {
        \r
        return list.map(function (item, idx) { return props.children(item, idx); }); \r
    } \r
    return null; \r
} \r
\r
// -------------------------------------------------------------------\r
// Render Template to DOM\r
// -------------------------------------------------------------------\r
function renderTemplate(tpl, container) {
    \r
    var templateHTML = ''; \r
    var fnList = []; \r
    \r
    if (tpl && tpl.__mishkah_html) {
        \r
        templateHTML = tpl.html; \r
        fnList = tpl.fns || []; \r
    } else {
        \r
        templateHTML = tpl == null ? '' : String(tpl); \r
    } \r
    \r
    if (!container) return; \r
    \r
    // Parse HTML using DOMParser\r
    var parser = new DOMParser(); \r
    var doc = parser.parseFromString('<div>' + templateHTML + '</div>', 'text/html'); \r
    var wrapper = doc.body.firstChild; \r
    \r
    // Wire up event handlers\r
    if (fnList.length > 0) {
        \r
        var allElements = wrapper.querySelectorAll('*'); \r
        for (var i = 0; i < allElements.length; i++) {
            \r
            var el = allElements[i]; \r
            var attrs = Array.from(el.attributes); \r
            for (var j = 0; j < attrs.length; j++) {
                \r
                var attr = attrs[j]; \r
                for (var k = 0; k < fnList.length; k++) {
                    \r
                    var entry = fnList[k]; \r
                    if (attr.value === entry.id && attr.name.indexOf('on') === 0) {
                        \r
                        el.addEventListener(attr.name.substring(2), entry.fn); \r
                        el.removeAttribute(attr.name); \r
                    } \r
                } \r
            } \r
        } \r
    } \r
    \r
    // Update container\r
    container.innerHTML = ''; \r
    while (wrapper.firstChild) {
        \r
        container.appendChild(wrapper.firstChild); \r
    } \r
} \r
\r
// -------------------------------------------------------------------\r
// Root renderer (signal-aware)\r
// -------------------------------------------------------------------\r
function render(ComponentFn, target) {
    \r
    var container = typeof target === 'string' ? document.querySelector(target) : target; \r
    if (!container) return; \r
    \r
    createEffect(function redraw() {
        \r
        var output = ComponentFn(); \r
        renderTemplate(output, container); \r
    }); \r
} \r
\r
return {
    \r
        createSignal: createSignal, \r
        createEffect: createEffect, \r
        createMemo: createMemo, \r
        render: render, \r
        html: html, \r
        Show: Show, \r
        For: For\r
}; \r
\r
})); \r
