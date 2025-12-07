/*!
 * mishkah-solid.js — SolidJS (Signals) Layer for Mishkah
 * Provides: createSignal, createEffect, createMemo, render
 * 2025-12-03 (standalone edition)
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

    var activeSubscriber = null;
    var htmlId = 0;

    function html(strings) {
        var values = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            values[_i - 1] = arguments[_i];
        }

        var fns = [];
        var out = '';
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
                out += typeof val === 'function' ? val() : (val == null ? '' : val);
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

    function createSignal(initialValue) {
        var value = initialValue;
        var listeners = new Set();

        function getter() {
            if (activeSubscriber) listeners.add(activeSubscriber);
            return value;
        }

        function setter(next) {
            value = typeof next === 'function' ? next(value) : next;
            listeners.forEach(function (fn) { return fn(); });
        }

        return [getter, setter];
    }

    function createEffect(fn) {
        var runner = function () {
            activeSubscriber = runner;
            fn();
            activeSubscriber = null;
        };
        runner();
    }

    function createMemo(fn) {
        var cache;
        var dirty = true;
        var listeners = new Set();

        function recompute() {
            activeSubscriber = track;
            cache = fn();
            dirty = false;
            activeSubscriber = null;
        }

        function track() {
            listeners.add(activeSubscriber);
        }

        return function () {
            if (dirty) recompute();
            if (activeSubscriber) listeners.add(activeSubscriber);
            return cache;
        };
    }

    function render(ComponentFn, target) {
        var container = typeof target === 'string' ? document.querySelector(target) : target;
        if (!container) return;

        function redraw() {
            activeSubscriber = redraw;
            var output = ComponentFn();
            renderTemplate(output, {}, container);
            activeSubscriber = null;
        }

        redraw();
    }

    function Show(props) {
        var condition = typeof props.when === 'function' ? props.when() : props.when;
        return condition ? props.children : (props.fallback || null);
    }

    function For(props) {
        var list = typeof props.each === 'function' ? props.each() : props.each;
        if (!Array.isArray(list)) return null;
        if (typeof props.children === 'function') return list.map(props.children);
        return null;
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
