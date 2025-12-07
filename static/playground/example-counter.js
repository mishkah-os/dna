(function () {
  'use strict';

  // ============================================================
  // EXAMPLES Data - Counter Example with Multiple Frameworks
  // ============================================================

  const vueComplete = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Vue Counter Deluxe</title>
  <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
  <style>
    body { font-family: system-ui; background: #0f172a; color: #e2e8f0; display: grid; place-items: center; min-height: 100vh; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 16px; padding: 2rem; max-width: 520px; width: 100%; box-shadow: 0 20px 60px rgba(0,0,0,.25); }
    .count { font-size: 3rem; font-weight: 900; margin: 0.5rem 0 1rem; color: #22c55e; }
    button { padding: 0.75rem 1.5rem; border-radius: 10px; border: none; cursor: pointer; margin-right: 0.5rem; font-weight: 700; }
    button.inc { background: linear-gradient(135deg,#22c55e,#16a34a); color: #0f172a; }
    button.dec { background: linear-gradient(135deg,#f97316,#ea580c); color: #0f172a; }
    button.reset { background: #0ea5e9; color: #0f172a; }
  </style>
</head>
<body>
  <div id="app"></div>
  <script>
    const messages = {
      en: { title: 'Vue 3 Counter', count: 'Current Count', inc: 'Increment', dec: 'Decrement', reset: 'Reset' },
      ar: { title: 'عداد Vue 3', count: 'القيمة الحالية', inc: 'زيادة', dec: 'نقصان', reset: 'إعادة' }
    };

    const { createApp, ref, computed } = Vue;
    createApp({
      setup() {
        const lang = ref('en');
        const count = ref(0);
        const double = computed(() => count.value * 2);
        const t = (k) => messages[lang.value][k];
        const flip = () => { lang.value = lang.value === 'en' ? 'ar' : 'en'; document.documentElement.dir = lang.value === 'ar' ? 'rtl' : 'ltr'; };
        return { lang, count, double, t, flip, inc: () => count.value++, dec: () => count.value--, reset: () => count.value = 0 };
      },
      template: `
        <div class="card">
          <header style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
            <h1>{{ t('title') }}</h1>
            <button class="reset" @click="flip">🌐 {{ lang === 'ar' ? 'EN' : 'AR' }}</button>
          </header>
          <p>{{ t('count') }}</p>
          <div class="count">{{ count }}</div>
          <p style="color:#94a3b8">×2 = {{ double }}</p>
          <div>
            <button class="inc" @click="inc">{{ t('inc') }}</button>
            <button class="dec" @click="dec">{{ t('dec') }}</button>
            <button class="reset" @click="reset">{{ t('reset') }}</button>
          </div>
        </div>
      `
    }).mount('#app');
  </script>
</body>
</html>`;

  const reactBasic = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Mishkah React Counter</title>
  <script src="../lib/mishkah.core.js"></script>
  <script src="../lib/mishkah-react.js"></script>
  <script src="../lib/mishkah-jsx.js"></script>
  <style>
    body { font-family: 'Inter', system-ui; background: #0b1224; color: #e2e8f0; padding: 40px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 24px; max-width: 520px; }
    .count { font-size: 48px; font-weight: 900; margin: 16px 0; color: #a855f7; }
    .btn { border: none; padding: 12px 18px; margin-right: 10px; border-radius: 12px; cursor: pointer; font-weight: 700; }
    .btn.inc { background: #22c55e; color: #052e16; }
    .btn.dec { background: #ef4444; color: #fff; }
    .btn.reset { background: #3b82f6; color: #fff; }
  </style>
</head>
<body>
  <div id="app"></div>
  <script type="text/jsx">
    const { useState } = Mishkah.React;

    function Counter() {
      const [count, setCount] = useState(0);
      return (
        <div className="card">
          <h1>React JSX Counter</h1>
          <div className="count">{count}</div>
          <div>
            <button className="btn inc" onClick={() => setCount(count + 1)}>Increment</button>
            <button className="btn dec" onClick={() => setCount(count - 1)}>Decrement</button>
            <button className="btn reset" onClick={() => setCount(0)}>Reset</button>
          </div>
          <p style={{color:'#9ca3af'}}>Double: {count * 2}</p>
        </div>
      );
    }

    Mishkah.React.render(Counter, document.getElementById('app'));
  </script>
</body>
</html>`;

  const angularLite = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Mishkah Angular Counter</title>
  <script src="../lib/mishkah.core.js"></script>
  <script src="../lib/mishkah-react.js"></script>
  <script src="../lib/mishkah-angular.js"></script>
  <style>
    body { font-family: system-ui; background:#0a0a0a; color:#f1f5f9; display:grid; place-items:center; min-height:100vh; }
    .card { padding:24px; background:#111827; border-radius:12px; border:1px solid #1f2937; width:380px; }
    button { padding:10px 14px; border:none; border-radius:10px; cursor:pointer; margin-right:8px; font-weight:700; }
    .primary { background:#ef4444; color:white; }
    .ghost { background:#1f2937; color:#e2e8f0; }
  </style>
</head>
<body>
  <div id="app-angular"></div>
  <script>
    const { bootstrap, html } = Mishkah.Angular;

    class CounterComponent {
      static selector = 'app-counter';
      static template = function () {
        return html`
          <div class="card">
            <h2>Angular-like Component</h2>
            <p>Count: <strong>${this.count}</strong></p>
            <div style="display:flex;gap:8px;">
              <button class="primary" onclick="${this.inc}">Increment</button>
              <button class="ghost" onclick="${this.dec}">Decrement</button>
            </div>
            <p style="color:#94a3b8">${this.message}</p>
          </div>
        `;
      }

      count = 0;
      message = 'Ready';
      inc() { this.count++; this.message = 'Incremented!'; }
      dec() { this.count--; this.message = 'Decremented!'; }
      onInit() { console.log('Angular Component Initialized'); }
    }

    bootstrap(CounterComponent, '#app-angular');
  </script>
</body>
</html>`;

  const alpineLite = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Alpine-like Counter</title>
  <script src="../lib/mishkah.core.js"></script>
  <script src="../lib/mishkah-react.js"></script>
  <script src="../lib/mishkah-svelte.js"></script>
  <script src="../lib/mishkah-alpine.js"></script>
  <style>
    body { font-family: system-ui; background:#0f172a; color:#e2e8f0; padding:40px; }
    .card { background:#111827; border:1px solid #1f2937; border-radius:14px; padding:18px; max-width:420px; }
    button { padding:10px 14px; border:none; border-radius:10px; cursor:pointer; margin-right:8px; }
  </style>
</head>
<body>
  <div x-data="{ count: 0, open: false }" class="card">
    <h2>Alpine Style Counter</h2>
    <p>Count: <strong x-text="count"></strong></p>
    <button @click="count++">Increment</button>
    <button @click="open = !open">Toggle Details</button>
    <div x-show="open" style="margin-top: 10px; padding: 10px; background: #0b1224; border-radius:12px;">
      <p>This is toggled via <code>x-show</code>!</p>
      <p>Current count is <span x-text="count"></span></p>
    </div>
  </div>
</body>
</html>`;

  const svelteRunes = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Mishkah Svelte Counter</title>
  <script src="../lib/mishkah.core.js"></script>
  <script src="../lib/mishkah-react.js"></script>
  <script src="../lib/mishkah-svelte.js"></script>
  <style>
    body { font-family: system-ui; background:#0b1224; color:#e2e8f0; display:grid; place-items:center; min-height:100vh; }
    .card { padding:24px; border-radius:14px; background:#111827; border:1px solid #1f2937; width:420px; }
    button { padding:10px 14px; border:none; border-radius:10px; cursor:pointer; margin-right:8px; background:#ff3e00; color:#fff; font-weight:700; }
  </style>
</head>
<body>
  <div id="app-svelte"></div>
  <script>
    const { mount, state, derived, effect, html } = Mishkah.Svelte;

    function Counter() {
      const s = state({ count: 0, text: 'Hello Svelte' });
      const double = derived(() => s.count * 2);
      effect(() => console.log('Count changed', s.count));

      function inc() { s.count++; }
      function updateText(e) { s.text = e.target.value; }

      return () => html`
        <div class="card">
          <h2>Svelte 5 (Runes) 🧡</h2>
          <p>Count: <strong>${s.count}</strong> (Double: ${double.value})</p>
          <button onclick="${inc}">Increment</button>
          <hr>
          <p>${s.text}</p>
          <input value="${s.text}" oninput="${updateText}" />
        </div>
      `;
    }

    mount(Counter, document.getElementById('app-svelte'));
  </script>
</body>
</html>`;

  const solidSignals = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SolidJS Signals Counter</title>
  <script src="../lib/mishkah.core.js"></script>
  <script src="../lib/mishkah-react.js"></script>
  <script src="../lib/mishkah-solid.js"></script>
  <style>
    body { font-family: system-ui; background:#020617; color:#e2e8f0; display:grid; place-items:center; min-height:100vh; }
    .card { padding:22px; border-radius:16px; background:#0f172a; border:1px solid #1e293b; width:420px; }
    button { padding:10px 14px; border:none; border-radius:10px; cursor:pointer; margin-right:8px; background:#3b82f6; color:white; }
  </style>
</head>
<body>
  <div id="app-solid"></div>
  <script>
    const { render, createSignal, createEffect, createMemo, html, Show } = Mishkah.Solid;

    function Counter() {
      const [count, setCount] = createSignal(0);
      const [show, setShow] = createSignal(true);
      const double = createMemo(() => count() * 2);
      createEffect(() => console.log('Solid Effect: Count is', count()));

      return html`
        <div class="card">
          <h2>SolidJS (Signals) 🚀</h2>
          <p>Count: <strong>${() => count()}</strong> (Double: ${() => double()})</p>
          <div style="display:flex;gap:10px;margin-bottom:10px;">
            <button onclick="${() => setCount(c => c + 1)}">Increment</button>
            <button onclick="${() => setShow(s => !s)}" style="background:#6366f1">Toggle Info</button>
          </div>
          ${Show({ when: show, children: html`<p style="color:#94a3b8;font-size:0.9em;"><em>Fine-grained signals demo.</em></p>` })}
        </div>
      `;
    }

    render(Counter, document.getElementById('app-solid'));
  </script>
</body>
</html>`;

  const mishkahDsl = `// Mishkah DSL Counter inspired by upgrade test
const database = {
  showSecondButton: true,
  counter: 0
};

const orders = {
  toggle: { on: ['click'], gkeys: ['btn:toggle'], handler: (e, ctx) => ctx.setState(s => ({ ...s, showSecondButton: !s.showSecondButton })) },
  inc: { on: ['click'], gkeys: ['btn:inc'], handler: (e, ctx) => ctx.setState(s => ({ ...s, counter: s.counter + 1 })) }
};

function App(state) {
  const D = Mishkah.DSL;
  const h = Mishkah.h;
  return D.Div({ attrs: { class: 'container' } }, [
    D.Div({ attrs: { class: 'test-card' } }, [
      D.H3({}, ['Test 1: Flat DSL & h()']),
      D.P({}, ['✅ If you see this, D.Div and D.H3 are working directly.']),
      h('p', 'Text', { attrs: { style: 'color: blue' } }, ['✅ h() is working.'])
    ]),
    D.Div({ attrs: { class: 'test-card' } }, [
      D.H3({}, ['Counter']),
      D.Button({ gkey: 'btn:inc', attrs: { class: 'btn' } }, ['Increment Counter: ' + state.counter])
    ])
  ]);
}

Mishkah.app.setBody(App);
const app = Mishkah.app.createApp(database, orders);
app.mount('#app');`;

  const vanillaBasic = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Vanilla JS Counter</title>
  <style>
    body { display:flex; align-items:center; justify-content:center; min-height:100vh; margin:0; background:#0b1224; font-family:system-ui; }
    .card { background:#111827; border-radius:12px; padding:24px; width:360px; color:#e2e8f0; text-align:center; }
    .count { font-size:48px; color:#22c55e; margin:16px 0; font-weight:900; }
    button { padding:10px 16px; margin:0 6px; border:none; border-radius:10px; cursor:pointer; font-weight:700; }
    .inc { background:#22c55e; color:#052e16; }
    .dec { background:#ef4444; color:white; }
    .reset { background:#0ea5e9; color:#082f49; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Vanilla Counter</h1>
    <div class="count" id="count">0</div>
    <div>
      <button class="inc" id="inc">Increment</button>
      <button class="dec" id="dec">Decrement</button>
      <button class="reset" id="reset">Reset</button>
    </div>
  </div>
  <script>
    let count = 0;
    const display = document.getElementById('count');
    const inc = document.getElementById('inc');
    const dec = document.getElementById('dec');
    const reset = document.getElementById('reset');
    const update = () => display.textContent = count;
    inc.addEventListener('click', () => { count++; update(); });
    dec.addEventListener('click', () => { count--; update(); });
    reset.addEventListener('click', () => { count = 0; update(); });
  </script>
</body>
</html>`;

  const jqueryBasic = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>jQuery Counter</title>
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <style>
    body { display:flex; align-items:center; justify-content:center; min-height:100vh; margin:0; background:#0b1224; font-family:system-ui; }
    .card { background:#0f172a; border-radius:12px; padding:24px; width:360px; color:#e2e8f0; text-align:center; border:1px solid #1f2937; }
    .count { font-size:48px; color:#22c55e; margin:16px 0; font-weight:900; }
    button { padding:10px 16px; margin:0 6px; border:none; border-radius:10px; cursor:pointer; font-weight:700; }
  </style>
</head>
<body>
  <div class="card">
    <h1>jQuery Counter</h1>
    <div class="count" id="count">0</div>
    <div>
      <button id="inc">Increment</button>
      <button id="dec">Decrement</button>
      <button id="reset">Reset</button>
    </div>
  </div>
  <script>
    let count = 0;
    const update = () => $('#count').text(count);
    $('#inc').on('click', () => { count++; update(); });
    $('#dec').on('click', () => { count--; update(); });
    $('#reset').on('click', () => { count = 0; update(); });
  </script>
</body>
</html>`;

  window.EXAMPLES = [
    {
      id: 'counter',
      title: { en: 'Counter Example', ar: 'مثال العداد' },
      description: {
        en: 'A gallery of counter implementations across modern frameworks.',
        ar: 'مجموعة أمثلة عداد لأشهر أطر العمل.'
      },
      readme: {
        en: `# Counter Example\n\nقارن بين أكثر من 8 أطر عمل في نفس الوقت.`,
        ar: `# مثال العداد\n\nقارن بين أكثر من 8 أطر عمل في نفس الوقت.`
      },
      wikiId: 'counter-basics',
      implementations: [
        { framework: 'vanilla', wikiId: 'vanilla-counter', code: vanillaBasic },
        { framework: 'jquery', wikiId: 'jquery-counter', code: jqueryBasic },
        { framework: 'vue', wikiId: 'vue-counter', code: vueComplete },
        { framework: 'react', wikiId: 'react-counter', code: reactBasic },
        { framework: 'angular', wikiId: 'angular-counter', code: angularLite },
        { framework: 'alpine', wikiId: 'alpine-counter', code: alpineLite },
        { framework: 'svelte', wikiId: 'svelte-counter', code: svelteRunes },
        { framework: 'solid', wikiId: 'solid-counter', code: solidSignals },
        { framework: 'mishkah-dsl', wikiId: 'mishkah-dsl-counter', code: mishkahDsl }
      ]
    }
  ];
})();
