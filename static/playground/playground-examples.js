/**
 * Mishkah Playground - Examples Library
 * All examples in pure JavaScript object
 * No CORS issues - Works offline!
 */

(function (global) {
    'use strict';

    // ============================================================
    // Examples Database
    // ============================================================

    var EXAMPLES = {

        // ============================================================
        // BASIC EXAMPLES
        // ============================================================

        'counter': {
            id: 'counter',
            title: 'Counter Example',
            titleAr: 'مثال العداد',
            category: 'basic',
            difficulty: 1,
            description: 'Simple reactivity demonstration with increment/decrement counter',
            descriptionAr: 'عرض بسيط للـ reactivity مع عداد يمكن زيادته أو إنقاصه',
            tags: ['reactivity', 'beginner', 'htmlx', 'state'],
            code: `<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Counter Example</title>
  <script>window.MishkahAutoConfig = { css: 'mi' };</script>
  <script src="/lib/mishkah.js" data-htmlx></script>
</head>
<body>
  <div id="app"></div>

  <template id="counter">
    <script type="application/json" data-m-data data-m-path="data">
      { "count": 0 }
    <\/script>

    <div data-m-scope="counter" class="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-600 to-blue-600">
      <div class="bg-white rounded-3xl shadow-2xl p-12 text-center">
        <h1 class="text-4xl font-bold text-gray-800 mb-6">🔢 العداد</h1>
        <p class="text-8xl font-bold text-purple-600 mb-8">{state.data.count}</p>
        <div class="flex gap-4 justify-center">
          <button data-m-order="decrement" class="px-8 py-4 bg-red-500 text-white text-2xl font-bold rounded-2xl hover:bg-red-600 transition">-</button>
          <button data-m-order="reset" class="px-8 py-4 bg-gray-500 text-white text-lg font-bold rounded-2xl hover:bg-gray-600 transition">Reset</button>
          <button data-m-order="increment" class="px-8 py-4 bg-green-500 text-white text-2xl font-bold rounded-2xl hover:bg-green-600 transition">+</button>
        </div>
      </div>
    </div>
  </template>

  <script>
    MishkahAuto.ready(M => {
      M.app.make({}, {
        mount: '#app',
        orders: {
          increment: { on: ['click'], handler: (e, app) => app.setState(s => { s.data.count++; return s; }) },
          decrement: { on: ['click'], handler: (e, app) => app.setState(s => { s.data.count--; return s; }) },
          reset: { on: ['click'], handler: (e, app) => app.setState(s => { s.data.count = 0; return s; }) }
        }
      });
    });
  <\/script>
</body>
</html>`
        },

        'todo': {
            id: 'todo',
            title: 'Todo List',
            titleAr: 'قائمة المهام',
            category: 'basic',
            difficulty: 2,
            description: 'Todo list demonstrating x-for, x-if directives and list manipulation',
            descriptionAr: 'قائمة مهام توضح استخدام x-for و x-if والتعامل مع القوائم',
            tags: ['x-for', 'x-if', 'lists', 'forms'],
            code: `<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <title>Todo List</title>
  <script>window.MishkahAutoConfig = { css: 'mi' };</script>
  <script src="/lib/mishkah.js" data-htmlx></script>
</head>
<body>
  <div id="app"></div>
  <template id="todo">
    <script type="application/json" data-m-data data-m-path="data">
      {
        "todos": [
          { "id": 1, "text": "تعلم Mishkah", "done": true },
          { "id": 2, "text": "بناء تطبيق رائع", "done": false }
        ],
        "newTodo": ""
      }
    <\/script>
    <div data-m-scope="todo" class="min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 p-8">
      <div class="max-w-2xl mx-auto bg-white rounded-3xl shadow-2xl p-8">
        <h1 class="text-4xl font-bold text-gray-800 mb-8 text-center">✅ قائمة المهام</h1>
        <div class="flex gap-3 mb-8">
          <input type="text" placeholder="مهمة جديدة..." data-m-order="updateNewTodo" x-bind:value="state.data.newTodo" class="flex-1 px-4 py-3 border-2 border-purple-300 rounded-2xl focus:border-purple-500 focus:outline-none text-lg">
          <button data-m-order="addTodo" class="px-8 py-3 bg-purple-600 text-white font-bold rounded-2xl hover:bg-purple-700 transition">إضافة</button>
        </div>
        <div x-if="state.data.todos.length === 0" class="text-center py-12 text-gray-400">لا توجد مهام</div>
        <div x-if="state.data.todos.length > 0" class="space-y-3">
          <div x-for="todo in state.data.todos" key="todo.id" class="flex items-center gap-4 p-4 bg-gray-50 rounded-2xl border-2 border-gray-200">
            <input type="checkbox" x-bind:checked="todo.done" data-m-order="toggleTodo" x-bind:data-todo-id="todo.id" class="w-6 h-6 rounded-lg cursor-pointer">
            <span x-bind:class="todo.done ? 'flex-1 text-lg line-through text-gray-400' : 'flex-1 text-lg text-gray-800'">{todo.text}</span>
            <button data-m-order="deleteTodo" x-bind:data-todo-id="todo.id" class="px-4 py-2 bg-red-100 text-red-600 rounded-xl hover:bg-red-200">🗑️</button>
          </div>
        </div>
      </div>
    </div>
  </template>
  <script>
    MishkahAuto.ready(M => {
      M.app.make({}, {
        mount: '#app',
        orders: {
          updateNewTodo: { on: ['input'], handler: (e, app) => app.setState(s => { s.data.newTodo = e.target.value; return s; }) },
          addTodo: { on: ['click'], handler: (e, app) => app.setState(s => { var text = s.data.newTodo.trim(); if (!text) return s; var newId = s.data.todos.length > 0 ? Math.max(...s.data.todos.map(t => t.id)) + 1 : 1; s.data.todos.push({ id: newId, text, done: false }); s.data.newTodo = ''; return s; }) },
          toggleTodo: { on: ['change'], handler: (e, app) => { var todoId = parseInt(e.target.dataset.todoId); app.setState(s => { var todo = s.data.todos.find(t => t.id === todoId); if (todo) todo.done = !todo.done; return s; }); } },
          deleteTodo: { on: ['click'], handler: (e, app) => { var todoId = parseInt(e.target.dataset.todoId); app.setState(s => { s.data.todos = s.data.todos.filter(t => t.id !== todoId); return s; }); } }
        }
      });
    });
  <\/script>
</body>
</html>`
        },

        'theme': {
            id: 'theme',
            title: 'Theme Toggle',
            titleAr: 'تبديل السمة',
            category: 'basic',
            difficulty: 1,
            description: 'Dark/Light theme switcher demonstration',
            descriptionAr: 'عرض لكيفية تبديل السمة بين الداكنة والفاتحة',
            tags: ['theme', 'toggle', 'ui'],
            code: `<!DOCTYPE html>
<html lang="ar" dir="rtl" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <title>Theme Toggle</title>
  <script>window.MishkahAutoConfig = { css: 'mi' };</script>
  <script src="/lib/mishkah.js" data-htmlx></script>
  <style>
    :root[data-theme="dark"] { --bg: #0f172a; --fg: #e2e8f0; --card: #1e293b; }
    :root[data-theme="light"] { --bg: #f1f5f9; --fg: #0f172a; --card: #ffffff; }
    body { margin: 0; transition: background 0.3s; background: var(--bg); color: var(--fg); font-family: 'Cairo', sans-serif; }
  </style>
</head>
<body>
  <div id="app"></div>
  <template id="theme">
    <script type="application/json" data-m-env data-m-path="env">
      { "theme": "dark" }
    <\/script>
    <div data-m-scope="theme" class="min-h-screen flex items-center justify-center p-8">
      <div class="max-w-2xl w-full" style="background: var(--card); border-radius: 2rem; padding: 3rem; box-shadow: 0 25px 50px rgba(0,0,0,0.3);">
        <h1 class="text-5xl font-bold mb-6 text-center" style="color: var(--accent, #667eea);">
          <span x-if="state.env.theme === 'dark'">🌙</span>
          <span x-if="state.env.theme === 'light'">☀️</span>
          تبديل السمة
        </h1>
        <p class="text-xl text-center mb-8">
          <span x-if="state.env.theme === 'dark'">الوضع الليلي مفعّل</span>
          <span x-if="state.env.theme === 'light'">الوضع النهاري مفعّل</span>
        </p>
        <div class="flex gap-4 justify-center">
          <button data-m-order="setDark" class="px-8 py-4 rounded-2xl font-bold text-lg transition">🌙 ليلي</button>
          <button data-m-order="setLight" class="px-8 py-4 rounded-2xl font-bold text-lg transition">☀️ نهاري</button>
        </div>
      </div>
    </div>
  </template>
  <script>
    MishkahAuto.ready(M => {
      M.app.make({}, {
        mount: '#app',
        orders: {
          setDark: { on: ['click'], handler: (e, app) => { app.setState(s => { s.env.theme = 'dark'; document.documentElement.setAttribute('data-theme', 'dark'); return s; }); } },
          setLight: { on: ['click'], handler: (e, app) => { app.setState(s => { s.env.theme = 'light'; document.documentElement.setAttribute('data-theme', 'light'); return s; }); } }
        }
      });
    });
  <\/script>
</body>
</html>`
        },

        'lang': {
            id: 'lang',
            title: 'Language Switcher',
            titleAr: 'تبديل اللغة',
            category: 'basic',
            difficulty: 1,
            description: 'i18n demonstration with Arabic/English switching',
            descriptionAr: 'عرض لتعدد اللغات مع التبديل بين العربية والإنجليزية',
            tags: ['i18n', 'localization', 'multilingual'],
            code: `<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <title>Language Switcher</title>
  <script>window.MishkahAutoConfig = { css: 'mi' };</script>
  <script src="/lib/mishkah.js" data-htmlx></script>
</head>
<body>
  <div id="app"></div>
  <template id="i18n">
    <script type="application/json" data-m-env data-m-path="env">
      { "lang": "ar", "dir": "rtl" }
    <\/script>
    <script type="application/json" data-m-data data-m-path="i18n.strings">
      {
        "title": { "ar": "تبديل اللغة", "en": "Language Switcher" },
        "welcome": { "ar": "مرحباً بك في Mishkah!", "en": "Welcome to Mishkah!" },
        "currentLang": { "ar": "اللغة الحالية: العربية", "en": "Current language: English" }
      }
    <\/script>
    <div data-m-scope="i18n" x-bind:lang="state.env.lang" x-bind:dir="state.env.dir" class="min-h-screen bg-gradient-to-br from-indigo-600 to-purple-700 p-8">
      <div class="max-w-3xl mx-auto">
        <div class="flex justify-center gap-3 mb-12">
          <button data-m-order="setArabic" class="px-6 py-3 rounded-2xl font-bold transition bg-white text-indigo-600 shadow-lg">🇸🇦 العربية</button>
          <button data-m-order="setEnglish" class="px-6 py-3 rounded-2xl font-bold transition bg-indigo-500 text-white hover:bg-indigo-400">🇬🇧 English</button>
        </div>
        <div class="bg-white rounded-3xl shadow-2xl p-12">
          <h1 class="text-5xl font-bold text-indigo-600 mb-6 text-center">🌍 {i18n('title')}</h1>
          <p class="text-3xl text-gray-800 text-center mb-4">{i18n('welcome')}</p>
          <div class="text-center p-4 bg-gray-100 rounded-2xl text-gray-600">{i18n('currentLang')}</div>
        </div>
      </div>
    </div>
  </template>
  <script>
    MishkahAuto.ready(M => {
      M.app.make({}, {
        mount: '#app',
        orders: {
          setArabic: { on: ['click'], handler: (e, app) => app.setState(s => { s.env.lang = 'ar'; s.env.dir = 'rtl'; return s; }) },
          setEnglish: { on: ['click'], handler: (e, app) => app.setState(s => { s.env.lang = 'en'; s.env.dir = 'ltr'; return s; }) }
        }
      });
    });
  <\/script>
</body>
</html>`
        }

    };

    // ============================================================
    // Categories
    // ============================================================

    var CATEGORIES = [
        { id: 'basic', name: 'Basic', nameAr: 'أساسيات', icon: '⭐' },
        { id: 'forms', name: 'Forms', nameAr: 'نماذج', icon: '📝' },
        { id: 'ui', name: 'UI Components', nameAr: 'مكونات واجهة', icon: '🎨' },
        { id: 'charts', name: 'Charts', nameAr: 'رسوم بيانية', icon: '📊' },
        { id: 'tables', name: 'Tables', nameAr: 'جداول', icon: '📋' },
        { id: 'advanced', name: 'Advanced', nameAr: 'متقدم', icon: '🚀' },
        { id: 'pwa', name: 'PWA', nameAr: 'تطبيقات ويب', icon: '📱' },
        { id: 'mobile', name: 'Mobile UI', nameAr: 'واجهة موبايل', icon: '📲' }
    ];

    // ============================================================
    // Helper Functions
    // ============================================================

    function getExample(id) {
        return EXAMPLES[id] || null;
    }

    function getExamplesByCategory(categoryId) {
        var results = [];
        for (var id in EXAMPLES) {
            if (EXAMPLES[id].category === categoryId) {
                results.push(EXAMPLES[id]);
            }
        }
        return results;
    }

    function getAllExamples() {
        var results = [];
        for (var id in EXAMPLES) {
            results.push(EXAMPLES[id]);
        }
        return results;
    }

    function searchExamples(query) {
        if (!query) return getAllExamples();

        var q = query.toLowerCase();
        var results = [];

        for (var id in EXAMPLES) {
            var ex = EXAMPLES[id];
            if (ex.title.toLowerCase().indexOf(q) !== -1 ||
                ex.titleAr.indexOf(q) !== -1 ||
                ex.description.toLowerCase().indexOf(q) !== -1 ||
                ex.tags.some(function (tag) { return tag.indexOf(q) !== -1; })) {
                results.push(ex);
            }
        }

        return results;
    }

    function getCategoriesWithCount() {
        var counts = {};
        for (var id in EXAMPLES) {
            var cat = EXAMPLES[id].category;
            counts[cat] = (counts[cat] || 0) + 1;
        }

        return CATEGORIES.map(function (cat) {
            return Object.assign({}, cat, { count: counts[cat.id] || 0 });
        });
    }

    // ============================================================
    // Export
    // ============================================================

    global.PlaygroundExamples = {
        getExample: getExample,
        getExamplesByCategory: getExamplesByCategory,
        getAllExamples: getAllExamples,
        searchExamples: searchExamples,
        getCategories: getCategoriesWithCount,
        EXAMPLES: EXAMPLES,
        CATEGORIES: CATEGORIES
    };

})(typeof window !== 'undefined' ? window : this);
